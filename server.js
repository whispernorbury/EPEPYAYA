// server.js
import fs from "fs/promises";
import dotenv from "dotenv";
dotenv.config();

import Fastify from "fastify";
import { WebSocketServer } from "ws";
import Redis from "ioredis";
import { OpenAI } from "openai";
import { pipeline } from "@xenova/transformers";

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;
const VECTORS_FILE = "./vectors.json";
const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";
const CACHE_TTL = parseInt(process.env.CACHE_TTL_SECONDS || "3600");
const MIN_SCORE_THRESHOLD = parseFloat(process.env.MIN_SCORE_THRESHOLD || "0.75");
const QUANTIZE_DECIMALS = parseInt(process.env.QUANTIZE_DECIMALS || "2");
const QUANTIZE_DIM_PREFIX = parseInt(process.env.QUANTIZE_DIM_PREFIX || "8");

const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "Xenova/bge-m3";
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini"; // change to your finetuned model if any

// 전역 모델 인스턴스 (한 번만 로드)
let embeddingPipeline = null;

// utils
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  return Math.sqrt(dot(a, a));
}
function cosine(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-12);
}
function quantizeKey(vec) {
  // simple quantization: round first N dims to D decimals
  const parts = [];
  for (let i = 0; i < Math.min(QUANTIZE_DIM_PREFIX, vec.length); i++) {
    parts.push(vec[i].toFixed(QUANTIZE_DECIMALS));
  }
  return parts.join(",");
}

async function loadVectors() {
  const raw = await fs.readFile(VECTORS_FILE, "utf8");
  return JSON.parse(raw);
}

async function getEmbeddingPipeline() {
  if (embeddingPipeline === null) {
    console.log(`Loading embedding model: ${EMBEDDING_MODEL}`);
    try {
      // BGE-m3는 양자화된 ONNX 모델이 없으므로 quantized: false 옵션 사용
      embeddingPipeline = await pipeline('feature-extraction', EMBEDDING_MODEL, {
        quantized: false
      });
      console.log(`Model ${EMBEDDING_MODEL} loaded successfully!`);
    } catch (error) {
      console.error(`Failed to load model ${EMBEDDING_MODEL}:`, error);
      throw error;
    }
  }
  return embeddingPipeline;
}

async function createEmbedding(text) {
  try {
    const pipe = await getEmbeddingPipeline();
    
    // BGE-m3 모델로 임베딩 생성
    const output = await pipe(text, {
      pooling: 'mean',
      normalize: true
    });
    
    // output은 텐서 객체이므로 배열로 변환
    // output.data가 Float32Array인 경우와 output이 직접 배열인 경우 모두 처리
    let embedding;
    if (output.data) {
      embedding = Array.from(output.data);
    } else if (Array.isArray(output)) {
      embedding = output;
    } else if (output.tolist) {
      embedding = output.tolist();
    } else {
      // 텐서를 직접 배열로 변환 시도
      embedding = Array.from(output);
    }
    
    return embedding;
  } catch (error) {
    console.error("Embedding error:", error);
    throw new Error(`Failed to create embedding: ${error.message}`);
  }
}

async function callLLMFallback(prompt) {
  if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY not set");
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: prompt }],
    max_tokens: 200
  });
  return resp.choices?.[0]?.message?.content ?? "";
}

(async () => {
  console.log("Loading embedding model...");
  await getEmbeddingPipeline(); // 모델 미리 로드
  console.log("Loading vectors...");
  const vectors = await loadVectors();
  console.log(`Loaded ${vectors.length} vectors into memory.`);

  const redis = new Redis(REDIS_URL);

  const fastify = Fastify({ logger: true });
  const server = fastify.server;

  // Health check endpoint
  fastify.get('/health', async (request, reply) => {
    const health = {
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      vectors: vectors.length,
      redis: 'unknown',
      version: process.env.npm_package_version || '1.0.0'
    };
    
    try {
      await redis.ping();
      health.redis = 'connected';
    } catch (e) {
      health.redis = 'disconnected';
      health.status = 'degraded';
      health.error = e.message;
    }
    
    const statusCode = health.status === 'ok' ? 200 : 503;
    return reply.code(statusCode).send(health);
  });

  const wss = new WebSocketServer({ server });

  wss.on("connection", (ws) => {
    ws.on("message", async (message) => {
      // Expect message JSON: {type:"query", text:"...", lang:"ko", k:5}
      let msg;
      try { msg = JSON.parse(message.toString()); } catch (e) {
        ws.send(JSON.stringify({ type: "error", error: "invalid_json" }));
        return;
      }

      if (msg.type !== "query" || !msg.text) {
        ws.send(JSON.stringify({ type: "error", error: "invalid_payload" }));
        return;
      }

      const queryText = String(msg.text);
      const k = msg.k ? parseInt(msg.k) : 5;
      const mode = msg.mode || "search"; // "search" or "llm_fallback"
      
      // 입력 검증
      if (queryText.length > 1000) {
        ws.send(JSON.stringify({ type: "error", error: "text_too_long", details: "Maximum text length is 1000 characters" }));
        return;
      }
      if (k < 1 || k > 20) {
        ws.send(JSON.stringify({ type: "error", error: "invalid_k", details: "k must be between 1 and 20" }));
        return;
      }

      // 1) Compute embedding (server-side)
      let qVec;
      try {
        qVec = await createEmbedding(queryText);
      } catch (e) {
        ws.send(JSON.stringify({ type: "error", error: "embedding_failed", details: String(e) }));
        return;
      }

      // 2) Check Redis cache by quantized key
      const qKey = `semcache:${quantizeKey(qVec)}`;
      try {
        const cached = await redis.get(qKey);
        if (cached) {
          const cachedResult = JSON.parse(cached);
          // 캐시에 source 정보가 포함되어 있으면 사용, 없으면 fallback 속성으로 판단 (하위 호환성)
          const source = cachedResult.source || (cachedResult.fallback ? "llm" : "search");
          const data = cachedResult.data || cachedResult; // data 속성이 있으면 사용, 없으면 전체를 data로
          ws.send(JSON.stringify({ type: "result", source: source, data: data }));
          return;
        }
      } catch (e) {
        // continue if redis fails
        fastify.log.warn("Redis get failed: " + e);
      }

      // 3) Brute force cosine over vectors (small DB) - produce top k
      const results = [];
      for (const v of vectors) {
        const sc = cosine(qVec, v.vector);
        results.push({ 
          id: v.id, 
          text: v.text, 
          trans: v.trans || null,  // trans가 없으면 null로 설정
          tags: v.tags || [], 
          score: sc 
        });
      }
      results.sort((a, b) => b.score - a.score);
      const topK = results.slice(0, k);

      // 4) Mode에 따라 처리
      if (mode === "search") {
        // search 모드: DB 검색만 수행, 자동 fallback 없음
        // 5) Cache the topK result
        try {
          await redis.set(qKey, JSON.stringify({ source: "search", data: topK }), "EX", CACHE_TTL);
        } catch (e) {
          fastify.log.warn("Redis set failed: " + e);
        }

        // 6) Send results via WebSocket
        ws.send(JSON.stringify({ type: "result", source: "search", data: topK }));
      } else if (mode === "llm_fallback") {
        // llm_fallback 모드: LLM fallback 로직은 나중에 구현
        // TODO: LLM fallback 로직 구현
        ws.send(JSON.stringify({ type: "error", error: "llm_fallback_mode_not_implemented", message: "LLM fallback mode is not yet implemented" }));
      } else {
        ws.send(JSON.stringify({ type: "error", error: "invalid_mode", message: `Invalid mode: ${mode}. Use "search" or "llm_fallback"` }));
      }
    });

    ws.send(JSON.stringify({ type: "welcome", msg: "connected" }));
  });

  await fastify.listen({ port: PORT, host: "0.0.0.0" });
  console.log(`Server listening at http://0.0.0.0:${PORT}`);
})();
