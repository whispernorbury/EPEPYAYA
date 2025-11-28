// server.js
import fs from "fs/promises";
import dotenv from "dotenv";
dotenv.config();

import Fastify from "fastify";
import { WebSocketServer } from "ws";
import Redis from "ioredis";

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;
const VECTORS_FILE = "./vectors.json";
const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";
const CACHE_TTL = parseInt(process.env.CACHE_TTL_SECONDS || "3600");
const QUANTIZE_DECIMALS = parseInt(process.env.QUANTIZE_DECIMALS || "2");
const QUANTIZE_DIM_PREFIX = parseInt(process.env.QUANTIZE_DIM_PREFIX || "8");

const EMBEDDING_SERVICE_URL = process.env.EMBEDDING_SERVICE_URL || "http://embedding-service:8000";

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

async function createEmbedding(text) {
  try {
    // Python 임베딩 서비스 호출
    const response = await fetch(`${EMBEDDING_SERVICE_URL}/embed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Embedding service error: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.error("Embedding error:", error);
    throw new Error(`Failed to create embedding: ${error.message}`);
  }
}

(async () => {
  // Python 임베딩 서비스 헬스 체크
  console.log(`Checking embedding service at ${EMBEDDING_SERVICE_URL}...`);
  try {
    const healthResponse = await fetch(`${EMBEDDING_SERVICE_URL}/health`);
    if (healthResponse.ok) {
      const health = await healthResponse.json();
      console.log(`Embedding service is ready: ${health.model}`);
    } else {
      console.warn(`Embedding service health check failed: ${healthResponse.status}`);
    }
  } catch (error) {
    console.warn(`Could not connect to embedding service: ${error.message}`);
    console.warn("Server will start but embedding requests may fail until service is ready.");
  }

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
      embedding_service: 'unknown',
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

    // 임베딩 서비스 헬스 체크
    try {
      const embeddingHealth = await fetch(`${EMBEDDING_SERVICE_URL}/health`);
      if (embeddingHealth.ok) {
        const embeddingData = await embeddingHealth.json();
        health.embedding_service = embeddingData.model_loaded ? 'ready' : 'not_loaded';
      } else {
        health.embedding_service = 'unavailable';
        health.status = 'degraded';
      }
    } catch (e) {
      health.embedding_service = 'unavailable';
      health.status = 'degraded';
    }
    
    const statusCode = health.status === 'ok' ? 200 : 503;
    return reply.code(statusCode).send(health);
  });

  const wss = new WebSocketServer({ server });

  wss.on("connection", (ws) => {
    ws.on("message", async (message) => {
      // Expect message JSON: {type:"query", text:"...", k:5}
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

      // 4) Cache the topK result
      try {
        await redis.set(qKey, JSON.stringify({ source: "search", data: topK }), "EX", CACHE_TTL);
      } catch (e) {
        fastify.log.warn("Redis set failed: " + e);
      }

      // 5) Send results via WebSocket
      ws.send(JSON.stringify({ type: "result", source: "search", data: topK }));
    });

    ws.send(JSON.stringify({ type: "welcome", msg: "connected" }));
  });

  await fastify.listen({ port: PORT, host: "0.0.0.0" });
  console.log(`Server listening at http://0.0.0.0:${PORT}`);
})();
