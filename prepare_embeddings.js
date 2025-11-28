#!/usr/bin/env node
/**
 * prepare_embeddings.js
 * phrases.json을 읽어서 Xenova/bge-m3 모델로 임베딩을 생성하고 vectors.json을 만드는 스크립트
 * 
 * Usage:
 *   node prepare_embeddings.js
 */

import fs from "fs/promises";
import { pipeline } from "@xenova/transformers";
import dotenv from "dotenv";

dotenv.config();

const PHRASES_FILE = "./phrases.json";
const VECTORS_FILE = "./vectors.json";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "Xenova/bge-m3";

// 전역 모델 인스턴스 (한 번만 로드)
let embeddingPipeline = null;

async function loadPhrases() {
  const raw = await fs.readFile(PHRASES_FILE, "utf8");
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
    throw error;
  }
}

async function main() {
  console.log("Loading phrases...");
  const phrases = await loadPhrases();
  console.log(`Loaded ${phrases.length} phrases`);
  
  console.log(`Using embedding model: ${EMBEDDING_MODEL}`);
  console.log("Note: Model will be downloaded automatically on first run and cached locally.");
  
  // 모델 미리 로드
  await getEmbeddingPipeline();
  
  const out = [];
  for (let i = 0; i < phrases.length; i++) {
    const p = phrases[i];
    process.stdout.write(`\rEmbedding ${i + 1}/${phrases.length}: ${p.id}`);
    
    try {
      const vector = await createEmbedding(p.text);
      out.push({
        id: p.id,
        text: p.text,
        trans: p.trans || null,
        tags: p.tags || [],
        vector: vector
      });
    } catch (err) {
      console.error(`\nEmbedding error for ${p.id}: ${err}`);
      console.log("Retrying in 5 seconds...");
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      // 재시도
      try {
        const vector = await createEmbedding(p.text);
        out.push({
          id: p.id,
          text: p.text,
          trans: p.trans || null,
          tags: p.tags || [],
          vector: vector
        });
      } catch (retryErr) {
        console.error(`Retry failed for ${p.id}: ${retryErr}`);
        throw retryErr;
      }
    }
  }
  
  console.log(`\nWriting ${VECTORS_FILE} with ${out.length} vectors...`);
  await fs.writeFile(VECTORS_FILE, JSON.stringify(out, null, 2), "utf8");
  console.log(`Done! Created ${VECTORS_FILE} with ${out.length} vectors.`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});

