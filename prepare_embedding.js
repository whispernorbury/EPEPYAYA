// prepare_embeddings.js
// Usage:
// 1) set OPENAI_API_KEY in .env or env
// 2) node prepare_embeddings.js
// Optional: node prepare_embeddings.js --mock  (creates random vectors for dev)

import fs from "fs/promises";
import dotenv from "dotenv";
dotenv.config();
import { OpenAI } from "openai";

const PHRASES_FILE = "./phrases.json";
const VECTORS_FILE = "./vectors.json";
const MODEL = process.env.OPENAI_MODEL || "text-embedding-3-small";

const argv = process.argv.slice(2);
const MOCK = argv.includes("--mock");

async function loadPhrases() {
  const raw = await fs.readFile(PHRASES_FILE, "utf8");
  return JSON.parse(raw);
}

function randomVector(dim) {
  return Array.from({ length: dim }, () => (Math.random() - 0.5));
}

async function main() {
  const phrases = await loadPhrases();
  console.log(`Loaded ${phrases.length} phrases`);

  const out = [];
  if (MOCK || !process.env.OPENAI_API_KEY) {
    console.log("Creating MOCK vectors (no OpenAI key found or --mock used).");
    for (const p of phrases) {
      out.push({ id: p.id, text: p.text, trans: p.trans, tags: p.tags, vector: randomVector(1536) });
    }
    await fs.writeFile(VECTORS_FILE, JSON.stringify(out, null, 2), "utf8");
    console.log(`Wrote ${VECTORS_FILE} with mock vectors.`);
    return;
  }

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  for (let i = 0; i < phrases.length; i++) {
    const p = phrases[i];
    process.stdout.write(`Embedding ${i + 1}/${phrases.length}: ${p.id}\r`);
    try {
      const resp = await client.embeddings.create({
        model: MODEL,
        input: p.text
      });

      const vector = resp.data[0].embedding;
      out.push({ id: p.id, text: p.text, trans: p.trans, tags: p.tags, vector });
      // be polite if many requests - tiny sleep (optional)
      await new Promise(r => setTimeout(r, 50));
    } catch (err) {
      console.error("Embedding error for", p.id, err);
      process.exit(1);
    }
  }

  await fs.writeFile(VECTORS_FILE, JSON.stringify(out, null, 2), "utf8");
  console.log(`\nWrote ${VECTORS_FILE} with ${out.length} vectors.`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
