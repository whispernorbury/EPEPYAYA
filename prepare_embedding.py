#!/usr/bin/env python3
"""
prepare_embedding.py
Usage:
    python prepare_embedding.py
Note: 모델은 자동으로 다운로드되어 로컬에서 실행됩니다 (API 키 불필요)
"""

import json
import sys
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PHRASES_FILE = "./phrases.json"
VECTORS_FILE = "./vectors.json"
# 한국어에 강한 모델 추천: jhgan/ko-sroberta-multitask
# 다른 옵션: snunlp/KR-SBERT-V40K-klueNLI-augSTS, BM-K/KoSimCSE-roberta-multitask
MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")

# 전역 모델 인스턴스 (한 번만 로드)
embedding_model = None


def load_phrases():
    with open(PHRASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print(f"Loading model: {MODEL}")
        try:
            embedding_model = SentenceTransformer(MODEL)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model {MODEL}: {str(e)}")
            if "onnx" in str(e).lower() or "could not locate" in str(e).lower():
                print("\n이 모델은 현재 라이브러리와 호환되지 않을 수 있습니다.")
                print("환경 변수 EMBEDDING_MODEL을 호환되는 모델로 변경하세요.")
                print("추천 모델:")
                print("  - jhgan/ko-sroberta-multitask (한국어 특화, 추천)")
                print("  - snunlp/KR-SBERT-V40K-klueNLI-augSTS")
                print("  - BM-K/KoSimCSE-roberta-multitask")
            raise
    return embedding_model


def create_embedding(text):
    try:
        model = get_embedding_model()
        text = "passage: " + text
        # sentence-transformers는 자동으로 정규화된 벡터를 반환
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        raise


def main():
    phrases = load_phrases()
    print(f"Loaded {len(phrases)} phrases")
    
    print(f"Using local model: {MODEL}")
    print("Note: Model will be downloaded automatically on first run and cached locally.")
    
    out = []
    for i, p in enumerate(phrases):
        print(f"Embedding {i + 1}/{len(phrases)}: {p['id']}", end="\r")
        try:
            vector = create_embedding(p["text"])
            out.append({
                "id": p["id"],
                "text": p["text"],
                "trans": p.get("trans"),
                "vector": vector
            })
        except Exception as err:
            print(f"\nEmbedding error for {p['id']}: {err}")
            print("Retrying in 5 seconds...")
            import time
            time.sleep(5)
            # 재시도
            try:
                vector = create_embedding(p["text"])
                out.append({
                    "id": p["id"],
                    "text": p["text"],
                    "trans": p.get("trans"),
                    "vector": vector
                })
            except Exception as retry_err:
                print(f"Retry failed for {p['id']}: {retry_err}")
                raise
    
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {VECTORS_FILE} with {len(out)} vectors.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

