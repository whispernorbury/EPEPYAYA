"""
phrases.json을 읽어서 각 문구의 임베딩 벡터를 생성하고 vectors.json으로 저장합니다.
"""
import json
import os
import numpy as np
from typing import List, Dict, Any
from FlagEmbedding import FlagModel
from tqdm import tqdm

# 환경 변수
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
PHRASES_FILE = os.getenv("PHRASES_FILE", "./phrases.json")
VECTORS_FILE = os.getenv("VECTORS_FILE", "./vectors.json")
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


def load_phrases(file_path: str) -> List[Dict[str, Any]]:
    """phrases.json 파일을 로드합니다."""
    print(f"Loading phrases from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        phrases = json.load(f)
    print(f"Loaded {len(phrases)} phrases")
    return phrases


def create_embeddings(phrases: List[Dict[str, Any]], model: FlagModel) -> List[Dict[str, Any]]:
    """각 문구에 대한 임베딩 벡터를 생성합니다."""
    vectors = []
    texts = [phrase["text"] for phrase in phrases]
    
    print(f"Creating embeddings for {len(texts)} phrases...")
    
    # 배치 처리로 임베딩 생성
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_phrases = phrases[i:i + BATCH_SIZE]
        
        # 임베딩 생성
        embeddings = model.encode(batch_texts)
        
        # 수동으로 정규화 (normalize_embeddings 파라미터가 버전 호환성 문제로 제거됨)
        if isinstance(embeddings, np.ndarray):
            # L2 정규화
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-12)
        else:
            # 리스트인 경우 numpy로 변환 후 정규화
            embeddings = np.array(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-12)
        
        # 결과 저장
        for phrase, embedding in zip(batch_phrases, embeddings):
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            vectors.append({
                "id": phrase["id"],
                "text": phrase["text"],
                "trans": phrase.get("trans"),
                "vector": embedding_list
            })
    
    return vectors


def save_vectors(vectors: List[Dict[str, Any]], file_path: str):
    """벡터를 JSON 파일로 저장합니다."""
    print(f"Saving {len(vectors)} vectors to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)
    print(f"Saved vectors to {file_path}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("BGE-M3 Embedding Preparation Script")
    print("=" * 60)
    
    # 모델 로드
    print(f"\nLoading model: {MODEL_NAME}")
    print("This may take a few minutes on first run...")
    try:
        model = FlagModel(
            MODEL_NAME,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=USE_FP16
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have:")
        print("1. Installed FlagEmbedding: pip install FlagEmbedding")
        print("2. Sufficient disk space for model download")
        print("3. Internet connection for first-time model download")
        return
    
    # phrases 로드
    if not os.path.exists(PHRASES_FILE):
        print(f"Error: {PHRASES_FILE} not found")
        return
    
    phrases = load_phrases(PHRASES_FILE)
    
    if not phrases:
        print("Error: No phrases found in file")
        return
    
    # 임베딩 생성
    vectors = create_embeddings(phrases, model)
    
    # 저장
    save_vectors(vectors, VECTORS_FILE)
    
    print("\n" + "=" * 60)
    print("Embedding preparation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
