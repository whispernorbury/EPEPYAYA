#!/usr/bin/env python3
"""
embedding_service.py
독립적인 HTTP 서버로 실행되는 Python 임베딩 서비스
모델을 한 번만 로드하고 재사용하여 효율성 향상
"""

import json
import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
PORT = int(os.getenv("EMBEDDING_PORT", "5000"))

# 전역 모델 인스턴스 (한 번만 로드)
embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {MODEL}")
        try:
            embedding_model = SentenceTransformer(MODEL)
            print(f"Model {MODEL} loaded successfully!")
        except Exception as e:
            print(f"Failed to load model {MODEL}: {str(e)}")
            raise
    return embedding_model


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model": MODEL})


@app.route("/embed", methods=["POST"])
def embed():
    """임베딩 생성 엔드포인트"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data["text"]
        if not isinstance(text, str) or len(text) == 0:
            return jsonify({"error": "Invalid text"}), 400
        text = "query: " + text
        # 모델 로드 및 임베딩 생성
        model = get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        
        return jsonify({"vector": embedding.tolist()})
    
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 모델 미리 로드
    get_embedding_model()
    print(f"Embedding service starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
