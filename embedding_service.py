"""
BGE-M3 임베딩 서비스
FastAPI를 사용하여 bge-m3 모델을 제공하는 마이크로서비스
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from FlagEmbedding import FlagModel

app = FastAPI(title="BGE-M3 Embedding Service")

# 환경 변수
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# 모델 로딩 (전역 변수로 한 번만 로드)
model: Optional[FlagModel] = None


class EmbeddingRequest(BaseModel):
    text: str
    normalize: Optional[bool] = True


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    dim: int


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


def load_model():
    """모델을 로드합니다."""
    global model
    if model is None:
        print(f"Loading BGE-M3 model: {MODEL_NAME}")
        try:
            model = FlagModel(
                MODEL_NAME,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=USE_FP16
            )
            print(f"Model {MODEL_NAME} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return model


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델을 로드합니다."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        print("Model will be loaded on first request")


@app.get("/health", response_model=HealthResponse)
async def health():
    """헬스 체크 엔드포인트"""
    global model
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": model is not None
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    텍스트를 임베딩 벡터로 변환합니다.
    
    Args:
        request: EmbeddingRequest 객체 (text 필수, normalize 선택)
    
    Returns:
        EmbeddingResponse: 임베딩 벡터와 메타데이터
    """
    global model
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")
    
    # 모델이 로드되지 않았으면 로드 시도
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    try:
        # bge-m3는 encode 메서드를 사용
        # normalize_embeddings 파라미터는 버전 호환성 문제로 제거하고 수동 정규화
        embedding = model.encode([request.text])[0]
        
        # numpy array로 변환
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # 수동으로 L2 정규화 (normalize 옵션이 True인 경우)
        if request.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # numpy array를 list로 변환
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        return EmbeddingResponse(
            embedding=embedding_list,
            model=MODEL_NAME,
            dim=len(embedding_list)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )


@app.post("/embed/batch", response_model=List[EmbeddingResponse])
async def create_embeddings_batch(texts: List[str], normalize: Optional[bool] = True):
    """
    여러 텍스트를 한 번에 임베딩 벡터로 변환합니다.
    
    Args:
        texts: 텍스트 리스트
        normalize: 정규화 여부
    
    Returns:
        EmbeddingResponse 리스트
    """
    global model
    
    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Too many texts (max 100)")
    
    # 모델이 로드되지 않았으면 로드 시도
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    try:
        embeddings = model.encode(texts)
        
        results = []
        for embedding in embeddings:
            # numpy array로 변환
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # 수동으로 L2 정규화 (normalize 옵션이 True인 경우)
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            results.append(EmbeddingResponse(
                embedding=embedding_list,
                model=MODEL_NAME,
                dim=len(embedding_list)
            ))
        
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch embedding generation failed: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("EMBEDDING_SERVICE_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
