# 🚀 배포 가이드

## 빠른 시작

### 1. 환경 변수 설정

```bash
cp env.production.example .env
# .env 파일을 열어서 실제 값으로 변경
```

**필수 설정:**
- `REDIS_PASSWORD` - 강력한 비밀번호 설정
- `REDIS_URL` - 비밀번호 포함한 Redis URL (예: `redis://:password@redis:6379`)
- `OPENAI_API_KEY` - OpenAI API 키 (LLM fallback 사용 시)

### 2. 벡터 파일 생성

```bash
# 로컬에서 실행
npm install
pip3 install -r requirements.txt
python3 prepare_embedding.py
```

### 3. Docker Compose로 배포

```bash
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 상태 확인
docker-compose ps
```

### 로컬 개발 환경 (선택사항)

로컬에서 직접 실행하려면:

```bash
# 1. Embedding 서비스 실행 (별도 터미널)
python3 embedding_service.py

# 2. Redis 실행
docker-compose up redis -d

# 3. Node.js 서버 실행
# .env 파일에 EMBEDDING_SERVICE_URL=http://localhost:5000 설정
npm start
```

---

## 환경 변수

### 필수 변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `REDIS_PASSWORD` | Redis 비밀번호 | `strong-password-123` |
| `REDIS_URL` | Redis 연결 URL | `redis://:password@redis:6379` |

### 선택 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `PORT` | 서버 포트 | `3000` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| `EMBEDDING_SERVICE_URL` | Embedding 서비스 URL | `http://embedding:5000` |
| `OPENAI_API_KEY` | OpenAI API 키 | - |
| `CHAT_MODEL` | OpenAI 채팅 모델 | `gpt-4o-mini` |
| `CACHE_TTL_SECONDS` | 캐시 TTL (초) | `3600` |
| `MIN_SCORE_THRESHOLD` | 최소 유사도 임계값 | `0.75` |

---

## 배포 전 체크리스트

- [ ] `.env` 파일 생성 및 필수 값 설정
- [ ] `REDIS_PASSWORD` 설정 (강력한 비밀번호)
- [ ] `vectors.json` 파일 존재 확인
- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [ ] Docker 및 Docker Compose 설치 확인

---

## 서비스 구조

```
┌─────────────┐      HTTP     ┌──────────────┐
│  Node.js    │ ────────────> │   Python     │
│   Server    │               │  Embedding   │
│  (Port 3000)│               │  (Port 5000) │
└──────┬──────┘               └──────────────┘
       │
       │ Redis
       ▼
┌─────────────┐
│    Redis    │
│  (Port 6379)│
└─────────────┘
```

---

## 문제 해결

### 서버가 시작되지 않을 때

```bash
# 로그 확인
docker-compose logs server
docker-compose logs embedding

# 환경 변수 확인
docker-compose exec server env | grep -E "REDIS|EMBEDDING"
```

### Redis 연결 실패

- `.env` 파일에 `REDIS_PASSWORD`가 설정되어 있는지 확인
- `REDIS_URL`에 비밀번호가 포함되어 있는지 확인
- Redis 컨테이너가 실행 중인지 확인: `docker-compose ps redis`

### Embedding 서비스 연결 실패

- Embedding 서비스가 실행 중인지 확인: `docker-compose ps embedding`
- Health check: `docker-compose exec embedding wget -qO- http://localhost:5000/health`
- Node.js에서 연결 테스트: `docker-compose exec server wget -qO- http://embedding:5000/health`

### Health Check

```bash
# Node.js 서버
curl http://localhost:3000/health

# Embedding 서비스
curl http://localhost:5000/health
```

---

## 유지보수

### 로그 확인

```bash
# 모든 서비스 로그
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f server
docker-compose logs -f embedding
docker-compose logs -f redis
```

### 서비스 재시작

```bash
# 특정 서비스만 재시작
docker-compose restart server

# 모든 서비스 재시작
docker-compose restart
```

### 서비스 중지

```bash
# 서비스 중지 (데이터 유지)
docker-compose stop

# 서비스 중지 및 컨테이너 제거
docker-compose down

# 볼륨까지 제거 (주의!)
docker-compose down -v
```

### 벡터 파일 업데이트

```bash
# 1. 로컬에서 새 벡터 파일 생성
python3 prepare_embedding.py

# 2. 서비스 재시작 (볼륨 마운트로 자동 반영)
docker-compose restart server
```

---

## 보안 권장사항

- ✅ `.env` 파일을 Git에 커밋하지 마세요
- ✅ Redis 비밀번호를 강력하게 설정하세요
- ✅ 프로덕션에서는 Redis 포트를 외부에 노출하지 마세요
- ✅ HTTPS/WSS 사용을 권장합니다 (Nginx, Cloudflare 등)
- ⚠️ WebSocket 인증 구현 권장 (현재 미구현)
- ⚠️ Rate limiting 구현 권장 (현재 미구현)

---

## 성능 최적화

### 벡터 검색

현재는 brute-force 검색을 사용합니다. 벡터가 많아지면 (1000개 이상) 다음을 고려하세요:

- FAISS 라이브러리 도입
- Milvus/Pinecone 같은 벡터 DB 사용
- 인덱스 기반 검색

### 리소스 제한

`docker-compose.yml`에 리소스 제한을 추가할 수 있습니다:

```yaml
services:
  server:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

---

## 모니터링

### 기본 모니터링

```bash
# 컨테이너 리소스 사용량
docker stats

# 서비스 상태
docker-compose ps

# Health check
curl http://localhost:3000/health
```

### 로그 모니터링

```bash
# 실시간 로그
docker-compose logs -f

# 최근 로그만
docker-compose logs --tail=100
```

---

## 백업

### Redis 데이터 백업

```bash
# Redis 데이터 백업
docker-compose exec redis redis-cli --rdb /data/dump.rdb

# 볼륨 백업
docker run --rm -v epepyaya_redisdata:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data
```

### 벡터 파일 백업

```bash
# 벡터 파일은 이미 호스트에 있으므로 별도 백업 불필요
# 필요시 복사
cp vectors.json vectors.json.backup
```

