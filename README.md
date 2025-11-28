# EPEPServer - Semantic Autocomplete Server

의미 기반 자동완성 서버입니다. 문장을 입력받아 Python 기반 한국어 임베딩 모델을 통해 벡터로 변환하고, 미리 계산된 문장 벡터들과 코사인 유사도를 비교하여 가장 유사한 상위 5개의 문장을 추천해주는 WebSocket 기반 백엔드 서버입니다.

## 주요 기능

- 🔍 의미 기반 벡터 검색
- 💾 Redis 캐싱을 통한 빠른 응답
- 🇰🇷 Python 기반 한국어 임베딩 모델 (로컬 실행, API 키 불필요)
- 🤖 OpenAI LLM 폴백 (선택사항)
- 🔌 WebSocket 실시간 통신
- 🐳 Docker 지원

## 빠른 시작

```bash
# 1. 환경 변수 설정
cp env.production.example .env

# 2. 벡터 파일 생성 (로컬)
npm install && pip3 install -r requirements.txt
python3 prepare_embedding.py

# 3. Docker Compose로 실행
docker-compose up -d
```

자세한 배포 가이드와 환경 변수 설명은 **[DEPLOY.md](./DEPLOY.md)**를 참고하세요.

## 라이선스

MIT

