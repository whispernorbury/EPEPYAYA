# EPEPServer - Semantic Autocomplete Server

의미 기반 자동완성 서버입니다. 문장을 입력받아 OpenAI 임베딩을 통해 벡터로 변환하고, 미리 계산된 문장 벡터들과 코사인 유사도를 비교하여 가장 유사한 상위 5개의 문장을 추천해주는 WebSocket 기반 백엔드 서버입니다.

## 주요 기능

- 🔍 의미 기반 벡터 검색
- 💾 Redis 캐싱을 통한 빠른 응답
- 🤖 OpenAI 임베딩 및 LLM 폴백
- 🔌 WebSocket 실시간 통신
- 🐳 Docker 지원

## 빠른 시작

### 개발 환경

```bash
# 의존성 설치
npm install

# 벡터 파일 생성 (모의 데이터)
node prepare_embedding.js --mock

# 서버 실행
npm start
```

### Docker로 실행

```bash
# 기본 실행 (개발 모드)
docker-compose up -d

# 프로덕션 모드 (환경 변수 설정 필요)
# .env 파일에서 MOCK_MODE=false, REDIS_PASSWORD 설정 후
docker-compose up -d
```

## 문서

- **[배포체크리스트.md](./배포체크리스트.md)** - 프로덕션 배포 가이드
- **[배포간단가이드.md](./배포간단가이드.md)** - 빠른 배포 참조
- **[배포요약.md](./배포요약.md)** - 핵심 체크리스트

## 환경 변수

`.env` 파일을 생성하고 `env.production.example`을 참고하여 설정하세요.

필수 변수:
- `OPENAI_API_KEY` - OpenAI API 키
- `MOCK_MODE` - 모의 모드 (개발: true, 프로덕션: false)
- `REDIS_URL` - Redis 연결 URL

## 라이선스

MIT

