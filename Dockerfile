FROM node:18-slim

WORKDIR /app

# Node.js 의존성 설치 (캐시 최적화)
COPY package.json package-lock.json* ./
RUN npm ci --only=production && npm cache clean --force

# 애플리케이션 파일 복사
COPY server.js ./

# Production env
ENV NODE_ENV=production
ENV PYTHON_CMD=python3
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

CMD ["node", "server.js"]
