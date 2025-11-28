FROM node:18-slim

WORKDIR /app

# wget 설치 (healthcheck용)
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

# Node.js 의존성 설치 (캐시 최적화)
COPY package.json package-lock.json* ./
RUN npm ci --only=production && npm cache clean --force

# 애플리케이션 파일 복사
COPY server.js ./

# Production env
ENV NODE_ENV=production

EXPOSE 3000

CMD ["node", "server.js"]
