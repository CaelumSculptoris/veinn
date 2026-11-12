FROM node:22-alpine AS frontend

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY vite.config.js ./
COPY web ./web
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VEINN_HOST=0.0.0.0 \
    PORT=8765 \
    VEINN_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web.py ./
COPY --from=frontend /build/web/dist ./web/dist

RUN mkdir -p /data \
    && useradd --system --uid 10001 --create-home veinn \
    && chown veinn:veinn /data

USER veinn

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8765/', timeout=3)"

CMD ["python", "web.py"]
