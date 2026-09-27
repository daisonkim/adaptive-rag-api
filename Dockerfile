FROM python:3.12-slim-bookworm

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 작업 디렉토리 설정
WORKDIR /app

# 의존성만 먼저 설치 (레이어 캐시)
COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-dev --no-install-project

# 애플리케이션 코드 복사 후 프로젝트 설치
COPY . /app/
RUN uv sync --frozen --no-dev

# 포트 노출
EXPOSE 8000

# slim 이미지에 curl이 없어 python으로 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# FastAPI 애플리케이션 실행
CMD ["uv", "run", "--no-sync", "python", "-m", "adaptive_rag_api.main"]
