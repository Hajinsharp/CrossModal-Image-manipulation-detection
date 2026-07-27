FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, pinned to the exact version in requirements.txt so the
# `pip install -r requirements.txt` step below sees it as already satisfied
# and skips re-resolving it from the default (CUDA-bundled) PyPI wheel.
RUN pip install --no-cache-dir \
    torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY assets/ ./assets/

ENV HF_HOME=/app/.cache
EXPOSE 8000

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
