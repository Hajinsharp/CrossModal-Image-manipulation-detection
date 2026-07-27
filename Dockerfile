FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# NOTE: torch==2.13.0 (pinned in requirements.txt) has no build on the
# CPU-only index (download.pytorch.org/whl/cpu tops out around 2.9.x for
# any Python version as of this writing) — the CPU-only build channel
# lags several minor versions behind the main PyPI releases. Installing
# from the default index instead, which does have this exact version,
# at the cost of a larger image (CUDA runtime libs bundled in the wheel
# even though they're unused on CPU). Revisit if/when the CPU index
# catches up to this torch version.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY assets/ ./assets/

ENV HF_HOME=/app/.cache
EXPOSE 8000

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
