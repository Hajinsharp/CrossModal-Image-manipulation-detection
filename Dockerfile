FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
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

# Bake the model weights into the image at build time (one-time download
# from Cloud Build's infrastructure against a public repo, no auth needed)
# so the app never has to fetch ~520 MB from HF Hub on a cold start --
# eliminates both the startup latency and the rate-limit exposure that
# comes from every scale-from-zero event hitting HF Hub from Cloud Run's
# heavily-shared serving IP range. See src/predict.py's WEIGHTS_LOCAL_PATH.
RUN python -c "\
import shutil; \
from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='Roy407/crossmodalfusionnet', filename='weights.pth'); \
shutil.copy(p, '/app/src/weights.pth')"

ENV HF_HOME=/app/.cache
EXPOSE 8000

# Shell form (not exec-form JSON array) so $PORT actually expands. Cloud
# Run injects PORT at runtime (defaults to 8080) and requires the
# container to listen on it; falls back to 8000 for local `docker run`,
# matching the port this Dockerfile has always exposed.
CMD uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}
