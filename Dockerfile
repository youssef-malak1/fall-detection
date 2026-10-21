# ElderLife fall detection — inference service
#
#   docker build -t elderlife-fall .
#   docker run --rm -p 8100:8100 elderlife-fall
#
# The image is large (~1.2 GB) because it carries PyTorch and the MediaPipe pose
# model. That is the price of "analysing video, not tagging text"; if size matters,
# serve the ONNX export instead (see docs/model-card.md).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# OpenCV and MediaPipe need these shared libraries even in headless containers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what serving needs: no data/, no notebooks, no tests.
COPY src ./src
COPY models ./models
COPY configs ./configs

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health')"

CMD ["uvicorn", "src.inference.service:app", "--host", "0.0.0.0", "--port", "8100"]
