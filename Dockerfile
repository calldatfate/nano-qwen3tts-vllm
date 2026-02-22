FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md /app/
COPY api_server.py web_ui.py /app/
COPY nano-qwen3tts-vllm /app/nano-qwen3tts-vllm
COPY examples /app/examples

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install -e .

EXPOSE 8000 7860

ENV USE_ZMQ=1

CMD ["python3", "api_server.py"]
