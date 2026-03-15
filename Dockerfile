FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_ARCH_LIST=8.6
ARG FLASH_ATTN_MAX_JOBS=1
ARG FLASH_ATTN_NVCC_THREADS=1
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_ALLOW_SOURCE_BUILD=0
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=manual \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_PREFERENCE=managed \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${FLASH_ATTN_MAX_JOBS} \
    NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${FLASH_ATTN_MAX_JOBS}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints-docker.txt pyproject.toml README.md /app/
COPY api_server.py web_ui.py voice_store.py voice_uploads.py runtime_models.py /app/
COPY nano-qwen3tts-vllm /app/nano-qwen3tts-vllm
COPY examples /app/examples

RUN uv python install ${PYTHON_VERSION} && \
    uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION} --seed && \
    python --version && \
    python -m pip install --upgrade pip setuptools wheel packaging ninja && \
    python -m pip install -c constraints-docker.txt -r requirements.txt && \
    FLASH_ATTN_URL="$(python -c "import platform, sys, torch; version='${FLASH_ATTN_VERSION}'; py=f'cp{sys.version_info.major}{sys.version_info.minor}'; torch_version='.'.join(torch.__version__.split('.')[:2]); cxx11='TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'; plat=f'linux_{platform.uname().machine}'; print(f'https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/flash_attn-{version}+cu12torch{torch_version}cxx11abi{cxx11}-{py}-{py}-{plat}.whl')")" && \
    echo "Resolved flash-attn wheel URL: ${FLASH_ATTN_URL}" && \
    (python -m pip install "${FLASH_ATTN_URL}" || \
        ([ "${FLASH_ATTN_ALLOW_SOURCE_BUILD}" = "1" ] && python -m pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}")) && \
    python -m pip install --no-deps -e .

EXPOSE 8012 7860

ENV USE_ZMQ=1

CMD ["python", "api_server.py"]
