# Qwen3-TTS Streaming API (nano-vLLM)

An optimized backend for Qwen3-TTS providing high-throughput, real-time audio streaming. This repository utilizes the `nano-vLLM` framework with an O(1) Sliding-Window algorithm to eliminate streaming latency (TTFT) and efficiently manage memory for large TTS models.

Flash-Attention and Triton require a Linux environment. On Windows, this is achieved natively via WSL2.

---

## 1. Installation

### Native Linux (Ubuntu/Debian)
```bash
# 1. Install prerequisites
# SoX is used for audio processing; FFmpeg is required for robust audio decode/transcode
sudo apt update && sudo apt install -y sox libsox-dev ffmpeg

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel packaging ninja
pip install -r requirements.txt
pip install --no-build-isolation flash-attn

# 4. Install this repo as a local package
pip install -e .
```

### Windows (via WSL2)
```bash
# 1. Open your Windows terminal and enter the WSL subsystem
wsl

# 2. Install prerequisites inside the Linux subsystem
sudo apt update && sudo apt install -y sox libsox-dev ffmpeg

# 3. Create and activate a virtual environment
python3 -m venv venv_wsl
source venv_wsl/bin/activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel packaging ninja
pip install -r requirements.txt
pip install --no-build-isolation flash-attn

# 5. Install this repo as a local package
pip install -e .
```

### Verify FFmpeg
```bash
ffmpeg -version
```
If this command is missing, auto-transcription / audio decoding (for example with `faster-whisper`) may fail for some input formats.

---

## 2. Docker Run

### Requirements
- Docker Engine or Docker Desktop
- NVIDIA GPU + NVIDIA Container Toolkit (for `--gpus all`)

### Build image
```bash
docker build -t nano-qwen3tts-vllm .
```

The Docker image keeps the CUDA 12.4.1 Ubuntu 22.04 base, but installs a managed
`Python 3.12` runtime with `uv` on top so the Docker pipeline matches modern
project environments without depending on the distro's system Python.

The Docker image compiles `flash-attn` during build. The first build is therefore
noticeably slower than a plain Python image build and requires the CUDA `devel`
toolchain inside the container.

To make the build more stable on Docker Desktop / constrained RAM setups, the
Dockerfile limits `flash-attn` build parallelism by default (`MAX_JOBS=1`,
`NVCC_THREADS=1`).

During Docker build, the image first resolves the exact prebuilt `flash-attn`
wheel URL from the current Python / Torch ABI. If that wheel is unavailable,
the build fails fast by default instead of silently spending a long time on a
source build. Set `FLASH_ATTN_ALLOW_SOURCE_BUILD=1` only if you explicitly want
to allow the slow source-build fallback.

Docker also applies a dedicated constraints file so `qwen-tts` does not
silently pull a newer `torch` version for which no compatible prebuilt
`flash-attn` wheel exists.

By default, the Dockerfile targets Ampere (`TORCH_CUDA_ARCH_LIST=8.6`), which
matches RTX 3080-class GPUs. Override it for other GPUs if needed:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  -t nano-qwen3tts-vllm .
```

You can also override `flash-attn` build parallelism if your machine has more RAM:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.6 \
  --build-arg FLASH_ATTN_MAX_JOBS=2 \
  --build-arg FLASH_ATTN_NVCC_THREADS=2 \
  -t nano-qwen3tts-vllm .
```

If you intentionally want to allow source compilation when no prebuilt wheel is
available:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg FLASH_ATTN_ALLOW_SOURCE_BUILD=1 \
  -t nano-qwen3tts-vllm .
```

### Run REST API
```bash
docker run --rm -it \
  --gpus all \
  -p 8012:8012 \
  -e QWEN_TTS_PORT=8012 \
  -e USE_ZMQ=1 \
  -v qwen3tts_hf_cache:/root/.cache/huggingface \
  nano-qwen3tts-vllm
```

### Run Web UI
```bash
docker run --rm -it \
  --gpus all \
  -p 7860:7860 \
  -e USE_ZMQ=0 \
  -v qwen3tts_hf_cache:/root/.cache/huggingface \
  nano-qwen3tts-vllm python3 web_ui.py
```

### Run with Docker Compose
```bash
docker compose up api
# or
docker compose up web
```

For non-Ampere GPUs, export a matching `TORCH_CUDA_ARCH_LIST` before compose build/run.
You may also raise `FLASH_ATTN_MAX_JOBS` / `FLASH_ATTN_NVCC_THREADS` if Docker has enough RAM,
or set `FLASH_ATTN_ALLOW_SOURCE_BUILD=1` if you explicitly want the slow source-build fallback.

---

## 3. Execution (without Docker)

Once the environment is active, choose your preferred interface:

### Option A: Web UI (Gradio)
Interactive interface for testing models and generating local audio files.
```bash
python web_ui.py
# Access at: http://127.0.0.1:7860
```

### Option B: REST API (FastAPI)
Production-ready backend for asynchronous audio streaming via ZeroMQ with fair tenant queue scheduling.
```bash
python api_server.py
# Access demo player and API schemas at: http://127.0.0.1:8012
```

The API also exposes lightweight health endpoints:

```bash
curl http://127.0.0.1:8012/health/live
curl http://127.0.0.1:8012/health/ready
```

---

## 4. Base Voice Cloning Reference Audio Formats

For `Base` models, `ref_audio` is decoded with `soundfile` in the API path. In this repository's tested WSL runtime (`soundfile 0.13.1`), the following container formats are available:

- `AIFF`
- `AU`
- `AVR`
- `CAF`
- `FLAC`
- `HTK`
- `IRCAM`
- `MAT4`
- `MAT5`
- `MP3`
- `MPC2K`
- `NIST`
- `OGG`
- `PAF`
- `PVF`
- `RAW`
- `RF64`
- `SD2`
- `SDS`
- `SVX`
- `VOC`
- `W64`
- `WAV`
- `WAVEX`
- `WVE`
- `XI`

In practice, `WAV` is the most reliable choice for voice cloning samples. `MP3`, `FLAC`, and `OGG` are also supported in the tested setup.

To check formats in your own environment:
```bash
python -c "import soundfile as sf; print(sf.__version__); print(sorted(sf.available_formats().keys()))"
```

---

## 5. Supported Models

The API supports 5 model variants, accessible via the `model` parameter:
- **Voice Design models** (`1.7B`): Generates voice characteristics strictly from textual descriptions.
- **Custom Voice models** (`0.6B`, `1.7B`): Utilizes pre-trained, high-quality speaker embeddings.
- **Base models** (`0.6B`, `1.7B`): Full zero-shot Voice Cloning from a provided reference audio snippet.

### Runtime model policy

By default the worker can expose the full catalog above. For production deployments it is usually better to expose only the model you actually run:

- `QWEN3_TTS_MODEL_PATH=<exact model id or local path>`
  - exposes a single runtime model via `/api/models`
- `QWEN_TTS_ALLOWED_MODELS=base`
  - exposes only the requested family
- `QWEN_TTS_ALLOWED_MODELS=Qwen/Qwen3-TTS-12Hz-0.6B-Base,Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
  - exposes only the listed exact models

Keep `QWEN_VOICE_STORAGE_DIR` on a persistent shared volume. Stored voice samples survive runtime restarts and become usable again when a `Base` runtime is brought back online.

---

## 6. Streaming API Reference (cURL)

The API is now optimized for multi-tenant chat pipelines with a single generation engine and a fair queue.
Scheduling is round-robin by tenant, not plain FIFO by request.

Workflow:
1. `POST /api/prepare` to enqueue a request.
2. (Optional) `GET /api/status/{stream_id}` to poll queue/execution state.
3. `GET /api/stream/{stream_id}` to wait for turn and receive streamed WAV bytes.
4. `POST /api/cancel/{stream_id}` to cancel queued/running requests.

### Tenant Identity (important)

Use `tenant_id` as the fairness key (for example `twitch:<broadcaster_id>`).
If `tenant_id` is not provided, server falls back to `channel_name` (`channel:<name>`), then `default`.

### Request Fields

Required:
- `model`
- `text`
- `language`

Optional:
- `temperature` (float, default `0.9`, must be `> 0`)
- `instruction` (VoiceDesign)
- `speaker` (CustomVoice)
- `ref_audio` + optional `ref_text` (Base)
- `tenant_id`
- `channel_name`
- `author`
- `user_id`

### Queue Limits (env vars)

- `MAX_QUEUE_PER_TENANT` (default `20`)
- `MAX_TOTAL_QUEUED` (default `200`)
- `STREAM_WAIT_TIMEOUT_SEC` (default `0`, disabled)

### Example: Enqueue (CustomVoice)

```bash
curl -X POST "http://localhost:8012/api/prepare" \
  -F "model=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
  -F "text=Replace this with your desired output text." \
  -F "language=Russian" \
  -F "temperature=0.9" \
  -F "speaker=serena" \
  -F "tenant_id=twitch:12345678" \
  -F "channel_name=my_streamer_channel" \
  -F "author=chat_user" \
  -F "user_id=987654321"
```

Response (example):
```json
{
  "stream_id": "7c4e1c2b-...-a6d5",
  "tenant_id": "twitch:12345678",
  "state": "queued",
  "tenant_queue_depth": 2,
  "global_queue_depth": 15
}
```

### Example: Poll Status

```bash
curl "http://localhost:8012/api/status/<stream_id>"
```

### Example: Stream Audio

```bash
curl -L "http://localhost:8012/api/stream/<stream_id>" --output out.wav
```

### Example: Cancel

```bash
curl -X POST "http://localhost:8012/api/cancel/<stream_id>"
```

### Voice Clone (Base) note

`ref_text` is optional. If omitted, server tries auto-transcription using `faster-whisper`.

---

## 7. Text Normalization

TTS architectures parse phonemes sequentially. Raw digits (e.g., "1945", "$5.00") or complex abbreviations may result in mispronunciations or silence.

**Best Practice:** Always preprocess input text using a normalizer (such as the included `num2words` Python library) to explicitly transliterate numbers into words (e.g., "one thousand nine hundred forty five") before sending the string to the API.
