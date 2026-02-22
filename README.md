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
pip install -r requirements.txt

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
pip install -r requirements.txt

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

### Run REST API
```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
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
# Access demo player and API schemas at: http://127.0.0.1:8000
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
curl -X POST "http://localhost:8000/api/prepare" \
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
curl "http://localhost:8000/api/status/<stream_id>"
```

### Example: Stream Audio

```bash
curl -L "http://localhost:8000/api/stream/<stream_id>" --output out.wav
```

### Example: Cancel

```bash
curl -X POST "http://localhost:8000/api/cancel/<stream_id>"
```

### Voice Clone (Base) note

`ref_text` is optional. If omitted, server tries auto-transcription using `faster-whisper`.

---

## 7. Text Normalization

TTS architectures parse phonemes sequentially. Raw digits (e.g., "1945", "$5.00") or complex abbreviations may result in mispronunciations or silence.

**Best Practice:** Always preprocess input text using a normalizer (such as the included `num2words` Python library) to explicitly transliterate numbers into words (e.g., "one thousand nine hundred forty five") before sending the string to the API.
