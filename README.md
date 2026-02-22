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
Production-ready backend for synchronous and asynchronous audio streaming via ZeroMQ.
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

The streaming endpoint operates via a Producer-Consumer pattern.
Endpoint workflow: Send `POST /api/prepare` -> Receive `stream_id` -> Connect to `GET /api/stream/{stream_id}` to receive `audio/L16` PCM bytes iteratively.

Common parameter:
- `temperature` (float, default `0.9`, must be `> 0`) controls sampling randomness.

### Voice Clone (Zero-Shot)
> Note: The `ref_text` parameter is optional. If omitted, the server automatically transcribes the reference audio on the GPU using `faster-whisper`.

```bash
curl -X POST "http://localhost:8000/api/prepare" \
  -F "model=Qwen/Qwen3-TTS-12Hz-0.6B-Base" \
  -F "text=Replace this with your desired output text." \
  -F "language=Russian" \
  -F "temperature=0.9" \
  -F "ref_audio=@/path/to/your/reference.wav"
```

### Voice Design (Natural Language Instructions)
```bash
curl -X POST "http://localhost:8000/api/prepare" \
  -F "model=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  -F "text=Replace this with your desired output text." \
  -F "language=Russian" \
  -F "temperature=0.9" \
  -F "instruction=Female voice, professional and calm tone"
```

### Custom Voice (Pre-trained Speakers)
```bash
curl -X POST "http://localhost:8000/api/prepare" \
  -F "model=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
  -F "text=Replace this with your desired output text." \
  -F "language=Russian" \
  -F "temperature=0.9" \
  -F "speaker=serena"
```

---

## 7. Text Normalization

TTS architectures parse phonemes sequentially. Raw digits (e.g., "1945", "$5.00") or complex abbreviations may result in mispronunciations or silence.

**Best Practice:** Always preprocess input text using a normalizer (such as the included `num2words` Python library) to explicitly transliterate numbers into words (e.g., "one thousand nine hundred forty five") before sending the string to the API.
