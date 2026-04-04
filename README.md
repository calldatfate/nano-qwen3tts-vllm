# nano-qwen3tts-vllm

Paidviewer-oriented Qwen3-TTS streaming runtime.

## Production role

This repository provides the Qwen runtime used by Paidviewer in two contexts:

- cloud mode behind `tts-gateway`
- self-host mode behind `tts_worker_agent`

## Platform requirement

Flash-Attention and Triton require Linux.

For Windows operators, the supported path is WSL2. Treat this runtime as Linux/WSL-first.

## Recommended runtime policy

Use a single-model-first deployment unless there is a deliberate need for a wider catalog.

Important env:

- `QWEN3_TTS_MODEL_PATH`
- `QWEN_ALLOWED_MODELS`
- `QWEN_TTS_ALLOWED_MODELS`
- `QWEN_VOICE_STORAGE_DIR`

`QWEN_VOICE_STORAGE_DIR` must live on a persistent volume.

## Health

- `GET /health/live`
- `GET /health/ready`

## Run

### Local Linux or WSL

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python api_server.py
```

### Docker

```bash
docker build -t nano-qwen3tts-vllm:local .
docker compose up api
```

## Notes

- avoid `latest` in active deploy examples
- this repo is the runtime, not the product control plane
- `bot_service` remains the source of truth for user settings and routing decisions
