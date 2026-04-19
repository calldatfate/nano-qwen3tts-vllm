# nano-qwen3tts-vllm

Qwen3-TTS runtime для Paidviewer.

## Кому нужен этот репозиторий

Этот репозиторий нужен тому, кто поднимает Qwen runtime для облачного или self-host контура Paidviewer.

Если ты обычный пользователь Paidviewer и не обслуживаешь Qwen runtime, этот репозиторий тебе обычно не нужен.

## Роль в системе

Этот runtime используется в двух сценариях:

- `cloud` mode за `tts-gateway`
- `self_host` mode за `tts_worker_agent`

## Ограничения платформы

Flash-Attention и Triton требуют Linux GPU runtime.

Для Windows штатный путь — Docker Desktop с Linux containers и рабочей GPU-поддержкой через WSL2.
Считать этот runtime нужно Docker-first и Linux-container-first.

## Рекомендуемая политика запуска

Лучше держать single-model-first deployment, если нет явной необходимости в большем каталоге моделей.

Важные env:

- `QWEN3_TTS_MODEL_PATH`
- `QWEN_ALLOWED_MODELS`
- `QWEN_TTS_ALLOWED_MODELS`
- `QWEN_VOICE_STORAGE_DIR`

`QWEN_VOICE_STORAGE_DIR` должен жить на постоянном volume.

## Health

- `GET /health/live`
- `GET /health/ready`

## Быстрый запуск

Базовый runtime: Python `3.12`.

### Официальный путь: Docker

```bash
cp .env.example .env
# edit API_KEY before exposing the service
docker compose up --build api
```

По умолчанию:

- host: `http://localhost:8012`
- container: `http://api:8000`
- health: `GET /health/live`, `GET /health/ready`
- protected API: `API_KEY` via `Authorization: Bearer <key>` or `X-API-Key: <key>`
- API port is bound to `127.0.0.1:8012` by default; expose it through Paidviewer/gateway infrastructure, not directly to the Internet
- persistent volumes:
  - Hugging Face cache
  - Qwen voice storage

Если нужен и веб-интерфейс:

```bash
docker compose --profile debug-ui up --build api web
```

Веб-интерфейс предназначен для ручной отладки и тоже привязан к `127.0.0.1`.

### Локально на Linux или WSL

Этот путь оставляем только для осознанной отладки runtime вне контейнера.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python api_server.py
```

## Важные замечания

- не используй `latest` в активных deploy-примерах
- этот репозиторий — runtime, а не product control plane
- для Paidviewer основной рабочий контракт — запуск в Docker и подключение через `bot_service` / `tts-gateway`
- health endpoints оставляй открытыми для orchestrator, а рабочие `/api/*` маршруты защищай `API_KEY`
- `bot_service` остаётся источником истины для пользовательских настроек и routing policy
