import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch._dynamo
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from nano_qwen3tts_vllm.auth import reject_unauthorized_request
from nano_qwen3tts_vllm.model_runtime import ModelRuntime
from nano_qwen3tts_vllm.runtime_policy import (
    resolve_runtime_model_or_409,
    runtime_allowed_models_raw_from_env,
    runtime_allowed_models_source_from_env,
    runtime_configured_model_from_env,
    runtime_model_catalog_from_env,
)
from nano_qwen3tts_vllm.stream_routes import StreamService
from nano_qwen3tts_vllm.stream_scheduler import FairStreamScheduler
from nano_qwen3tts_vllm.voice_routes import VoiceApiService
from nano_qwen3tts_vllm.voice_store import FileVoiceStore

torch._dynamo.config.cache_size_limit = 64

ENFORCE_EAGER = False
logger = logging.getLogger(__name__)
runtime = ModelRuntime(enforce_eager=ENFORCE_EAGER, logger=logger)
model_switch_lock = asyncio.Lock()

MAX_QUEUE_PER_TENANT = int(os.environ.get("MAX_QUEUE_PER_TENANT", "20"))
MAX_TOTAL_QUEUED = int(os.environ.get("MAX_TOTAL_QUEUED", "200"))
STREAM_WAIT_TIMEOUT_SEC = float(os.environ.get("STREAM_WAIT_TIMEOUT_SEC", "0"))
SERVER_HOST = os.environ.get("QWEN_TTS_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("QWEN_TTS_PORT", os.environ.get("PORT", "8012")))
VOICE_STORAGE_DIR = Path(os.environ.get("QWEN_VOICE_STORAGE_DIR", "./runtime/qwen_voices")).resolve()
VOICE_FILES_DIR = VOICE_STORAGE_DIR / "files"
VOICE_PREVIEW_DIR = VOICE_STORAGE_DIR / "previews"
VOICE_STATE_PATH = VOICE_STORAGE_DIR / "state.json"

scheduler = FairStreamScheduler(
    max_queue_per_tenant=MAX_QUEUE_PER_TENANT,
    max_total_queued=MAX_TOTAL_QUEUED,
    stream_wait_timeout_sec=STREAM_WAIT_TIMEOUT_SEC,
)


async def switch_model_if_needed(model_name: str):
    await runtime.switch_model_if_needed(model_name)


def _decode_batch(codes: list):
    return runtime.decode_batch(codes)


stream_service = StreamService(
    runtime=runtime,
    scheduler=scheduler,
    model_switch_lock=model_switch_lock,
    switch_model_if_needed=switch_model_if_needed,
    resolve_runtime_model=resolve_runtime_model_or_409,
    decode_batch=_decode_batch,
    logger=logger,
)
voice_api_service = VoiceApiService(
    runtime=runtime,
    logger=logger,
    model_switch_lock=model_switch_lock,
    switch_model_if_needed=switch_model_if_needed,
    resolve_runtime_model=resolve_runtime_model_or_409,
    runtime_model_catalog=runtime_model_catalog_from_env,
    runtime_configured_model=runtime_configured_model_from_env,
    runtime_allowed_models_raw=runtime_allowed_models_raw_from_env,
    runtime_allowed_models_source=runtime_allowed_models_source_from_env,
    audio_stream_generator=stream_service.audio_stream_generator_async,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    VOICE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    app.state.voice_store = FileVoiceStore(VOICE_STATE_PATH, VOICE_FILES_DIR)
    await app.state.voice_store.startup()
    app.state.voice_files_dir = VOICE_FILES_DIR
    app.state.voice_preview_dir = VOICE_PREVIEW_DIR
    yield
    await app.state.voice_store.close()
    await runtime.shutdown()


app = FastAPI(title="Qwen3-TTS Streaming API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(voice_api_service.build_router())
app.include_router(stream_service.build_router())


@app.middleware("http")
async def enforce_api_key(request: Request, call_next):
    rejection = reject_unauthorized_request(request)
    if rejection is not None:
        return rejection
    return await call_next(request)

HTML_UI = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Qwen3-TTS Streaming API Server ⚡</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f9fafb; color: #111827; }
        .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h1 { margin-top: 0; color: #2563eb; }
        label { display: block; font-weight: 600; margin-top: 16px; margin-bottom: 8px; }
        select, input, textarea { width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 16px; box-sizing: border-box; }
        .btn-group { display: flex; gap: 10px; margin-top: 24px; }
        button { background: #2563eb; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; flex: 1; transition: background 0.2s; }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        .btn-stop { background: #dc2626; }
        .btn-stop:hover { background: #b91c1c; }
        audio { width: 100%; margin-top: 24px; display: none; }
        .status { margin-top: 20px; padding: 15px; border-radius: 4px; font-size: 14px; font-weight: bold; display: none; }
        .status.loading { background: #eff6ff; border-left: 4px solid #3b82f6; color: #3b82f6; }
        .status.error { background: #fee2e2; border-left: 4px solid #ef4444; color: #ef4444; }
        .status.success { background: #dcfce7; border-left: 4px solid #22c55e; color: #22c55e; }
        .info { padding: 12px; background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 4px; margin-bottom: 20px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎙️ Настоящий Streaming API</h1>
        <div class="info">
            Звук начинает воспроизводиться прямо в браузере сразу после получения ПЕРВОГО миллисекундного чанка (TTFT), не дожидаясь генерации всего текста! <br><br>
            <b>Для программистов:</b> Это работает через обычный REST API: сначала POST-запрос на <code>/api/prepare</code>, затем GET-стриминг <code>/api/stream/{id}</code>.
        </div>
        <label>API Key</label>
        <input type="password" id="apiKeyInput" placeholder="Required when API_KEY is configured">
        <form id="tts-form">
            <label>Выбор модели</label>
            <select id="modelSelect" name="model" onchange="updateUI()">
                <option value="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign">⭐️ Qwen3-TTS-12Hz-1.7B-VoiceDesign (Дизайн по тексту)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice">⚡ Qwen3-TTS-12Hz-0.6B-CustomVoice (Готовые дикторы, Быстро)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice">🎙️ Qwen3-TTS-12Hz-1.7B-CustomVoice (Готовые дикторы, Качественно)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-0.6B-Base">⚡ Qwen3-TTS-12Hz-0.6B-Base (Быстрое Клонирование)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-1.7B-Base">🎭 Qwen3-TTS-12Hz-1.7B-Base (Качественное Клонирование)</option>
            </select>
            <label>Язык текста</label>
            <select id="languageSelect" name="language">
                <option value="Russian">Russian</option>
                <option value="English">English</option>
            </select>
            <label>Temperature</label>
            <input id="temperature" name="temperature" type="number" min="0.1" max="2.0" step="0.05" value="0.9">
            <label>Текст для озвучки</label>
            <textarea id="textInput" name="text" rows="4">Привет! Это потоковое воспроизведение через API. Вы начнете слышать звук еще до того, как весь этот длинный текст будет полностью сгенерирован нашей нейросетью. Это работает очень быстро и круто!</textarea>
            <div id="field-instruction" class="dynamic-field">
                <label>Описание голоса (Instruction)</label>
                <input type="text" id="instruction" name="instruction" value="Уверенный мужской голос, профессиональный диктор, очень радостный">
            </div>
            <div id="field-speaker" class="dynamic-field">
                <label>Выбор встроенного диктора (CustomVoice)</label>
                <select id="speaker" name="speaker">
                    <option value="serena">serena</option>
                    <option value="vivian">vivian</option>
                    <option value="uncle_fu">uncle_fu</option>
                    <option value="ryan">ryan</option>
                    <option value="aiden">aiden</option>
                    <option value="ono_anna">ono_anna</option>
                    <option value="sohee">sohee</option>
                    <option value="eric">eric</option>
                    <option value="dylan">dylan</option>
                </select>
            </div>
            <div id="field-clone" class="dynamic-field">
                <label>Загрузите аудио-оригинал (.wav, .mp3)</label>
                <input type="file" id="ref_audio" name="ref_audio" accept="audio/*">
                <label>Оригинальный текст с аудио (буква в букву)</label>
                <textarea id="ref_text" name="ref_text" rows="2" placeholder="Оставьте пустым для АВТОМАТИЧЕСКОГО РАСПОЗНАВАНИЯ (Whisper)"></textarea>
            </div>
            <div class="btn-group">
                <button id="generateBtn" type="button" onclick="generateTTS()">Генерировать поток 🔊</button>
                <button id="stopBtn" type="button" onclick="stopTTS()" class="btn-stop" disabled>Остановить</button>
            </div>
        </form>
        <div id="status" class="status"></div>
        <audio id="audioPlayer" controls autoplay></audio>
    </div>
    <script>
        function updateUI() {
            const model = document.getElementById('modelSelect').value;
            document.getElementById('field-instruction').style.display = model.includes('VoiceDesign') ? 'block' : 'none';
            document.getElementById('field-speaker').style.display = model.includes('CustomVoice') ? 'block' : 'none';
            document.getElementById('field-clone').style.display = model.includes('Base') ? 'block' : 'none';
        }
        updateUI();
        let currentStreamId = null;
        async function stopTTS() {
            if (currentStreamId) {
                try {
                    await fetch(`/api/cancel/${currentStreamId}`, { method: 'POST', headers: buildApiHeaders() });
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = "Воспроизведение остановлено вручную.";
                    statusDiv.className = "status error";
                    statusDiv.style.display = 'block';
                    const player = document.getElementById('audioPlayer');
                    player.pause();
                    player.currentTime = 0;
                    player.src = "";
                } catch (e) {
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = "Ошибка при отмене: " + e.message;
                    statusDiv.className = "status error";
                    statusDiv.style.display = 'block';
                }
            }
            resetButtons();
        }
        function resetButtons() {
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            currentStreamId = null;
        }
        function buildApiHeaders() {
            const apiKey = (document.getElementById('apiKeyInput')?.value || '').trim();
            if (!apiKey) {
                return {};
            }
            return {
                'Authorization': `Bearer ${apiKey}`,
                'X-API-Key': apiKey,
            };
        }
        async function generateTTS() {
            const generateBtn = document.getElementById('generateBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusDiv = document.getElementById('status');
            const player = document.getElementById('audioPlayer');
            generateBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.style.display = 'block';
            statusDiv.className = 'status loading';
            statusDiv.textContent = 'Загрузка модели и подготовка потока...';
            player.src = '';
            player.style.display = 'none';
            try {
                const formData = new FormData();
                formData.append('model', document.getElementById('modelSelect').value);
                formData.append('text', document.getElementById('textInput').value);
                formData.append('language', document.getElementById('languageSelect').value);
                formData.append('temperature', document.getElementById('temperature').value);
                formData.append('instruction', document.getElementById('instruction').value);
                formData.append('speaker', document.getElementById('speaker').value);
                const refAudioInput = document.getElementById('ref_audio');
                if (refAudioInput && refAudioInput.files.length > 0) {
                    formData.append('ref_audio', refAudioInput.files[0]);
                    formData.append('ref_text', document.getElementById('ref_text').value);
                }
                const t0 = performance.now();
                const response = await fetch('/api/prepare', { method: 'POST', body: formData, headers: buildApiHeaders() });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Неизвестная ошибка API");
                }
                const t1 = performance.now();
                currentStreamId = data.stream_id;
                statusDiv.textContent = `Подготовка заняла ${Math.round(t1 - t0)}мс. Подключаемся к потоку звука... Слушайте!`;
                statusDiv.className = 'status success';
                const apiKey = (document.getElementById('apiKeyInput')?.value || '').trim();
                player.src = apiKey
                    ? `/api/stream/${currentStreamId}?api_key=${encodeURIComponent(apiKey)}`
                    : `/api/stream/${currentStreamId}`;
                player.style.display = 'block';
                player.play().catch(() => {});
                player.onended = () => {
                    statusDiv.textContent = "Воспроизведение завершено.";
                    statusDiv.className = "status success";
                    resetButtons();
                };
                player.onerror = () => {
                    statusDiv.textContent = "Ошибка воспроизведения аудио.";
                    statusDiv.className = "status error";
                    resetButtons();
                };
            } catch (err) {
                statusDiv.textContent = "Ошибка: " + err.message;
                statusDiv.className = 'status error';
                resetButtons();
            }
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def serve_ui():
    return HTMLResponse(content=HTML_UI, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS streaming API server")
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Qwen3-TTS Streaming API Server")
    logger.info("Go to: http://127.0.0.1:%s", args.port)
    uvicorn.run(app, host=args.host, port=args.port)
