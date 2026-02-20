import io
import struct
import numpy as np
import gc
import uuid
import torch
import soundfile as sf
import traceback
import os
import asyncio
import time
import threading
import sys
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph
from nano_qwen3tts_vllm.utils.prompt import prepare_custom_voice_prompt, _tokenize_texts
from nano_qwen3tts_vllm.utils.generation import prepare_inputs, generate_speaker_prompt, generate_icl_prompt

# Global state
current_model_name = None
interface = None
_zmq_bridge = None # Renamed from zmq_bridge
_tokenizer = None # New global for tokenizer
USE_ZMQ = False # New global for ZMQ status
ENFORCE_EAGER = False

# Global lock for safe decoding across async requests
decode_lock = threading.Lock()

# Store active generators and cancellation events for streaming
active_streams = {}

def load_model(model_name):
    global interface, _tokenizer, USE_ZMQ, _zmq_bridge, current_model_name
    
    USE_ZMQ = os.environ.get("USE_ZMQ", "1") == "1"
    
    if USE_ZMQ:
        _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
        interface = Qwen3TTSInterface.from_pretrained(
            model_name,
            zmq_bridge=_zmq_bridge,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9  # Set high to avoid KV cache negative calculation on heavy models
        )
    else:
        interface = Qwen3TTSInterface.from_pretrained(
            model_name,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        
    _tokenizer = SpeechTokenizerCUDAGraph(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device="cuda:0",
    )
    current_model_name = model_name

async def switch_model_if_needed(model_name: str):
    global current_model_name, interface, _zmq_bridge, USE_ZMQ, _tokenizer
    if interface is not None and current_model_name != model_name:
        print(f"\n🔄 Switching model from {current_model_name} to {model_name}...")
        if USE_ZMQ and hasattr(interface, 'zmq_bridge') and interface.zmq_bridge:
            await interface.stop_zmq_tasks()
            # Forcefully close ZMQ context to release TCP ports instantly
            if hasattr(interface.zmq_bridge, 'context'):
                interface.zmq_bridge.context.destroy(linger=0)
            interface.zmq_bridge.close()
            
        # Call shutdown BEFORE deleting to ensure LLMEngines invoke exit() and destroy CUDA Graphs
        if hasattr(interface, 'shutdown'):
            interface.shutdown()
            
        if _tokenizer is not None and hasattr(_tokenizer, 'shutdown'):
            _tokenizer.shutdown()
        
        # Free GPU memory and reset torch Dynamo cache to prevent cache_size_limit crash
        del interface
        del _tokenizer
        interface = None
        _tokenizer = None
        _zmq_bridge = None
        
        torch._dynamo.reset()
        
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
        current_model_name = None
        
    if interface is None:
        print(f"\n🚀 Loading model (ZMQ Mode): {model_name}...")
        try:
            load_model(model_name)
            if USE_ZMQ and hasattr(interface, 'zmq_bridge') and interface.zmq_bridge:
                # Start background ZMQ loop (sync context)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(interface.start_zmq_tasks())
                else:
                    asyncio.run(interface.start_zmq_tasks())
            current_model_name = model_name
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"Error loading model:\n{err_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    TARGET_SAMPLE_RATE = 24000
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * TARGET_SAMPLE_RATE / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)

def _decode_batch(codes: list):
    with decode_lock:
        wav_list, sr = _tokenizer.decode([{"audio_codes": codes}])
    wav = wav_list[0]
    wav_24k = _resample_to_24k(wav, sr)
    wav_24k = np.clip(wav_24k, -1.0, 1.0)
    return (wav_24k * 32767.0).astype(np.int16)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    yield
    # Cleanup on shutdown
    global interface, _zmq_bridge, USE_ZMQ
    if interface is not None:
        if USE_ZMQ and hasattr(interface, 'zmq_bridge') and interface.zmq_bridge:
            await interface.stop_zmq_tasks()
            interface.zmq_bridge.close()

app = FastAPI(title="Qwen3-TTS Streaming API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
# These are re-declared here, but the ones above are the actual global state.
# Keeping them for now as per instruction to not make unrelated edits, but ideally they should be removed.
# current_model_name = None
# interface = None
# zmq_bridge = None
# ENFORCE_EAGER = False

# Global lock for safe decoding across async requests
# decode_lock = threading.Lock() # Already declared above

# Store active generators and cancellation events for streaming
# active_streams = {} # Already declared above

# The original load_model is removed as per instruction 1.
# The new load_model and switch_model_if_needed handle the logic.

def generate_wav_header(sample_rate: int, num_channels: int = 1, bit_depth: int = 16) -> bytes:
    """Generate a standard WAV header with an unknown data size (0xFFFFFFFF) for streaming."""
    byte_rate = sample_rate * num_channels * (bit_depth // 8)
    block_align = num_channels * (bit_depth // 8)
    
    header = b'RIFF'
    header += struct.pack('<I', 0xFFFFFFFF) # ChunkSize (unknown)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)         # Subchunk1Size
    header += struct.pack('<H', 1)          # AudioFormat (PCM)
    header += struct.pack('<H', num_channels) # NumChannels
    header += struct.pack('<I', sample_rate)  # SampleRate
    header += struct.pack('<I', byte_rate)    # ByteRate
    header += struct.pack('<H', block_align)  # BlockAlign
    header += struct.pack('<H', bit_depth)    # BitsPerSample
    header += b'data'
    header += struct.pack('<I', 0xFFFFFFFF) # Subchunk2Size (unknown)
    return header

async def audio_stream_generator_async(stream_id: str, request_data: dict):
    """Generator that leverages ZMQ async generator, decodes natively, and yields PCM bytes."""
    global interface
    
    # Send WAV header first (24000 Hz, Mono, 16-bit PCM)
    yield generate_wav_header(24000, 1, 16)
    
    try:
        # We must implement VoiceDesign and VoiceClone Async logic ourselves 
        # because nano-qwen3tts-vllm only shipped with generate_custom_voice_async!
        
        model = request_data["model"]
        text = request_data["text"]
        language = request_data["language"]
        instruction = request_data["instruction"]
        speaker = request_data["speaker"]
        ref_audio = request_data.get("ref_audio")
        ref_sr = request_data.get("ref_sr")
        ref_text = request_data.get("ref_text")

        if "VoiceDesign" in model:
            # 1. Custom ZMQ wrapper for Voice Design
            def _prep_voice_design() -> tuple:
                with interface._prep_lock:
                    input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
                        text=[text], speaker=[""], language=[language], instruct=[instruction],
                        processor=interface.processor, device=interface.device,
                    )
                    return prepare_inputs(
                        config=interface.model_config,
                        input_ids=input_ids, instruct_ids=instruct_ids, languages=languages,
                        speakers=None, non_streaming_mode=True,
                        text_embedding=interface.text_embedding, input_embedding=interface.input_embedding,
                        text_projection=interface.text_projection, device=interface.device,
                    )
            loop = asyncio.get_event_loop()
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(None, _prep_voice_design)
            async_gen = interface.generate_async(talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask)
            
        elif "CustomVoice" in model:
            async_gen = interface.generate_custom_voice_async(text=text, language=language, speaker=speaker)
            
        elif "Base" in model:
            # 1. First, correct the `create_voice_clone_prompt` signature perfectly matching examples
            prompt = interface.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_sr),
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=False
            )
            
            # 2. Custom ZMQ wrapper for Voice Clone
            def _prep_voice_clone() -> tuple:
                with interface._prep_lock:
                    input_txt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
                    input_ids = _tokenize_texts([input_txt], interface.processor, interface.device)
                    
                    ref_ids = None
                    if ref_text and ref_text.strip():
                        ref_ids = [_tokenize_texts([interface._build_ref_text(ref_text)], interface.processor, interface.device)[0]]
                        
                    voice_clone_prompt_lists = {
                        "ref_code": [prompt["ref_code"]],
                        "ref_spk_embedding": [prompt["ref_spk_embedding"]],
                        "x_vector_only_mode": [prompt["x_vector_only_mode"]],
                        "icl_mode": [prompt["icl_mode"]],
                    }
                    
                    def generate_speaker_prompt_fn(p, **kwargs):
                        return generate_speaker_prompt(p, interface.device)
                    def generate_icl_prompt_fn(text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed, non_streaming_mode, **kwargs):
                        return generate_icl_prompt(
                            text_id=text_id, ref_id=ref_id, ref_code=ref_code, tts_pad_embed=tts_pad_embed,
                            tts_eos_embed=tts_eos_embed, non_streaming_mode=non_streaming_mode,
                            config=interface.model_config, text_embedding=interface.text_embedding,
                            input_embedding=interface.input_embedding, text_projection=interface.text_projection,
                            code_predictor_embeddings=interface.predictor_input_embeddings, device=interface.device,
                        )
                        
                    return prepare_inputs(
                        config=interface.model_config,
                        input_ids=input_ids,
                        ref_ids=ref_ids,
                        voice_clone_prompt=voice_clone_prompt_lists,
                        languages=[language],
                        non_streaming_mode=True,
                        text_embedding=interface.text_embedding,
                        input_embedding=interface.input_embedding,
                        text_projection=interface.text_projection,
                        device=interface.device,
                        generate_speaker_prompt_fn=generate_speaker_prompt_fn,
                        generate_icl_prompt_fn=generate_icl_prompt_fn,
                    )
                    
            loop = asyncio.get_event_loop()
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(None, _prep_voice_clone)
            async_gen = interface.generate_async(talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask)
        else:
            return

        # ZMQ async loop bridging
        # To avoid blocking event loop with decode, we use a Producer-Consumer architecture:
        
        codes_queue = asyncio.Queue(maxsize=4) # backpressure
        loop = asyncio.get_event_loop()
        cancel_event = request_data.get("cancel_event", asyncio.Event())
        
        print(f"\n[STREAM {stream_id[:8]}] Started generating: {model}")
        
        async def producer():
            audio_codes = []
            chunk_count = 0
            start_time = time.time()
            first_chunk_time = None
            last_chunk_time = None
            
            try:
                async for chunk in async_gen:
                    if cancel_event.is_set():
                        print(f"[STREAM {stream_id[:8]}] 🛑 Generation cancelled by user.")
                        break
                        
                    current_time = time.time()
                    chunk_count += 1
                    
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        ttft = first_chunk_time - start_time
                        print(f"[STREAM {stream_id[:8]}] ⚡ First chunk received! TTFT: {ttft*1000:.2f}ms")
                    else:
                        latency = current_time - last_chunk_time
                        if latency > 1.0:
                            print(f"[STREAM {stream_id[:8]}] >> Chunk #{chunk_count} arrived (+{latency*1000:.2f}ms) ⚠️")
                    
                    last_chunk_time = current_time
                    audio_codes.append(chunk)
                    
                    # Offload to queue every 4 chunks (Exactly like official repo)
                    if len(audio_codes) % 4 == 0:
                        await codes_queue.put(list(audio_codes))
                
                # Push any last remaining chunks
                if len(audio_codes) % 4 != 0:
                    await codes_queue.put(list(audio_codes))
                    
            except Exception as e:
                print(f"[STREAM {stream_id[:8]}] Producer error: {e}")
                traceback.print_exc()
            finally:
                await codes_queue.put(None)  # Sentinel to denote generation end

        producer_task = asyncio.create_task(producer())
        total_start = time.time()
        prev_len_chunks = 0
        
        try:
            while True:
                if cancel_event.is_set():
                     break
                     
                item = await codes_queue.get()
                if item is None:
                    break
                
                num_total_chunks = len(item)
                new_chunks_count = num_total_chunks - prev_len_chunks
                
                # O(1) SLIDING WINDOW: Limit context to the last 48 chunks.
                # Why 48? The `SpeechTokenizerCUDAGraph` natively caches execution graphs for T <= 50.
                # Passing the full history natively breaks the CUDA Graph limit, forcing slow eager evaluation
                # and causing O(N^2) latency (which backs up the queue and stalls the stream).
                MAX_WINDOW = 48
                codes_to_decode = item[-MAX_WINDOW:] if num_total_chunks > MAX_WINDOW else item

                pcm16 = await loop.run_in_executor(None, _decode_batch, codes_to_decode)
                
                # Neural vocoders map sequence lengths symmetrically. We dynamically measure the exact
                # samples-per-chunk mapping (e.g., 1920 at 24kHz) to mathematically perfectly truncate the
                # left-historical context, yielding ONLY the mathematically perfect newly synthesized audio tail.
                spc = len(pcm16) // len(codes_to_decode)
                new_samples_count = new_chunks_count * spc
                
                if new_samples_count > 0:
                    new_chunk_array = pcm16[-new_samples_count:]
                    new_chunk = new_chunk_array.tobytes()
                    prev_len_chunks = num_total_chunks
                    
                    if new_chunk:
                        yield new_chunk
                    
            print(f"[STREAM {stream_id[:8]}] 🏁 Stream complete. Total Time: {(time.time() - total_start):.2f}s")
        finally:
            # Ensure producer task is cancelled if consumer loop breaks early
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass # Expected if cancelled
            except Exception as e:
                print(f"[STREAM {stream_id[:8]}] Error awaiting producer task: {e}")

    except Exception as e:
        print(f"Streaming error: {e}")
        traceback.print_exc()

    finally:
        if stream_id in active_streams:
            del active_streams[stream_id]

@app.post("/api/prepare")
async def prepare_stream(
    model: str = Form(...),
    text: str = Form(...),
    language: str = Form(...),
    instruction: str = Form(""),
    speaker: str = Form(""),
    ref_audio: UploadFile = File(None),
    ref_text: str = Form("")
):
    """
    Endpoint 1: Receives form data and initializes the model.
    """
    try:
        await switch_model_if_needed(model)
    except HTTPException as e:
        raise e # Re-raise the HTTPException from switch_model_if_needed

    stream_id = str(uuid.uuid4())
    
# We delay the generator creation to the GET request because async generation 
    # must be instantiated inside the same event loop task the StreamingResponse consumes it from.
    request_data = {
        "model": model,
        "text": text,
        "language": language,
        "instruction": instruction,
        "speaker": speaker,
        "cancel_event": asyncio.Event()
    }
    
    if "Base" in model:
        if not ref_audio:
            raise HTTPException(status_code=400, detail="Base model requires ref_audio")
        
        audio_bytes = await ref_audio.read()
        ref_wav, ref_sr = sf.read(io.BytesIO(audio_bytes))
        request_data["ref_audio"] = ref_wav
        request_data["ref_sr"] = ref_sr
        
        if not ref_text.strip():
            print(f"[STREAM {stream_id[:8]}] 🤖 Auto-transcribing reference audio with Whisper...")
            try:
                from faster_whisper import WhisperModel
                import tempfile
                
                # Faster-whisper accepts file paths or 16000Hz numpy arrays.
                # Writing exactly to disk is the safest parsing method for av.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                    
                # Load small model on GPU (takes ~1-2 secs first time, caches in VRAM)
                whisper_model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
                segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
                
                # Combine all segments
                transcribed_text = " ".join([segment.text for segment in segments]).strip()
                os.unlink(tmp_path)
                
                if not transcribed_text:
                    raise ValueError("Whisper transcribed an empty string")
                    
                print(f"[STREAM {stream_id[:8]}] 📝 Auto-transcription result: '{transcribed_text}'")
                request_data["ref_text"] = transcribed_text
            except Exception as e:
                print(f"[STREAM {stream_id[:8]}] ❌ Auto-transcription failed: {e}")
                raise HTTPException(status_code=400, detail="Reference text was empty and Auto-Transcription failed. Please provide text manually.")
        else:
            request_data["ref_text"] = ref_text

    active_streams[stream_id] = request_data
    return {"stream_id": stream_id, "message": "Ready to stream"}

@app.get("/api/stream/{stream_id}")
async def stream_tts(stream_id: str):
    """
    Endpoint 2: Client connects via GET stream endpoint (or <audio src="...">) to receive raw audio chunks.
    """
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream ID not found or already consumed")
        
    request_data = active_streams[stream_id]
    return StreamingResponse(audio_stream_generator_async(stream_id, request_data), media_type="audio/wav")

@app.post("/api/cancel/{stream_id}")
async def cancel_stream(stream_id: str):
    """
    Endpoint 3: Cancel an ongoing stream.
    """
    if stream_id in active_streams:
        request_data = active_streams[stream_id]
        if "cancel_event" in request_data:
            request_data["cancel_event"].set()
            return {"message": "Stream cancellation requested"}
    return {"message": "Stream not found or already cancelled"}


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
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 24px;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            flex: 1;
            transition: 0.2s;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        .btn-stop { background-color: #f44336; }
        .btn-stop:hover { background-color: #da190b; }
        #audioPlayer { width: 100%; margin-top: 24px; display: none; }
        .dynamic-field { display: none; }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
            display: none; /* Hidden by default */
        }
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
        
        // Init UI state
        updateUI();

        let currentStreamId = null;
        let audioContext = null; // Not used in this version, but kept from snippet

        async function stopTTS() {
            if (currentStreamId) {
                try {
                    await fetch(`/api/cancel/${currentStreamId}`, { method: 'POST' });
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = "Воспроизведение остановлено вручную.";
                    statusDiv.className = "status error";
                    statusDiv.style.display = 'block';
                    
                    // Stop HTML5 audio player
                    const player = document.getElementById('audioPlayer');
                    player.pause();
                    player.currentTime = 0;
                    player.src = "";
                    
                } catch (e) {
                    console.error("Cancel failed:", e);
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
            player.style.display = 'none'; // Hide player until stream starts

            try {
                const formData = new FormData();
                formData.append('model', document.getElementById('modelSelect').value);
                formData.append('text', document.getElementById('textInput').value);
                formData.append('language', document.getElementById('languageSelect').value);
                formData.append('instruction', document.getElementById('instruction').value);
                formData.append('speaker', document.getElementById('speaker').value);

                const refAudioInput = document.getElementById('ref_audio');
                if (refAudioInput && refAudioInput.files.length > 0) {
                    formData.append('ref_audio', refAudioInput.files[0]);
                    formData.append('ref_text', document.getElementById('ref_text').value);
                }

                // 1. Send inputs to the server to initialize the generator and allocate memory
                const t0 = performance.now();
                const response = await fetch('/api/prepare', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Неизвестная ошибка API");
                }
                
                const t1 = performance.now();
                
                // 2. We received a stream_id! Now connect the HTML5 Audio player directly to the stream.
                // The browser will handle the HTTP chunked stream and start playing immediately!
                currentStreamId = data.stream_id;
                statusDiv.textContent = `Подготовка заняла ${Math.round(t1 - t0)}мс. Подключаемся к потоку звука... Слушайте!`;
                statusDiv.className = 'status success';
                
                player.src = `/api/stream/${currentStreamId}`;
                player.style.display = 'block';
                player.play().catch(e => console.error("Автовоспроизведение заблокировано:", e));
                
                // Setup event listeners for audio end/error
                player.onended = () => {
                    statusDiv.textContent = "Воспроизведение завершено.";
                    statusDiv.className = "status success";
                    resetButtons();
                };
                player.onerror = (e) => {
                    console.error("Ошибка воспроизведения аудио:", e);
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
    """Serves the main HTML Web UI interface."""
    return HTMLResponse(content=HTML_UI, status_code=200)

if __name__ == "__main__":
    print("\nStarting Qwen3-TTS Streaming API Server...")
    print("Go to: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
