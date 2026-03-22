from __future__ import annotations

import asyncio
import io
import logging
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from .stream_scheduler import FairStreamScheduler
from .utils.generation import generate_icl_prompt, generate_speaker_prompt, prepare_inputs
from .utils.prompt import _tokenize_texts, prepare_custom_voice_prompt
from .voice_service import resolve_base_model_reference_voice
from .voice_uploads import transcribe_voice_file


class StreamService:
    def __init__(
        self,
        *,
        runtime: Any,
        scheduler: FairStreamScheduler,
        model_switch_lock: asyncio.Lock,
        switch_model_if_needed: Callable[[str], Awaitable[None]],
        resolve_runtime_model: Callable[..., str],
        decode_batch: Callable[[list], Any],
        logger: logging.Logger,
    ) -> None:
        self.runtime = runtime
        self.scheduler = scheduler
        self.model_switch_lock = model_switch_lock
        self.switch_model_if_needed = switch_model_if_needed
        self.resolve_runtime_model = resolve_runtime_model
        self.decode_batch = decode_batch
        self.logger = logger

    @staticmethod
    def generate_wav_header(sample_rate: int, num_channels: int = 1, bit_depth: int = 16) -> bytes:
        byte_rate = sample_rate * num_channels * (bit_depth // 8)
        block_align = num_channels * (bit_depth // 8)

        header = b"RIFF"
        header += struct.pack("<I", 0xFFFFFFFF)
        header += b"WAVE"
        header += b"fmt "
        header += struct.pack("<I", 16)
        header += struct.pack("<H", 1)
        header += struct.pack("<H", num_channels)
        header += struct.pack("<I", sample_rate)
        header += struct.pack("<I", byte_rate)
        header += struct.pack("<H", block_align)
        header += struct.pack("<H", bit_depth)
        header += b"data"
        header += struct.pack("<I", 0xFFFFFFFF)
        return header

    async def audio_stream_generator_async(self, stream_id: str, request_data: dict) -> AsyncIterator[bytes]:
        final_state = "finished"
        final_error = None

        yield self.generate_wav_header(24000, 1, 16)

        try:
            model = request_data["model"]
            text = request_data["text"]
            language = request_data["language"]
            instruction = request_data["instruction"]
            speaker = request_data["speaker"]
            temperature = request_data.get("temperature", 0.9)
            ref_audio = request_data.get("ref_audio")
            ref_sr = request_data.get("ref_sr")
            ref_text = request_data.get("ref_text")
            interface = self.runtime.interface
            if interface is None:
                raise RuntimeError("Model runtime is not loaded")

            if "VoiceDesign" in model:
                def _prep_voice_design() -> tuple:
                    with interface._prep_lock:
                        input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
                            text=[text],
                            speaker=[""],
                            language=[language],
                            instruct=[instruction],
                            processor=interface.processor,
                            device=interface.device,
                        )
                        return prepare_inputs(
                            config=interface.model_config,
                            input_ids=input_ids,
                            instruct_ids=instruct_ids,
                            languages=languages,
                            speakers=None,
                            non_streaming_mode=True,
                            text_embedding=interface.text_embedding,
                            input_embedding=interface.input_embedding,
                            text_projection=interface.text_projection,
                            device=interface.device,
                        )

                loop = asyncio.get_running_loop()
                talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = (
                    await loop.run_in_executor(None, _prep_voice_design)
                )
                async_gen = interface.generate_async(
                    talker_input_embeds,
                    trailing_text_hiddens,
                    tts_pad_embed,
                    talker_attention_mask,
                    temperature=temperature,
                )
            elif "CustomVoice" in model:
                async_gen = interface.generate_custom_voice_async(
                    text=text,
                    language=language,
                    speaker=speaker,
                    temperature=temperature,
                )
            elif "Base" in model:
                loop = asyncio.get_running_loop()
                prompt = await loop.run_in_executor(
                    None,
                    lambda: interface.create_voice_clone_prompt(
                        ref_audio=(ref_audio, ref_sr),
                        ref_text=ref_text if ref_text else None,
                        x_vector_only_mode=False,
                    ),
                )

                def _prep_voice_clone() -> tuple:
                    with interface._prep_lock:
                        input_txt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
                        input_ids = _tokenize_texts([input_txt], interface.processor, interface.device)

                        ref_ids = None
                        if ref_text and ref_text.strip():
                            ref_ids = [
                                _tokenize_texts(
                                    [interface._build_ref_text(ref_text)],
                                    interface.processor,
                                    interface.device,
                                )[0]
                            ]

                        voice_clone_prompt_lists = {
                            "ref_code": [prompt["ref_code"]],
                            "ref_spk_embedding": [prompt["ref_spk_embedding"]],
                            "x_vector_only_mode": [prompt["x_vector_only_mode"]],
                            "icl_mode": [prompt["icl_mode"]],
                        }

                        def generate_speaker_prompt_fn(p, **kwargs):
                            return generate_speaker_prompt(p, interface.device)

                        def generate_icl_prompt_fn(
                            text_id,
                            ref_id,
                            ref_code,
                            tts_pad_embed,
                            tts_eos_embed,
                            non_streaming_mode,
                            **kwargs,
                        ):
                            return generate_icl_prompt(
                                text_id=text_id,
                                ref_id=ref_id,
                                ref_code=ref_code,
                                tts_pad_embed=tts_pad_embed,
                                tts_eos_embed=tts_eos_embed,
                                non_streaming_mode=non_streaming_mode,
                                config=interface.model_config,
                                text_embedding=interface.text_embedding,
                                input_embedding=interface.input_embedding,
                                text_projection=interface.text_projection,
                                code_predictor_embeddings=interface.predictor_input_embeddings,
                                device=interface.device,
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

                talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = (
                    await loop.run_in_executor(None, _prep_voice_clone)
                )
                async_gen = interface.generate_async(
                    talker_input_embeds,
                    trailing_text_hiddens,
                    tts_pad_embed,
                    talker_attention_mask,
                    temperature=temperature,
                )
            else:
                return

            codes_queue: asyncio.Queue[list | None] = asyncio.Queue(maxsize=4)
            loop = asyncio.get_running_loop()
            cancel_event = request_data.get("cancel_event", asyncio.Event())

            self.logger.info("[STREAM %s] Started generating model=%s", stream_id[:8], model)

            async def producer() -> None:
                audio_codes = []
                chunk_count = 0
                start_time = time.time()
                first_chunk_time = None
                last_chunk_time = None

                try:
                    async for chunk in async_gen:
                        if cancel_event.is_set():
                            self.logger.info("[STREAM %s] Generation cancelled by user", stream_id[:8])
                            break

                        current_time = time.time()
                        chunk_count += 1

                        if first_chunk_time is None:
                            first_chunk_time = current_time
                            ttft = first_chunk_time - start_time
                            self.logger.info(
                                "[STREAM %s] First chunk received ttft_ms=%.2f",
                                stream_id[:8],
                                ttft * 1000,
                            )
                        else:
                            latency = current_time - last_chunk_time
                            if latency > 1.0:
                                self.logger.warning(
                                    "[STREAM %s] Chunk #%s delayed latency_ms=%.2f",
                                    stream_id[:8],
                                    chunk_count,
                                    latency * 1000,
                                )

                        last_chunk_time = current_time
                        audio_codes.append(chunk)

                        if len(audio_codes) % 4 == 0:
                            await codes_queue.put(list(audio_codes))

                    if len(audio_codes) % 4 != 0:
                        await codes_queue.put(list(audio_codes))
                except Exception as error:
                    self.logger.exception("[STREAM %s] Producer error: %s", stream_id[:8], error)
                finally:
                    await codes_queue.put(None)

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
                    max_window = 48
                    codes_to_decode = item[-max_window:] if num_total_chunks > max_window else item

                    pcm16 = await loop.run_in_executor(None, self.decode_batch, codes_to_decode)
                    spc = len(pcm16) // len(codes_to_decode)
                    new_samples_count = new_chunks_count * spc

                    if new_samples_count > 0:
                        new_chunk_array = pcm16[-new_samples_count:]
                        new_chunk = new_chunk_array.tobytes()
                        prev_len_chunks = num_total_chunks
                        if new_chunk:
                            yield new_chunk

                self.logger.info(
                    "[STREAM %s] Stream complete total_sec=%.2f",
                    stream_id[:8],
                    time.time() - total_start,
                )
            finally:
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
                except Exception as error:
                    self.logger.warning("[STREAM %s] Error awaiting producer task: %s", stream_id[:8], error)
        except Exception as error:
            final_state = "failed"
            final_error = str(error)
            self.logger.exception("Streaming error for stream_id=%s: %s", stream_id, error)
        finally:
            cancel_event = request_data.get("cancel_event")
            if final_state != "failed" and cancel_event is not None and cancel_event.is_set():
                final_state = "cancelled"
            await self.scheduler.mark_stream_done(
                stream_id,
                final_state=final_state,
                error=final_error,
            )

    def build_router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/api/prepare")
        async def prepare_stream(
            request: Request,
            model: str = Form(...),
            text: str = Form(...),
            language: str = Form(...),
            temperature: float = Form(0.9),
            instruction: str = Form(""),
            speaker: str = Form(""),
            ref_audio: UploadFile = File(None),
            ref_text: str = Form(""),
            tenant_id: str = Form(""),
            channel_name: str = Form(""),
            author: str = Form(""),
            user_id: str = Form(""),
            request_id: str = Form(""),
            event_id: str = Form(""),
        ):
            if temperature <= 0:
                raise HTTPException(status_code=400, detail="temperature must be > 0")

            required_family = "base" if "Base" in model else None
            resolved_model = self.resolve_runtime_model(model, required_family=required_family)
            stream_id = str(uuid.uuid4())

            request_data = {
                "model": resolved_model,
                "text": text,
                "language": language,
                "temperature": float(temperature),
                "instruction": instruction,
                "speaker": speaker,
                "cancel_event": asyncio.Event(),
                "channel_name": channel_name,
                "author": author,
                "user_id": user_id,
                "request_id": request_id,
                "event_id": event_id,
            }

            if "Base" in resolved_model:
                if ref_audio:
                    audio_bytes = await ref_audio.read()
                    ref_wav, ref_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    if getattr(ref_wav, "ndim", 1) > 1:
                        ref_wav = np.mean(ref_wav, axis=1).astype(np.float32)
                    request_data["ref_audio"] = ref_wav
                    request_data["ref_sr"] = ref_sr

                    resolved_ref_text = str(ref_text or "").strip()
                    if not resolved_ref_text:
                        self.logger.info("[STREAM %s] Auto-transcribing uploaded reference audio", stream_id[:8])
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp.write(audio_bytes)
                                tmp_path = Path(tmp.name)
                            try:
                                resolved_ref_text = await transcribe_voice_file(tmp_path)
                            finally:
                                tmp_path.unlink(missing_ok=True)
                            self.logger.info(
                                "[STREAM %s] Auto-transcription result=%r",
                                stream_id[:8],
                                resolved_ref_text,
                            )
                        except Exception as error:
                            self.logger.warning("[STREAM %s] Auto-transcription failed: %s", stream_id[:8], error)
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "Reference text was empty and auto-transcription failed. "
                                    "Please provide text manually."
                                ),
                            ) from error
                    request_data["ref_text"] = resolved_ref_text
                else:
                    voice_record, ref_wav, ref_sr = await resolve_base_model_reference_voice(
                        request.app,
                        user_id_raw=user_id,
                        requested_voice=speaker,
                    )
                    request_data["ref_audio"] = ref_wav
                    request_data["ref_sr"] = ref_sr
                    request_data["speaker"] = str(voice_record.get("name") or speaker or "")

                    resolved_ref_text = str(ref_text or voice_record.get("reference_text") or "").strip()
                    if not resolved_ref_text:
                        voice_path = Path(str(voice_record.get("file_path") or "")).resolve()
                        try:
                            resolved_ref_text = await transcribe_voice_file(voice_path)
                            await request.app.state.voice_store.update_voice_settings(
                                int(voice_record["id"]),
                                {"reference_text": resolved_ref_text},
                            )
                        except Exception as error:
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "Stored Qwen clone voice has no reference_text and retranscription failed."
                                ),
                            ) from error
                    request_data["ref_text"] = resolved_ref_text

            return await self.scheduler.enqueue(
                stream_id=stream_id,
                request_data=request_data,
                tenant_id=tenant_id,
                channel_name=channel_name,
            )

        @router.get("/api/stream/{stream_id}")
        async def stream_tts(stream_id: str):
            job = await self.scheduler.wait_until_stream_can_run(stream_id)
            request_data = job["request_data"]

            try:
                async with self.model_switch_lock:
                    await self.switch_model_if_needed(request_data["model"])
            except HTTPException as error:
                await self.scheduler.mark_stream_done(
                    stream_id,
                    final_state="failed",
                    error=str(error.detail),
                )
                raise error
            except Exception as error:
                await self.scheduler.mark_stream_done(
                    stream_id,
                    final_state="failed",
                    error=str(error),
                )
                raise HTTPException(status_code=500, detail=f"Failed to load model for stream: {error}") from error

            async with self.scheduler.queue_condition:
                if job.get("stream_opened"):
                    raise HTTPException(status_code=409, detail="Stream already consumed")
                job["stream_opened"] = True

            return StreamingResponse(
                self.audio_stream_generator_async(stream_id, request_data),
                media_type="audio/wav",
            )

        @router.get("/api/status/{stream_id}")
        async def stream_status(stream_id: str):
            return await self.scheduler.status(stream_id)

        @router.post("/api/cancel/{stream_id}")
        async def cancel_stream(stream_id: str):
            return await self.scheduler.cancel(stream_id)

        return router
