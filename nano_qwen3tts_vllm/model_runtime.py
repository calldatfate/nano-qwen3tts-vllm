from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import traceback
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException


class ModelRuntime:
    """Lazy-loaded model runtime for the streaming API server."""

    def __init__(self, *, enforce_eager: bool = False, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._enforce_eager = enforce_eager
        self.current_model_name: str | None = None
        self.interface: Any | None = None
        self.zmq_bridge: Any | None = None
        self.tokenizer: Any | None = None
        self.use_zmq = False
        self.decode_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.interface is not None

    def load_model(self, model_name: str) -> None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph
        from nano_qwen3tts_vllm.zmq import ZMQOutputBridge

        self.use_zmq = os.environ.get("USE_ZMQ", "1") == "1"

        if self.use_zmq:
            self.zmq_bridge = ZMQOutputBridge(auto_find_port=True)
            self.interface = Qwen3TTSInterface.from_pretrained(
                model_name,
                zmq_bridge=self.zmq_bridge,
                enforce_eager=self._enforce_eager,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            )
        else:
            self.interface = Qwen3TTSInterface.from_pretrained(
                model_name,
                enforce_eager=self._enforce_eager,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            )

        self.tokenizer = SpeechTokenizerCUDAGraph(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device="cuda:0",
        )
        self.current_model_name = model_name
        self._logger.info("Loaded Qwen model=%s use_zmq=%s", model_name, self.use_zmq)

    def dispose_loaded_model_state(self) -> None:
        if self.interface is not None and hasattr(self.interface, "shutdown"):
            self.interface.shutdown()

        if self.tokenizer is not None and hasattr(self.tokenizer, "shutdown"):
            self.tokenizer.shutdown()

        self.interface = None
        self.tokenizer = None
        self.zmq_bridge = None
        self.current_model_name = None

        torch._dynamo.reset()
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    async def switch_model_if_needed(self, model_name: str) -> None:
        if self.interface is not None and self.current_model_name != model_name:
            self._logger.info(
                "Switching Qwen model from %s to %s",
                self.current_model_name,
                model_name,
            )
            if self.use_zmq and hasattr(self.interface, "zmq_bridge") and self.interface.zmq_bridge:
                await self.interface.stop_zmq_tasks()
                if hasattr(self.interface.zmq_bridge, "context"):
                    self.interface.zmq_bridge.context.destroy(linger=0)
                self.interface.zmq_bridge.close()

            await asyncio.to_thread(self.dispose_loaded_model_state)

        if self.interface is None:
            self._logger.info("Loading Qwen model=%s", model_name)
            try:
                await asyncio.to_thread(self.load_model, model_name)
                if self.use_zmq and hasattr(self.interface, "zmq_bridge") and self.interface.zmq_bridge:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.interface.start_zmq_tasks())
                    else:
                        asyncio.run(self.interface.start_zmq_tasks())
            except Exception as error:
                self._logger.error("Error loading model:\n%s", traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to load model: {error}") from error

    async def shutdown(self) -> None:
        if self.interface is not None and self.use_zmq and hasattr(self.interface, "zmq_bridge") and self.interface.zmq_bridge:
            await self.interface.stop_zmq_tasks()
            self.interface.zmq_bridge.close()
        if self.interface is not None or self.tokenizer is not None:
            await asyncio.to_thread(self.dispose_loaded_model_state)

    def decode_batch(self, codes: list[Any]) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")
        with self.decode_lock:
            wav_list, sample_rate = self.tokenizer.decode([{"audio_codes": codes}])
        wav = wav_list[0]
        wav_24k = self._resample_to_24k(wav, sample_rate)
        wav_24k = np.clip(wav_24k, -1.0, 1.0)
        return (wav_24k * 32767.0).astype(np.int16)

    @staticmethod
    def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
        target_sample_rate = 24000
        if orig_sr == target_sample_rate:
            return wav
        n_orig = len(wav)
        n_new = int(round(n_orig * target_sample_rate / orig_sr))
        if n_new == 0:
            return wav
        indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
        return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)

