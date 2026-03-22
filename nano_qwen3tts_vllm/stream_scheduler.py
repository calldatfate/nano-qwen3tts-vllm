from __future__ import annotations

import asyncio
import time
from collections import deque

from fastapi import HTTPException


class FairStreamScheduler:
    def __init__(
        self,
        *,
        max_queue_per_tenant: int,
        max_total_queued: int,
        stream_wait_timeout_sec: float,
    ) -> None:
        self.max_queue_per_tenant = max_queue_per_tenant
        self.max_total_queued = max_total_queued
        self.stream_wait_timeout_sec = stream_wait_timeout_sec
        self.active_streams: dict[str, dict] = {}
        self.tenant_queues: dict[str, deque[str]] = {}
        self.tenant_rr: deque[str] = deque()
        self.tenant_rr_set: set[str] = set()
        self.queue_condition = asyncio.Condition()
        self.active_stream_id: str | None = None

    @staticmethod
    def normalize_tenant_id(tenant_id: str, channel_name: str) -> str:
        tenant = (tenant_id or "").strip()
        if tenant:
            return tenant
        channel = (channel_name or "").strip()
        if channel:
            return f"channel:{channel.lower()}"
        return "default"

    def _queued_total_locked(self) -> int:
        return sum(len(queue) for queue in self.tenant_queues.values())

    def _remove_tenant_from_rr_locked(self, tenant_id: str) -> None:
        if tenant_id in self.tenant_rr_set:
            self.tenant_rr_set.discard(tenant_id)
            try:
                self.tenant_rr.remove(tenant_id)
            except ValueError:
                pass

    def _prune_tenant_head_locked(self, tenant_id: str):
        queue = self.tenant_queues.get(tenant_id)
        if queue is None:
            return None
        while queue:
            stream_id = queue[0]
            job = self.active_streams.get(stream_id)
            if job is None or job.get("state") != "queued":
                queue.popleft()
                continue
            return job
        self.tenant_queues.pop(tenant_id, None)
        self._remove_tenant_from_rr_locked(tenant_id)
        return None

    def _pop_next_ready_stream_locked(self) -> str | None:
        tenants_count = len(self.tenant_rr)
        for _ in range(tenants_count):
            tenant_id = self.tenant_rr.popleft()
            head_job = self._prune_tenant_head_locked(tenant_id)
            if head_job is None:
                continue

            if not head_job.get("stream_requested", False):
                self.tenant_rr.append(tenant_id)
                continue

            stream_id = self.tenant_queues[tenant_id].popleft()
            head_after = self._prune_tenant_head_locked(tenant_id)
            if head_after is not None:
                self.tenant_rr.append(tenant_id)
            return stream_id
        return None

    def _try_activate_next_locked(self) -> str | None:
        if self.active_stream_id is not None:
            return None

        next_stream_id = self._pop_next_ready_stream_locked()
        if next_stream_id is None:
            return None

        job = self.active_streams.get(next_stream_id)
        if job is None:
            return None

        job["state"] = "running"
        job["started_at"] = time.time()
        self.active_stream_id = next_stream_id
        return next_stream_id

    async def enqueue(
        self,
        *,
        stream_id: str,
        request_data: dict,
        tenant_id: str,
        channel_name: str,
    ) -> dict[str, object]:
        tenant_key = self.normalize_tenant_id(tenant_id, channel_name)

        async with self.queue_condition:
            if self._queued_total_locked() >= self.max_total_queued:
                raise HTTPException(
                    status_code=429,
                    detail=f"Queue is full (MAX_TOTAL_QUEUED={self.max_total_queued})",
                )

            tenant_queue = self.tenant_queues.get(tenant_key)
            if tenant_queue is None:
                tenant_queue = deque()
                self.tenant_queues[tenant_key] = tenant_queue

            if len(tenant_queue) >= self.max_queue_per_tenant:
                raise HTTPException(
                    status_code=429,
                    detail=f"Tenant queue is full (MAX_QUEUE_PER_TENANT={self.max_queue_per_tenant})",
                )

            self.active_streams[stream_id] = {
                "tenant_id": tenant_key,
                "request_data": request_data,
                "state": "queued",
                "stream_requested": False,
                "stream_opened": False,
                "error": None,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
            }
            tenant_queue.append(stream_id)

            if tenant_key not in self.tenant_rr_set:
                self.tenant_rr.append(tenant_key)
                self.tenant_rr_set.add(tenant_key)

            self.queue_condition.notify_all()

            return {
                "stream_id": stream_id,
                "tenant_id": tenant_key,
                "state": "queued",
                "tenant_queue_depth": len(tenant_queue),
                "global_queue_depth": self._queued_total_locked(),
                "message": "Queued. Connect GET /api/stream/{stream_id} and wait for your fair turn.",
            }

    async def wait_until_stream_can_run(self, stream_id: str) -> dict:
        deadline = None
        if self.stream_wait_timeout_sec > 0:
            deadline = time.monotonic() + self.stream_wait_timeout_sec

        async with self.queue_condition:
            job = self.active_streams.get(stream_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Stream ID not found")
            if job.get("state") == "queued":
                job["stream_requested"] = True

            while True:
                job = self.active_streams.get(stream_id)
                if job is None:
                    raise HTTPException(status_code=404, detail="Stream ID not found")

                state = job.get("state")
                if state == "running" and self.active_stream_id == stream_id:
                    return job
                if state == "cancelled":
                    raise HTTPException(status_code=409, detail="Stream was cancelled")
                if state == "failed":
                    raise HTTPException(status_code=500, detail=job.get("error", "Stream failed"))
                if state == "finished":
                    raise HTTPException(status_code=410, detail="Stream already finished")

                self._try_activate_next_locked()
                if job.get("state") == "running" and self.active_stream_id == stream_id:
                    return job

                timeout = None
                if deadline is not None:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise HTTPException(status_code=408, detail="Timed out waiting in queue")

                try:
                    await asyncio.wait_for(self.queue_condition.wait(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise HTTPException(status_code=408, detail="Timed out waiting in queue") from exc

    async def mark_stream_done(
        self,
        stream_id: str,
        *,
        final_state: str,
        error: str | None = None,
    ) -> None:
        async with self.queue_condition:
            job = self.active_streams.get(stream_id)
            if job is not None:
                if job.get("state") not in {"cancelled", "failed"}:
                    job["state"] = final_state
                job["finished_at"] = time.time()
                if error:
                    job["error"] = error

            if self.active_stream_id == stream_id:
                self.active_stream_id = None

            self._try_activate_next_locked()
            self.queue_condition.notify_all()

    async def status(self, stream_id: str) -> dict[str, object]:
        async with self.queue_condition:
            job = self.active_streams.get(stream_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Stream ID not found")
            tenant_id = job["tenant_id"]
            tenant_queue_depth = len(self.tenant_queues.get(tenant_id, ()))
            return {
                "stream_id": stream_id,
                "tenant_id": tenant_id,
                "state": job.get("state"),
                "active_stream_id": self.active_stream_id,
                "tenant_queue_depth": tenant_queue_depth,
                "global_queue_depth": self._queued_total_locked(),
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "error": job.get("error"),
            }

    async def cancel(self, stream_id: str) -> dict[str, object]:
        async with self.queue_condition:
            job = self.active_streams.get(stream_id)
            if job is None:
                return {"message": "Stream not found or already cancelled"}

            request_data = job.get("request_data", {})
            cancel_event = request_data.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()

            state = job.get("state")
            if state == "queued":
                tenant_id = job["tenant_id"]
                queue = self.tenant_queues.get(tenant_id)
                if queue is not None:
                    try:
                        queue.remove(stream_id)
                    except ValueError:
                        pass
                    if not queue:
                        self.tenant_queues.pop(tenant_id, None)
                        self._remove_tenant_from_rr_locked(tenant_id)
                job["state"] = "cancelled"
                job["finished_at"] = time.time()
                self._try_activate_next_locked()
                self.queue_condition.notify_all()
                return {"message": "Queued stream cancelled", "state": "cancelled"}

            if state == "running":
                self.queue_condition.notify_all()
                return {"message": "Stream cancellation requested", "state": "running"}

            return {"message": "Stream already completed", "state": state}
