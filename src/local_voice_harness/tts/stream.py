from __future__ import annotations

import json

from ..errors import HarnessError

STREAM_POLL_SECONDS = 0.08
STREAM_TIMEOUT_SECONDS = 120.0


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessError(f"TTS stream {field} must be an integer")
    if positive and value <= 0:
        raise HarnessError(f"TTS stream {field} must be positive")
    if not positive and value < 0:
        raise HarnessError(f"TTS stream {field} must not be negative")
    return value


class TTSStreamParser:
    """Incrementally validate one newline-delimited TTS stream."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._buffer = bytearray()
        self._started = False
        self._done = False
        self._sample_rate = 0
        self._expected_chunks = 0
        self._chunks = 0
        self._done_event: dict[str, object] | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def done_event(self) -> dict[str, object]:
        if self._done_event is None:
            raise HarnessError("TTS stream ended before completion")
        return self._done_event

    def feed(self, data: bytes) -> list[dict[str, object]]:
        self._buffer.extend(data)
        events: list[dict[str, object]] = []
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer[:] = remainder
            if not line:
                raise HarnessError("TTS backend returned an empty stream event")
            try:
                decoded = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise HarnessError(
                    "TTS backend returned an invalid stream event"
                ) from exc
            if not isinstance(decoded, dict):
                raise HarnessError("TTS backend returned a non-object stream event")
            event: dict[str, object] = decoded
            self._validate_event(event)
            events.append(event)
        return events

    def finish(self) -> dict[str, object]:
        if self._buffer:
            raise HarnessError("TTS backend returned an incomplete stream event")
        if not self._done:
            raise HarnessError("TTS stream ended before completion")
        done = self.done_event
        if done["cancelled"]:
            raise HarnessError("TTS stream was cancelled")
        return done

    def _validate_event(self, event: dict[str, object]) -> None:
        if event.get("ok") is not True:
            raise HarnessError(
                f"TTS backend failed: {event.get('error', 'unknown error')}"
            )
        if self._done:
            raise HarnessError("TTS stream returned an event after completion")
        if event.get("request_id") != self.request_id:
            raise HarnessError("TTS stream request_id did not match the request")

        kind = event.get("event")
        if kind == "start":
            self._start(event)
        elif kind == "chunk":
            self._chunk(event)
        elif kind == "done":
            self._complete(event)
        else:
            raise HarnessError(f"unknown TTS stream event: {kind}")

    def _start(self, event: dict[str, object]) -> None:
        if self._started:
            raise HarnessError("TTS stream returned duplicate start events")
        self._sample_rate = _integer(
            event.get("sample_rate"), "start.sample_rate", positive=True
        )
        self._expected_chunks = _integer(
            event.get("chunks"), "start.chunks", positive=True
        )
        self._started = True

    def _chunk(self, event: dict[str, object]) -> None:
        if not self._started:
            raise HarnessError("TTS stream returned a chunk before start")
        index = _integer(event.get("index"), "chunk.index")
        if index != self._chunks:
            raise HarnessError(
                f"TTS stream chunk index {index} did not match {self._chunks}"
            )
        output = event.get("output")
        if not isinstance(output, str) or not output:
            raise HarnessError("TTS stream chunk.output must be a non-empty string")
        if not isinstance(event.get("text"), str):
            raise HarnessError("TTS stream chunk.text must be a string")
        self._chunks += 1

    def _complete(self, event: dict[str, object]) -> None:
        if not self._started:
            raise HarnessError("TTS stream completed before start")
        cancelled = event.get("cancelled")
        if not isinstance(cancelled, bool):
            raise HarnessError("TTS stream done.cancelled must be a boolean")
        chunks = _integer(event.get("chunks"), "done.chunks")
        if chunks != self._chunks:
            raise HarnessError(
                f"TTS stream completion count {chunks} did not match {self._chunks}"
            )
        if not cancelled and chunks != self._expected_chunks:
            raise HarnessError(
                "TTS stream completion count did not match the start event"
            )
        self._done = True
        self._done_event = event
