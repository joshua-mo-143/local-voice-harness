from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .config import BackendSettings
from .credentials import get_venice_api_key
from .diagnostic_safety import redact_diagnostic, redact_fields
from .errors import HarnessError
from .user_config import default_user_config

_SENTENCE = re.compile(r'^(.+?[.!?]["\']?)(?:\s+)', re.DOTALL)


@dataclass(frozen=True)
class LlmTransportConfig:
    provider: str
    model: str
    endpoint: str
    timeout: float
    api_key: str | None = None


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: Sequence[Mapping[str, object]]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool | None = None
    tools: Sequence[Mapping[str, object]] | None = None
    tool_choice: object | None = None
    parallel_tool_calls: bool | None = None


def _log_transport_event(event: str, **fields: object) -> None:
    print(
        json.dumps(
            redact_fields({"stage": "llm", "event": event, **fields}, limit=None),
            ensure_ascii=False,
        ),
        flush=True,
    )


def response_message(result: object) -> dict[str, object]:
    if not isinstance(result, dict):
        raise HarnessError("LLM returned a malformed response")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise HarnessError("LLM returned a malformed response")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise HarnessError("LLM returned a malformed response")
    return message


TextChunkCallback = Callable[[str], bool | None]
CancellationCheck = Callable[[], bool]


def _chunk_callback(
    callback: TextChunkCallback | None,
    content_filter: Callable[[str], str] | None,
) -> TextChunkCallback | None:
    if callback is None:
        return None

    def emit(text: str) -> bool | None:
        spoken = content_filter(text) if content_filter is not None else text
        if spoken:
            return callback(spoken)
        return None

    return emit


class _TextChunker:
    def __init__(self, callback: TextChunkCallback | None, *, emit_early: bool) -> None:
        self.callback = callback
        self.emit_early = emit_early
        self.buffer = ""
        self._held = ""

    def _emit(self, text: str) -> bool:
        if self.callback is None:
            return False
        if self.callback(text) is False:
            self.discard()
            return False
        return True

    def feed(self, text: str) -> bool:
        if self.callback is None:
            return True
        self.buffer += text
        if not self.emit_early:
            return True
        while match := _SENTENCE.match(self.buffer):
            chunk = match.group(1).strip()
            self.buffer = self.buffer[match.end() :]
            if not chunk:
                continue
            if self.emit_early and self._held and not self._emit(self._held):
                return False
            self._held = chunk
        if self.emit_early and self._held and self.buffer:
            if not self._emit(self._held):
                return False
            self._held = ""
        return True

    def flush(self) -> bool:
        if self.callback is None:
            self.buffer = ""
            self._held = ""
            return False
        if self._held:
            if not self._emit(self._held):
                return False
            self._held = ""
        chunk = self.buffer.strip()
        self.buffer = ""
        return not chunk or self._emit(chunk)

    def discard(self) -> None:
        self.callback = None
        self.buffer = ""
        self._held = ""


def streamed_message(
    response: Iterable[bytes | str],
    on_text_chunk: TextChunkCallback | None,
    *,
    content_filter: Callable[[str], str] | None = None,
    emit_text_early: bool = False,
    should_cancel: CancellationCheck | None = None,
) -> dict[str, object]:
    content: list[str] = []
    tool_calls: dict[int, dict[str, object]] = {}
    chunker = _TextChunker(
        _chunk_callback(on_text_chunk, content_filter),
        emit_early=emit_text_early,
    )
    received_event = False
    cancelled = False
    for raw_line in response:
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        line = (
            raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        ).strip()
        if not line or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise HarnessError("LLM returned a malformed streaming event")
        if event.get("error"):
            detail = redact_diagnostic(event["error"])
            raise HarnessError(f"LLM streaming request failed: {detail}")
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            raise HarnessError("LLM returned a malformed streaming event")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        received_event = True
        delta_tools = delta.get("tool_calls")
        if delta_tools is not None:
            if not isinstance(delta_tools, list):
                raise HarnessError("LLM returned malformed streaming tool calls")
            for delta_call in delta_tools:
                if not isinstance(delta_call, dict):
                    raise HarnessError("LLM returned malformed streaming tool calls")
                try:
                    index = int(delta_call.get("index", 0))
                except (TypeError, ValueError) as exc:
                    raise HarnessError(
                        "LLM returned malformed streaming tool calls"
                    ) from exc
                call = tool_calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if delta_call.get("id") is not None:
                    call["id"] = str(call["id"]) + str(delta_call["id"])
                if delta_call.get("type") is not None:
                    call["type"] = str(delta_call["type"])
                function_delta = delta_call.get("function")
                if function_delta is not None:
                    if not isinstance(function_delta, dict):
                        raise HarnessError(
                            "LLM returned malformed streaming tool calls"
                        )
                    function = call["function"]
                    assert isinstance(function, dict)
                    for key in ("name", "arguments"):
                        if function_delta.get(key) is not None:
                            function[key] = str(function[key]) + str(
                                function_delta[key]
                            )
            chunker.discard()
        delta_content = delta.get("content")
        if delta_content is not None:
            content.append(str(delta_content))
            if not tool_calls and not chunker.feed(str(delta_content)):
                cancelled = True
                break
    if not received_event:
        raise HarnessError("LLM returned an empty streaming response")
    answer = "".join(content)
    if answer and not tool_calls:
        if content_filter is not None:
            answer = content_filter(answer)
        if not cancelled:
            chunker.flush()
    message: dict[str, object] = {"content": answer or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return message


class LlmTransport:
    def __init__(self, config: LlmTransportConfig) -> None:
        self._config = config

    @classmethod
    def from_settings(cls, settings: BackendSettings | None = None) -> LlmTransport:
        resolved = settings or default_user_config().providers
        api_key = get_venice_api_key() if resolved.llm_provider == "venice" else None
        return cls(
            LlmTransportConfig(
                provider=resolved.llm_provider,
                model=resolved.llm_model,
                endpoint=resolved.llm_endpoint,
                timeout=resolved.llm_timeout,
                api_key=api_key,
            )
        )

    @property
    def config(self) -> LlmTransportConfig:
        return self._config

    def _build_payload(self, request: ChatCompletionRequest) -> dict[str, object]:
        stream = (
            request.stream
            if request.stream is not None
            else self._config.provider == "venice"
        )
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": list(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.tools is not None:
            payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if self._config.provider == "venice":
            payload["reasoning"] = {"enabled": False}
        return payload

    def _build_headers(self, *, stream: bool) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
            if stream:
                headers["Accept"] = "text/event-stream"
        return headers

    def _translate_http_error(self, exc: urllib.error.HTTPError) -> HarnessError:
        try:
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        detail = redact_diagnostic(detail)
        suffix = f": {detail}" if detail else ""
        return HarnessError(f"LLM request failed: HTTP {exc.code} {exc.reason}{suffix}")

    def chat_completion(
        self,
        request: ChatCompletionRequest,
        *,
        on_text_chunk: TextChunkCallback | None = None,
        should_cancel: CancellationCheck | None = None,
        content_filter: Callable[[str], str] | None = None,
        telemetry_round: int | None = None,
    ) -> dict[str, object]:
        payload_dict = self._build_payload(request)
        stream = bool(payload_dict["stream"])
        payload = json.dumps(payload_dict).encode()
        if telemetry_round is not None:
            _log_transport_event(
                "request",
                round=telemetry_round,
                payload=payload.decode(),
            )
        headers = self._build_headers(stream=stream)
        http_request = urllib.request.Request(
            self._config.endpoint,
            data=payload,
            headers=headers,
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self._config.timeout
            ) as response:
                message = (
                    streamed_message(
                        response,
                        on_text_chunk,
                        content_filter=content_filter,
                        emit_text_early=request.tools is None,
                        should_cancel=should_cancel,
                    )
                    if stream
                    else response_message(json.load(response))
                )
                if telemetry_round is not None:
                    _log_transport_event(
                        "aggregated_response",
                        round=telemetry_round,
                        response=message,
                    )
        except urllib.error.HTTPError as exc:
            raise self._translate_http_error(exc) from exc
        except TimeoutError as exc:
            raise HarnessError(
                f"LLM request failed: timed out after {self._config.timeout}s"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HarnessError(f"LLM request failed: {exc}") from exc
        print(
            json.dumps(
                {
                    "stage": "llm",
                    "round": telemetry_round,
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
        )
        return message
