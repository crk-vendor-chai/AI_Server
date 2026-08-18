from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_REDACTED = "***REDACTED***"
_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "access_key",
    "accesskey",
    "api_key",
    "apikey",
    "mongo_uri",
)


def _redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if key and any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return _REDACTED
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def log_event(event: str, **fields: Any) -> None:
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(
        json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")),
        flush=True,
    )


def _header_value(headers: Dict[str, str], name: str) -> str:
    return headers.get(name.lower(), "")


def _decode_body(
    raw: bytes,
    content_type: str,
    truncated: bool = False,
) -> Any:
    if not raw:
        return None

    text = raw.decode("utf-8", errors="replace")
    is_json = "json" in content_type.lower() or text.lstrip().startswith(("{", "["))
    if is_json:
        try:
            parsed = _redact(json.loads(text))
            if truncated:
                return {"truncated": True, "payload_prefix": parsed}
            return parsed
        except json.JSONDecodeError:
            pass

    result: Dict[str, Any] = {"text": text}
    if truncated:
        result["truncated"] = True
    return result


def request_payload_from_state(request: Any) -> Dict[str, Any]:
    raw = getattr(request.state, "http_request_body", b"")
    content_type = getattr(request.state, "http_request_content_type", "")
    parsed = _decode_body(raw, content_type)
    return parsed if isinstance(parsed, dict) else {}


class HttpTransactionLoggingMiddleware:
    """Capture HTTP request/response bodies without changing the ASGI stream."""

    def __init__(self, app: Any):
        self.app = app
        self.max_body_bytes = max(
            1024,
            int(os.getenv("HTTP_LOG_BODY_MAX_BYTES", "65536")),
        )
        self.log_health = os.getenv("HTTP_LOG_HEALTH", "false").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        excluded = os.getenv(
            "HTTP_LOG_EXCLUDED_PATHS",
            "/health,/docs,/openapi.json,/redoc",
        )
        self.excluded_paths = {path.strip() for path in excluded.split(",") if path.strip()}
        if self.log_health:
            self.excluded_paths.discard("/health")

    def _emit_transaction(
        self,
        *,
        scope: Dict[str, Any],
        request_headers: Dict[str, str],
        request_body: bytes,
        request_truncated: bool,
        response_headers: Dict[str, str],
        response_body: bytes,
        response_truncated: bool,
        response_status: int,
        duration_ms: float,
        exception: Optional[Exception],
    ) -> None:
        request_payload = _decode_body(
            request_body,
            _header_value(request_headers, "content-type"),
            request_truncated,
        )
        response_payload = _decode_body(
            response_body,
            _header_value(response_headers, "content-type"),
            response_truncated,
        )

        request_id = _header_value(request_headers, "x-request-id")
        if isinstance(request_payload, dict):
            request_header = request_payload.get("HEADER", {})
            if isinstance(request_header, dict):
                request_id = str(request_header.get("IF_SYSID") or request_id)
        request_id = request_id or str(uuid.uuid4())
        scope.setdefault("state", {})["http_request_id"] = request_id

        client = scope.get("client")
        client_ip = client[0] if client else ""
        query = scope.get("query_string", b"").decode("utf-8", errors="replace")

        fields: Dict[str, Any] = {
            "level": "error"
            if response_status >= 500
            else "warning"
            if response_status >= 400
            else "info",
            "request_id": request_id,
            "method": scope.get("method", ""),
            "path": scope.get("path", ""),
            "query": query,
            "client_ip": client_ip,
            "status_code": response_status,
            "duration_ms": duration_ms,
            "request_payload": request_payload,
            "response_payload": response_payload,
        }
        if exception is not None:
            fields["exception_type"] = type(exception).__name__
            fields["exception_message"] = str(exception)
        log_event("http_transaction", **fields)

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_body = bytearray()
        response_body = bytearray()
        request_truncated = False
        response_truncated = False
        response_status: Optional[int] = None
        response_headers: Dict[str, str] = {}
        exception: Optional[Exception] = None

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        state = scope.setdefault("state", {})
        state["http_request_content_type"] = _header_value(headers, "content-type")

        async def logging_receive() -> Dict[str, Any]:
            nonlocal request_truncated
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"")
                remaining = self.max_body_bytes - len(request_body)
                if remaining > 0:
                    request_body.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    request_truncated = True
                state["http_request_body"] = bytes(request_body)
            return message

        async def logging_send(message: Dict[str, Any]) -> None:
            nonlocal response_status, response_headers, response_truncated
            if message.get("type") == "http.response.start":
                response_status = int(message.get("status", 500))
                response_headers = {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
            elif message.get("type") == "http.response.body":
                chunk = message.get("body", b"")
                remaining = self.max_body_bytes - len(response_body)
                if remaining > 0:
                    response_body.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    response_truncated = True
            await send(message)

        started = time.perf_counter()
        state["http_started_perf"] = started
        try:
            await self.app(scope, logging_receive, logging_send)
        except Exception as exc:
            exception = exc
            if response_status is None:
                response_status = 500
            raise
        finally:
            path = scope.get("path", "")
            if exception is None and path not in self.excluded_paths:
                self._emit_transaction(
                    scope=scope,
                    request_headers=headers,
                    request_body=bytes(request_body),
                    request_truncated=request_truncated,
                    response_headers=response_headers,
                    response_body=bytes(response_body),
                    response_truncated=response_truncated,
                    response_status=response_status or 500,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    exception=exception,
                )

