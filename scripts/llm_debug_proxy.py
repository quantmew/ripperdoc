#!/usr/bin/env python3
"""Debug reverse proxy for LLM requests.

This proxy forwards requests to an upstream model endpoint and records request/response
metadata into JSONL logs. It is designed to help debug prompt-cache behavior (including
Anthropic cache fields like cache_read_input_tokens).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import http.server
import json
import socketserver
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}

USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "cached_tokens",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> Optional[bytes]:
    try:
        normalized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return normalized.encode("utf-8")


def redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            redacted[key] = _mask_secret(value)
        else:
            redacted[key] = value
    return redacted


def safe_json_loads(raw: bytes) -> Optional[Any]:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def collect_usage_numbers(payload: Any, out: Dict[str, int]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in USAGE_KEYS and isinstance(value, (int, float)):
                out[key] = int(value)
            collect_usage_numbers(value, out)
        return
    if isinstance(payload, list):
        for item in payload:
            collect_usage_numbers(item, out)


def normalize_usage(raw_usage: Dict[str, int]) -> Dict[str, int]:
    usage = dict(raw_usage)
    if "input_tokens" not in usage and "prompt_tokens" in usage:
        usage["input_tokens"] = usage["prompt_tokens"]
    if "output_tokens" not in usage and "completion_tokens" in usage:
        usage["output_tokens"] = usage["completion_tokens"]
    if "cache_read_input_tokens" not in usage and "prompt_cache_hit_tokens" in usage:
        usage["cache_read_input_tokens"] = usage["prompt_cache_hit_tokens"]
    return usage


def summarize_request_json(payload: Any, body_len: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"body_bytes": body_len}
    if not isinstance(payload, dict):
        return summary
    summary["model"] = payload.get("model")
    summary["stream"] = bool(payload.get("stream", False))
    if isinstance(payload.get("messages"), list):
        summary["messages_count"] = len(payload["messages"])
    if isinstance(payload.get("tools"), list):
        summary["tools_count"] = len(payload["tools"])
    if isinstance(payload.get("input"), list):
        summary["input_items"] = len(payload["input"])
    system = payload.get("system")
    if isinstance(system, str):
        summary["system_chars"] = len(system)
    elif isinstance(system, list):
        summary["system_blocks"] = len(system)
    return summary


def extract_request_message_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    out: Dict[str, Any] = {}
    if "system" in payload:
        out["system"] = payload.get("system")
    if "messages" in payload:
        out["messages"] = payload.get("messages")
    if "input" in payload:
        out["input"] = payload.get("input")
    return out or None


def capture_text_for_log(raw: bytes, max_bytes: int) -> Dict[str, Any]:
    total = len(raw)
    if total <= max_bytes:
        return {"truncated": False, "bytes": total, "text": raw.decode("utf-8", errors="replace")}
    return {
        "truncated": True,
        "bytes": total,
        "captured_bytes": max_bytes,
        "text": raw[:max_bytes].decode("utf-8", errors="replace"),
    }


def capture_json_for_log(value: Any, max_bytes: int) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    encoded = canonical_json_bytes(value)
    if encoded is not None:
        if len(encoded) <= max_bytes:
            return {"truncated": False, "bytes": len(encoded), "payload": value}
        return {
            "truncated": True,
            "bytes": len(encoded),
            "captured_bytes": max_bytes,
            "payload_preview": encoded[:max_bytes].decode("utf-8", errors="replace"),
        }
    text = str(value).encode("utf-8", errors="replace")
    return capture_text_for_log(text, max_bytes)


def _collect_text_fragments(value: Any, out: List[str]) -> None:
    if isinstance(value, dict):
        vtype = value.get("type")
        if vtype in {"text", "output_text"} and isinstance(value.get("text"), str):
            out.append(value["text"])
        for item in value.values():
            _collect_text_fragments(item, out)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text_fragments(item, out)


def extract_response_message_payload(payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out

    message = payload.get("message")
    if isinstance(message, dict):
        out["message"] = message

    if "content" in payload:
        out["content"] = payload.get("content")

    output = payload.get("output")
    if output is not None:
        out["output"] = output

    fragments: List[str] = []
    _collect_text_fragments(payload, fragments)
    if fragments:
        out["assistant_text"] = "".join(fragments)
    return out


def build_request_hashes(body: bytes, payload: Any) -> Dict[str, Any]:
    hashes: Dict[str, Any] = {"request_body_sha256": sha256_hex(body)}
    if not isinstance(payload, dict):
        return hashes

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return hashes

    full_messages_bytes = canonical_json_bytes(messages)
    if full_messages_bytes is not None:
        hashes["messages_sha256"] = sha256_hex(full_messages_bytes)

    last_user_index: Optional[int] = None
    for idx in range(len(messages) - 1, -1, -1):
        item = messages[idx]
        if isinstance(item, dict) and item.get("role") == "user":
            last_user_index = idx
            break
    if last_user_index is None:
        return hashes

    without_last_user = list(messages[:last_user_index]) + list(messages[last_user_index + 1 :])
    without_last_user_bytes = canonical_json_bytes(without_last_user)
    if without_last_user_bytes is not None:
        hashes["messages_sha256_without_last_user"] = sha256_hex(without_last_user_bytes)
        hashes["messages_last_user_index"] = last_user_index
    return hashes


def join_upstream_path(base_path: str, request_path: str) -> str:
    parsed = urlsplit(request_path)
    req_path = parsed.path or "/"
    if not req_path.startswith("/"):
        req_path = "/" + req_path
    prefix = base_path.rstrip("/")
    merged = (prefix + req_path) if prefix else req_path
    if parsed.query:
        return f"{merged}?{parsed.query}"
    return merged


class JsonlLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"llm-proxy-{stamp}.jsonl"
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class SseUsageTracker:
    def __init__(self, *, capture_response_body: bool, max_capture_bytes: int) -> None:
        self._buffer = ""
        self._usage: Dict[str, int] = {}
        self._text_parts: List[str] = []
        self._raw = bytearray()
        self._capture_response_body = capture_response_body
        self._max_capture_bytes = max_capture_bytes
        self.backend_model: Optional[str] = None
        self.event_count = 0
        self.event_samples: List[Dict[str, Any]] = []

    def feed(self, chunk: bytes) -> None:
        if self._capture_response_body and len(self._raw) < self._max_capture_bytes:
            keep = self._max_capture_bytes - len(self._raw)
            self._raw.extend(chunk[:keep])
        self._buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            self.event_count += 1
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            ptype = payload.get("type") if isinstance(payload, dict) else None
            if isinstance(payload, dict):
                if ptype == "message_start":
                    message = payload.get("message")
                    if isinstance(message, dict):
                        model_name = message.get("model")
                        if isinstance(model_name, str):
                            self.backend_model = model_name
                elif ptype == "content_block_start":
                    block = payload.get("content_block")
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            self._text_parts.append(text)
                elif ptype == "content_block_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, dict):
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            self._text_parts.append(text)
            if len(self.event_samples) < 20 and isinstance(payload, dict):
                sample: Dict[str, Any] = {"type": payload.get("type")}
                if ptype == "content_block_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, dict):
                        sample["delta_type"] = delta.get("type")
                        if isinstance(delta.get("text"), str):
                            sample["text_preview"] = delta["text"][:80]
                self.event_samples.append(sample)
            partial: Dict[str, int] = {}
            collect_usage_numbers(payload, partial)
            if partial:
                self._usage.update(partial)

    @property
    def usage(self) -> Dict[str, int]:
        return normalize_usage(self._usage)

    @property
    def assistant_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def captured_body(self) -> str:
        return self._raw.decode("utf-8", errors="replace")


@dataclass
class ProxyConfig:
    upstream_scheme: str
    upstream_host: str
    upstream_port: int
    upstream_base_path: str
    request_timeout: float
    max_preview_bytes: int
    max_capture_bytes: int
    log_message_payloads: bool
    capture_request_body: bool
    capture_response_body: bool
    logger: JsonlLogger


class LlmProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RipperdocLLMDebugProxy/0.1"

    @property
    def proxy_config(self) -> ProxyConfig:
        return cast(ProxyConfig, self.server.proxy_config)

    def do_GET(self) -> None:  # noqa: N802
        self._handle_proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._handle_proxy()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle_proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle_proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle_proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle_proxy()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_error_text(self, code: int, message: str) -> None:
        body = message.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _read_request_body(self) -> bytes:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return b""
        try:
            length = int(length_header)
        except ValueError:
            return b""
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _build_upstream_headers(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in self.headers.items():
            lk = key.lower()
            if lk in HOP_BY_HOP_HEADERS:
                continue
            if lk in {"host", "accept-encoding", "content-length"}:
                continue
            out[key] = value
        out["Host"] = self.proxy_config.upstream_host
        out["Connection"] = "close"
        return out

    def _write_stream_response(
        self,
        upstream: http.client.HTTPResponse,
    ) -> tuple[int, str, Dict[str, int], str, Dict[str, Any]]:
        response_headers = upstream.getheaders()
        content_type = upstream.getheader("Content-Type") or ""
        self.send_response(upstream.status, upstream.reason)
        for key, value in response_headers:
            lk = key.lower()
            if lk in HOP_BY_HOP_HEADERS or lk in {"content-length", "transfer-encoding"}:
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()

        tracker = SseUsageTracker(
            capture_response_body=self.proxy_config.capture_response_body,
            max_capture_bytes=self.proxy_config.max_capture_bytes,
        )
        sent_bytes = 0
        preview = bytearray()
        max_preview = self.proxy_config.max_preview_bytes
        while True:
            chunk = upstream.read(8192)
            if not chunk:
                break
            sent_bytes += len(chunk)
            tracker.feed(chunk)
            if len(preview) < max_preview:
                keep = max_preview - len(preview)
                preview.extend(chunk[:keep])
            self.wfile.write(chunk)
            self.wfile.flush()
        parsed_message_payload = {
            "backend_model": tracker.backend_model,
            "assistant_text": tracker.assistant_text,
            "event_count": tracker.event_count,
            "event_samples": tracker.event_samples,
        }
        stream_message_payload: Dict[str, Any] = {
            "message_payload": parsed_message_payload,
        }
        if self.proxy_config.capture_response_body:
            stream_message_payload["raw_sse"] = tracker.captured_body
            stream_message_payload["raw_sse_truncated"] = sent_bytes > self.proxy_config.max_capture_bytes
        return (
            sent_bytes,
            content_type,
            tracker.usage,
            preview.decode("utf-8", errors="replace"),
            stream_message_payload,
        )

    def _write_buffered_response(
        self,
        upstream: http.client.HTTPResponse,
    ) -> tuple[int, str, Dict[str, int], str, Dict[str, Any]]:
        body = upstream.read()
        response_headers = upstream.getheaders()
        content_type = upstream.getheader("Content-Type") or ""
        self.send_response(upstream.status, upstream.reason)
        for key, value in response_headers:
            lk = key.lower()
            if lk in HOP_BY_HOP_HEADERS or lk in {"content-length", "transfer-encoding"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)
            self.wfile.flush()

        usage: Dict[str, int] = {}
        payload = safe_json_loads(body)
        if payload is not None:
            collect_usage_numbers(payload, usage)
        response_message_payload = extract_response_message_payload(payload)
        response_body_capture = (
            capture_text_for_log(body, self.proxy_config.max_capture_bytes)
            if self.proxy_config.capture_response_body
            else None
        )
        details: Dict[str, Any] = {
            "message_payload": response_message_payload,
        }
        if response_body_capture is not None:
            details["body"] = response_body_capture
        preview = body[: self.proxy_config.max_preview_bytes].decode("utf-8", errors="replace")
        return len(body), content_type, normalize_usage(usage), preview, details

    def _handle_proxy(self) -> None:
        request_id = uuid.uuid4().hex
        started_at = time.perf_counter()
        body = self._read_request_body()
        request_json = safe_json_loads(body)
        request_summary = summarize_request_json(request_json, len(body))
        request_hashes = build_request_hashes(body, request_json)
        request_message_payload = extract_request_message_payload(request_json)
        request_headers = {k: v for k, v in self.headers.items()}

        upstream_path = join_upstream_path(self.proxy_config.upstream_base_path, self.path)
        upstream_headers = self._build_upstream_headers()
        conn_cls = (
            http.client.HTTPSConnection
            if self.proxy_config.upstream_scheme == "https"
            else http.client.HTTPConnection
        )

        status_code = 0
        response_body_preview = ""
        response_usage: Dict[str, int] = {}
        response_content_type = ""
        response_bytes = 0
        response_headers: Dict[str, str] = {}
        response_details: Dict[str, Any] = {}
        error_text: Optional[str] = None
        stream_mode = False

        try:
            conn = conn_cls(
                self.proxy_config.upstream_host,
                self.proxy_config.upstream_port,
                timeout=self.proxy_config.request_timeout,
            )
            conn.request(
                self.command,
                upstream_path,
                body=body if body else None,
                headers=upstream_headers,
            )
            upstream_resp = conn.getresponse()
            status_code = upstream_resp.status
            response_headers = {k: v for k, v in upstream_resp.getheaders()}
            response_content_type = upstream_resp.getheader("Content-Type") or ""
            transfer_encoding = (upstream_resp.getheader("Transfer-Encoding") or "").lower()
            stream_mode = (
                "text/event-stream" in response_content_type.lower()
                or "chunked" in transfer_encoding
            )
            if stream_mode:
                (
                    response_bytes,
                    response_content_type,
                    response_usage,
                    response_body_preview,
                    response_details,
                ) = self._write_stream_response(upstream_resp)
            else:
                (
                    response_bytes,
                    response_content_type,
                    response_usage,
                    response_body_preview,
                    response_details,
                ) = self._write_buffered_response(upstream_resp)
            conn.close()
        except Exception as exc:  # noqa: BLE001
            status_code = 502
            error_text = f"proxy_error: {type(exc).__name__}: {exc}"
            self._send_error_text(502, error_text)

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        record: Dict[str, Any] = {
            "ts": utc_now_iso(),
            "request_id": request_id,
            "method": self.command,
            "path": self.path,
            "upstream_path": upstream_path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "stream_mode": stream_mode,
            "request": {
                "headers": redact_headers(request_headers),
                "summary": request_summary,
                "hashes": request_hashes,
                "body_preview": body[: self.proxy_config.max_preview_bytes].decode(
                    "utf-8", errors="replace"
                ),
            },
            "response": {
                "headers": redact_headers(response_headers),
                "content_type": response_content_type,
                "bytes": response_bytes,
                "usage": response_usage,
                "body_preview": response_body_preview,
                "error": error_text,
            },
        }
        if self.proxy_config.capture_request_body:
            record["request"]["body"] = capture_text_for_log(body, self.proxy_config.max_capture_bytes)
        if self.proxy_config.log_message_payloads:
            record["request"]["message_payload"] = capture_json_for_log(
                request_message_payload,
                self.proxy_config.max_capture_bytes,
            )
            record["response"]["message_payload"] = capture_json_for_log(
                response_details.get("message_payload"),
                self.proxy_config.max_capture_bytes,
            )
        if self.proxy_config.capture_response_body and "body" in response_details:
            record["response"]["body"] = response_details["body"]
        if stream_mode and self.proxy_config.log_message_payloads:
            stream_payload = dict(response_details)
            stream_payload.pop("message_payload", None)
            record["response"]["stream_payload"] = capture_json_for_log(
                stream_payload,
                self.proxy_config.max_capture_bytes,
            )
        self.proxy_config.logger.write(record)
        print(
            f"[{record['ts']}] {self.command} {self.path} -> {status_code} "
            f"in {duration_ms}ms usage={response_usage}",
            flush=True,
        )


class ThreadingHttpServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _resolve_profile(config_data: Dict[str, Any], pointer: str, profile: Optional[str]) -> tuple[str, Dict[str, Any]]:
    profiles = config_data.get("model_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("config.json missing non-empty model_profiles")
    if profile:
        candidate = profiles.get(profile)
        if not isinstance(candidate, dict):
            raise ValueError(f"profile '{profile}' not found in model_profiles")
        return profile, candidate
    pointers = config_data.get("model_pointers")
    if not isinstance(pointers, dict):
        raise ValueError("config.json missing model_pointers")
    profile_name = pointers.get(pointer)
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError(f"model_pointers.{pointer!s} not found")
    candidate = profiles.get(profile_name)
    if not isinstance(candidate, dict):
        raise ValueError(f"profile '{profile_name}' (from pointer {pointer}) not found")
    return profile_name, candidate


def resolve_upstream_from_config(
    config_path: Path,
    pointer: str,
    profile: Optional[str],
) -> tuple[str, str, Dict[str, Any]]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.json root is not an object")
    profile_name, profile_data = _resolve_profile(data, pointer, profile)
    api_base = profile_data.get("api_base")
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError(
            f"profile '{profile_name}' has empty api_base; pass --upstream-base explicitly"
        )
    return api_base.strip(), profile_name, cast(Dict[str, Any], profile_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM debug reverse proxy for Ripperdoc traffic",
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18080)
    parser.add_argument("--upstream-base", default=None)
    parser.add_argument(
        "--config-path",
        default=str(Path("~/.ripperdoc/config.json").expanduser()),
    )
    parser.add_argument("--pointer", default="main", help="model pointer in config.json")
    parser.add_argument("--profile", default=None, help="profile name (overrides --pointer)")
    parser.add_argument(
        "--log-dir",
        default=str(Path("~/.ripperdoc/proxy_logs").expanduser()),
    )
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-preview-bytes", type=int, default=12000)
    parser.add_argument("--max-capture-bytes", type=int, default=300000)
    parser.add_argument(
        "--no-log-message-payloads",
        action="store_true",
        help="Disable logging parsed request/response message payloads",
    )
    parser.add_argument(
        "--no-capture-request-body",
        action="store_true",
        help="Disable logging raw request body",
    )
    parser.add_argument(
        "--no-capture-response-body",
        action="store_true",
        help="Disable logging raw response body",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config_path).expanduser()

    if args.upstream_base:
        upstream_base = args.upstream_base
        profile_name = args.profile or "<manual>"
        profile_data: Dict[str, Any] = {}
    else:
        upstream_base, profile_name, profile_data = resolve_upstream_from_config(
            config_path=config_path,
            pointer=args.pointer,
            profile=args.profile,
        )

    parsed = urlsplit(upstream_base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"invalid upstream base URL: {upstream_base}")
    upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)

    logger = JsonlLogger(Path(args.log_dir).expanduser())
    proxy_config = ProxyConfig(
        upstream_scheme=parsed.scheme,
        upstream_host=parsed.hostname,
        upstream_port=upstream_port,
        upstream_base_path=parsed.path or "",
        request_timeout=max(args.request_timeout, 1.0),
        max_preview_bytes=max(args.max_preview_bytes, 1024),
        max_capture_bytes=max(args.max_capture_bytes, 1024),
        log_message_payloads=not args.no_log_message_payloads,
        capture_request_body=not args.no_capture_request_body,
        capture_response_body=not args.no_capture_response_body,
        logger=logger,
    )

    server = ThreadingHttpServer((args.listen_host, args.listen_port), LlmProxyHandler)
    server.proxy_config = proxy_config  # type: ignore[attr-defined]

    protocol = profile_data.get("protocol", "<unknown>")
    model = profile_data.get("model", "<unknown>")
    print(
        f"Proxy listening on http://{args.listen_host}:{args.listen_port} -> {upstream_base}",
        flush=True,
    )
    print(
        f"Resolved profile={profile_name!r} protocol={protocol!r} model={model!r}",
        flush=True,
    )
    print(f"Log file: {logger.path}", flush=True)
    print(
        "Run Ripperdoc with RIPPERDOC_BASE_URL pointing to proxy, for example:\n"
        f"  RIPPERDOC_BASE_URL=http://{args.listen_host}:{args.listen_port} ripperdoc",
        flush=True,
    )

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping proxy...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
