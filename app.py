import os
import time
import json
import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

# =========================
# LOGGING SETUP (configurable)
# =========================
# Environment-controlled logging configuration. Use LOG_LEVEL and LOG_TO_FILE
# to control verbosity and whether logs are written to a file.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "0") == "1"
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "/tmp/foundry-adapter.log")

_log_level_const = getattr(logging, LOG_LEVEL, logging.INFO)
_handlers = [logging.StreamHandler(sys.stdout)]
if LOG_TO_FILE:
    # Only add file logging when explicitly enabled
    _handlers.append(logging.FileHandler(LOG_FILE_PATH))

logging.basicConfig(
    level=_log_level_const,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=_handlers,
)
logger = logging.getLogger("foundry-adapter")
logger.setLevel(_log_level_const)

# =========================
# ENV CONFIG
# =========================
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

# "azure_openai_deployments" expected for Cognitive Services endpoint
BACKEND_MODE = os.environ.get("BACKEND_MODE", "azure_openai_deployments").strip()

ADAPTER_BEARER_KEY = os.environ.get("ADAPTER_BEARER_KEY", "")
ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
MODEL_MAP_JSON = os.environ.get("MODEL_MAP_JSON", "{}").strip()

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "300"))
CONNECT_TIMEOUT_SECONDS = int(os.environ.get("CONNECT_TIMEOUT_SECONDS", "30"))
SSE_KEEPALIVE_SECONDS = int(os.environ.get("SSE_KEEPALIVE_SECONDS", "15"))

# IMPORTANT for GPT-5 on Azure: use max_completion_tokens (not max_tokens)
DEFAULT_MAX_COMPLETION_TOKENS = int(os.environ.get("DEFAULT_MAX_COMPLETION_TOKENS", "1024"))

ADAPTER_DEBUG = os.environ.get("ADAPTER_DEBUG", "0") == "1"


def _d(*args):
    if ADAPTER_DEBUG:
        logger.debug(" ".join(str(a) for a in args))


app = FastAPI()


# =========================
# UTIL
# =========================
def _require_config():
    if not AZURE_OPENAI_ENDPOINT:
        logger.error("Missing AZURE_OPENAI_ENDPOINT")
        raise HTTPException(500, "Missing AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        logger.error("Missing AZURE_OPENAI_API_KEY")
        raise HTTPException(500, "Missing AZURE_OPENAI_API_KEY")
    logger.debug("Configuration validated")


def _check_auth(authorization: Optional[str]):
    if not ADAPTER_BEARER_KEY:
        logger.debug("Auth bypass: no ADAPTER_BEARER_KEY configured")
        return
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Auth failed: missing or malformed Bearer token")
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != ADAPTER_BEARER_KEY:
        logger.warning("Auth failed: invalid Bearer token")
        raise HTTPException(401, "Invalid Bearer token")
    logger.debug("Auth successful")


def _model_allowed(model: str) -> bool:
    return (not ALLOWED_MODELS) or (model in ALLOWED_MODELS)


def _load_model_map() -> Dict[str, str]:
    try:
        d = json.loads(MODEL_MAP_JSON) if MODEL_MAP_JSON else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


MODEL_MAP = _load_model_map()


# Log startup configuration
logger.info("=" * 80)
logger.info("Foundry Adapter Starting")
logger.info("=" * 80)
logger.info(f"Backend Mode: {BACKEND_MODE}")
logger.info(f"Azure Endpoint: {AZURE_OPENAI_ENDPOINT}")
logger.info(f"Azure API Version: {AZURE_OPENAI_API_VERSION}")
logger.info(f"Auth Enabled: {bool(ADAPTER_BEARER_KEY)}")
logger.info(f"Allowed Models: {ALLOWED_MODELS if ALLOWED_MODELS else 'all'}")
logger.info(f"Model Map: {MODEL_MAP}")
logger.info(f"Request Timeout: {REQUEST_TIMEOUT_SECONDS}s")
logger.info(f"Connect Timeout: {CONNECT_TIMEOUT_SECONDS}s")
logger.info(f"SSE Keepalive: {SSE_KEEPALIVE_SECONDS}s")
logger.info(f"Debug Mode: {ADAPTER_DEBUG}")
logger.info("=" * 80)


def _resolve_deployment(incoming_model: str) -> str:
    m = (incoming_model or "").strip()
    if not m:
        logger.error("Model resolution failed: empty model string")
        raise HTTPException(400, "Missing model")
    resolved = MODEL_MAP.get(m, m)
    if resolved != m:
        logger.info(f"Model mapped: {m} -> {resolved}")
    else:
        logger.debug(f"Model resolved (no mapping): {m}")
    return resolved


def _normalize_content(c: Any) -> Any:
    """
    If client sends OpenAI multipart content:
      [{"type":"text","text":"..."}]
    convert to string.
    Ensures content is always a non-null string.
    """
    if c is None:
        logger.warning("Content is None, defaulting to empty string")
        return ""
    
    if isinstance(c, list):
        parts: List[str] = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined
        logger.warning(f"Content list had no text parts or resulted in empty string, original: {c}")
        return ""  # Return empty string instead of original list
    
    # Ensure string conversion
    if isinstance(c, str):
        return c if c else ""
    
    # For other types, convert to string
    return str(c) if c is not None else ""


# =========================
# Pydantic models
# =========================
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False


# =========================
# Backend helpers
# =========================
def _backend_url(deployment: str) -> str:
    if BACKEND_MODE != "azure_openai_deployments":
        raise HTTPException(500, f"Unsupported BACKEND_MODE: {BACKEND_MODE}")
    return (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )


def _backend_headers() -> Dict[str, str]:
    return {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }


def _build_payload(req: ChatCompletionsRequest) -> Dict[str, Any]:
    messages = []
    tool_calls_indices = set()
    
    for i, m in enumerate(req.messages):
        normalized_content = _normalize_content(m.content)
        if normalized_content is None:
            logger.error(f"Message {i}: content normalized to None, using empty string")
            normalized_content = ""
        
        msg_dict = {"role": m.role, "content": normalized_content}
        
        # Track if message has tool_calls for validation
        if hasattr(m, 'tool_calls') and m.tool_calls:
            tool_calls_indices.add(i)
            msg_dict['tool_calls'] = m.tool_calls
            _d(f"Message {i}: role={m.role}, has_tool_calls=True, content_len={len(str(normalized_content))}")
        else:
            _d(f"Message {i}: role={m.role}, has_tool_calls=False, content_len={len(str(normalized_content))}")
        
        # For tool messages, include tool_call_id if present
        if m.role == "tool":
            if hasattr(m, 'tool_call_id') and m.tool_call_id:
                msg_dict['tool_call_id'] = m.tool_call_id
                _d(f"Message {i}: tool message with tool_call_id={m.tool_call_id}")
            else:
                logger.warning(f"Message {i}: role='tool' but missing tool_call_id")
            
            # Validate that tool messages follow tool_calls
            if i > 0 and i - 1 not in tool_calls_indices:
                logger.warning(f"Message {i}: role='tool' but no tool_calls found in preceding messages. "
                              f"tool_calls found at indices: {sorted(tool_calls_indices)}")
        
        messages.append(msg_dict)
    
    payload: Dict[str, Any] = {
        "messages": messages,
        "stream": bool(req.stream),
    }

    if req.temperature is not None:
        payload["temperature"] = req.temperature

    # Use ONLY max_completion_tokens for Azure GPT-5 models
    mct = req.max_completion_tokens
    if mct is None:
        mct = req.max_tokens
    if mct is None:
        mct = DEFAULT_MAX_COMPLETION_TOKENS
    payload["max_completion_tokens"] = int(mct)

    # forward extra fields, but never re-inject max_tokens / max_completion_tokens
    extras = req.model_dump(
        exclude={"model", "messages", "temperature", "max_tokens", "max_completion_tokens", "stream"}
    )
    for k, v in extras.items():
        if v is not None:
            payload[k] = v

    return payload


def _sse_headers() -> Dict[str, str]:
    return {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


# =========================
# API
# =========================
@app.get("/healthz")
def healthz():
    logger.debug("Healthz check")
    return {"ok": True, "mode": BACKEND_MODE}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    models = ALLOWED_MODELS or list(MODEL_MAP.keys()) or ["gpt-5-mini", "gpt-5-nano"]
    logger.debug(f"Listing {len(models)} available models")
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    _require_config()

    if not _model_allowed(req.model):
        logger.warning(f"Request rejected: model not allowed: {req.model}")
        raise HTTPException(400, f"Model not allowed: {req.model}")

    deployment = _resolve_deployment(req.model)
    url = _backend_url(deployment)
    payload = _build_payload(req)
    headers = _backend_headers()

    logger.info(
        f"Chat completion request - model={req.model}, deployment={deployment}, "
        f"stream={req.stream}, max_tokens={req.max_tokens}, "
        f"max_completion_tokens={req.max_completion_tokens}, temp={req.temperature}, "
        f"messages_count={len(req.messages)}"
    )
    
    # Log message structure for debugging
    if ADAPTER_DEBUG:
        for i, msg in enumerate(payload["messages"]):
            tool_info = f", tool_call_id='{msg.get('tool_call_id')}'" if msg.get('tool_call_id') else ""
            logger.debug(f"  Msg[{i}]: role='{msg.get('role')}', content_len={len(str(msg.get('content', '')))}, "
                        f"has_tool_calls={'tool_calls' in msg}{tool_info}")
    
    _d(f"Request payload: {json.dumps(payload, default=str)}")

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)

    # -------------------------
    # Non-stream
    # -------------------------
    if not req.stream:
        logger.debug(f"Non-streaming request to {url}")
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(url, headers=headers, json=payload)
                logger.info(f"Backend response status: {r.status_code}")
                _d(f"Response headers: {dict(r.headers)}")
            except Exception as e:
                logger.error(f"Backend request failed: {e}", exc_info=True)
                raise HTTPException(500, f"Backend request failed: {str(e)}")

        if r.status_code >= 400:
            logger.error(f"Backend error {r.status_code}: {r.text}")
            raise HTTPException(r.status_code, r.text)

        try:
            result = r.json()
            logger.debug("Successfully parsed JSON response")
            return JSONResponse(result)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}, returning raw text")
            return JSONResponse({"raw": r.text})

    # -------------------------
    # Stream
    # -------------------------
    async def event_generator() -> AsyncGenerator[bytes, None]:
        logger.info(f"Streaming request started - deployment={deployment}")
        _d("event_generator start")

        # Starter chunk to make UI attach
        created = int(time.time())
        chunk_id = f"chatcmpl-adapter-{created}"
        starter = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": " "}, "finish_reason": None}],
        }
        yield b"data: " + json.dumps(starter, ensure_ascii=False).encode("utf-8") + b"\n\n"
        logger.debug("Sent starter chunk")

        buf = b""
        last_emit = time.time()
        chunks_sent = 0

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug(f"Opening stream to {url}")
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    _d(f"Azure response status: {resp.status_code}, content-type: {resp.headers.get('content-type')}")
                    logger.info(f"Stream opened - status={resp.status_code}, content-type={resp.headers.get('content-type')}")

                    if resp.status_code >= 400:
                        body = await resp.aread()
                        msg = body.decode("utf-8", errors="replace")
                        logger.error(f"Backend stream error {resp.status_code}: {msg}")
                        yield b"data: " + json.dumps({"error": msg}, ensure_ascii=False).encode("utf-8") + b"\n\n"
                        yield b"data: [DONE]\n\n"
                        return

                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        buf += chunk

                        now = time.time()
                        if now - last_emit >= SSE_KEEPALIVE_SECONDS:
                            yield b": ping\n\n"
                            _d("Sent keepalive ping")
                            last_emit = now

                        while b"\n\n" in buf:
                            raw, buf = buf.split(b"\n\n", 1)
                            raw = raw.strip(b"\r\n")
                            if not raw:
                                continue

                            if not raw.startswith(b"data:"):
                                _d(f"Skipping non-data line: {raw[:50]}")
                                continue

                            data = raw[len(b"data:") :].strip()

                            if data == b"[DONE]":
                                logger.info(f"Stream completed - total chunks sent: {chunks_sent}")
                                yield b"data: [DONE]\n\n"
                                return

                            try:
                                obj = json.loads(data.decode("utf-8", errors="replace"))
                            except Exception as e:
                                logger.debug(f"Failed to parse chunk JSON: {e}")
                                continue

                            # Filter out the "prompt_filter_results" prelude / non-chunk events
                            if isinstance(obj, dict) and obj.get("object") == "chat.completion.chunk":
                                yield b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"
                                chunks_sent += 1
                                last_emit = time.time()
                                _d(f"Sent chunk {chunks_sent}")

                    # If backend ended without DONE
                    logger.info(f"Stream ended without [DONE] signal - total chunks sent: {chunks_sent}")
                    yield b"data: [DONE]\n\n"
                    return

        except asyncio.CancelledError:
            logger.info(f"Client cancelled stream after {chunks_sent} chunks")
            _d("client cancelled")
            return
        except Exception as e:
            logger.error(f"Stream exception after {chunks_sent} chunks: {e}", exc_info=True)
            _d(f"stream exception: {repr(e)}")
            yield b"data: " + json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8") + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

    return StreamingResponse(event_generator(), headers=_sse_headers())
