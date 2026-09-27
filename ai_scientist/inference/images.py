"""Validate real image bytes and preserve every submitted image occurrence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import warnings
from pathlib import Path

from .contracts import json_copy
from .errors import ConfigurationError, ContextLimitExceeded, UnsupportedCapability

_FORMATS = {"PNG": ("image/png", ".png"), "JPEG": ("image/jpeg", ".jpg"), "WEBP": ("image/webp", ".webp"), "GIF": ("image/gif", ".gif")}
DEFAULT_MAX_BYTES = 32 * 1024 * 1024


def _read_image(part: dict, max_bytes: int) -> tuple[bytes, str, str]:
    raw = None
    source = part.get("source", {})
    url = part.get("image_url", {})
    if isinstance(url, dict):
        url = url.get("url", "")
    path = part.get("path")
    try:
        if isinstance(source, dict) and source.get("type") == "base64":
            encoded = source.get("data")
            if not isinstance(encoded, str):
                raise ConfigurationError("Image base64 data must be text")
            if len(encoded) > max_bytes * 4 // 3 + 8:
                raise ContextLimitExceeded("Image exceeds the configured byte budget")
            raw = base64.b64decode(encoded, validate=True)
        elif isinstance(url, str) and url.startswith("data:image/"):
            header, separator, encoded = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ConfigurationError("Only base64 image data URIs are supported")
            if len(encoded) > max_bytes * 4 // 3 + 8:
                raise ContextLimitExceeded("Image exceeds the configured byte budget")
            raw = base64.b64decode(encoded, validate=True)
        else:
            if path is None and isinstance(url, str) and url:
                path = url
            if path is None and isinstance(source, dict) and source.get("type") == "url":
                path = source.get("url")
            if not isinstance(path, (str, Path)) or not str(path):
                raise UnsupportedCapability("Unsupported image source")
            if str(path).startswith(("http://", "https://", "file://")):
                raise UnsupportedCapability("Remote image URLs must be explicitly downloaded before inference")
            source_path = Path(path).expanduser().resolve(strict=True)
            if not source_path.is_file() or source_path.stat().st_size > max_bytes:
                raise ContextLimitExceeded("Image is not a regular file within the byte budget")
            with source_path.open("rb") as stream:
                raw = stream.read(max_bytes + 1)
    except (binascii.Error, ValueError, OSError) as exc:
        raise ConfigurationError("Cannot read or decode image input") from exc
    if not raw or len(raw) > max_bytes:
        raise ContextLimitExceeded("Image is empty or exceeds the configured byte budget")
    try:
        from PIL import Image
    except ImportError as exc:
        raise UnsupportedCapability("Image inference requires Pillow") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as img:
                fmt = img.format
                if getattr(img, "n_frames", 1) != 1:
                    raise UnsupportedCapability("Animated images require an explicit frame-selection policy")
                img.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ContextLimitExceeded("Image pixel count exceeds Pillow's safe decoding limit") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ConfigurationError("Image bytes are invalid") from exc
    if fmt not in _FORMATS:
        raise UnsupportedCapability("Unsupported image format; use PNG, JPEG, WebP, or static GIF")
    mime, extension = _FORMATS[fmt]
    return raw, mime, extension


def _parts(message: dict) -> list[dict]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or not all(isinstance(p, dict) for p in content):
        raise ConfigurationError("Invalid multimodal message content")
    return content


def freeze_messages(messages: list[dict], max_bytes: int = DEFAULT_MAX_BYTES) -> list[dict]:
    """Snapshot file bytes before queueing; later caller edits cannot change input."""
    frozen = []
    used = 0
    for message in messages:
        parts = []
        for part in _parts(message):
            kind = part.get("type")
            if kind in ("text", "input_text"):
                if not isinstance(part.get("text"), str):
                    raise ConfigurationError("Text content must be a string")
                used += len(part["text"].encode("utf-8"))
                parts.append({"type": "text", "text": part["text"]})
            elif kind in ("image", "image_url", "input_image"):
                if kind == "input_image":
                    part = {**part, "image_url": part.get("image_url")}
                raw, mime, _ = _read_image(part, max_bytes)
                used += len(raw)
                parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(raw).decode("ascii")}})
            else:
                raise UnsupportedCapability("Unsupported conversation content part")
            if used > max_bytes:
                raise ContextLimitExceeded("Conversation exceeds the configured input byte budget")
        frozen.append({"role": message["role"], "content": parts})
    return json_copy(frozen)


def materialize_messages(messages: list[dict], request_dir: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> list[dict]:
    directory = Path(request_dir) / "images"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    normalized = []
    index = 0
    used = 0
    for message in messages:
        parts = []
        for part in _parts(message):
            if part.get("type") in ("text", "input_text"):
                if not isinstance(part.get("text"), str):
                    raise ConfigurationError("Text content must be a string")
                used += len(part["text"].encode("utf-8"))
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") in ("image", "image_url", "input_image"):
                raw, mime, extension = _read_image(part, max_bytes)
                used += len(raw)
                digest = hashlib.sha256(raw).hexdigest()
                target = directory / (digest + extension)
                if not target.exists():
                    with target.open("xb") as stream:
                        stream.write(raw)
                    target.chmod(0o600)
                parts.append({"type": "image", "path": str(target.resolve()), "mime_type": mime,
                              "occurrence_id": f"image-{index}", "sha256": digest, "byte_size": len(raw)})
                index += 1
            else:
                raise UnsupportedCapability("Unsupported conversation content part")
            if used > max_bytes:
                raise ContextLimitExceeded("Conversation exceeds the configured input byte budget")
        normalized.append({"role": message["role"], "content": parts})
    return normalized


def iter_images(messages: list[dict]) -> list[dict]:
    return [part for message in messages for part in _parts(message) if part.get("type") == "image"]
