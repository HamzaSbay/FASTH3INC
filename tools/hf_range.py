"""HTTP Range reader with retry, backoff, and resumable streaming support.

Provides byte-range fetching against remote HTTP/HTTPS endpoints (e.g. Hugging
Face) as well as local file fallback for testing and offline runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse
import requests

logger = logging.getLogger(__name__)

# Status codes considered transient and safe to retry with backoff
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

DEFAULT_CACHE_DIR = os.environ.get("FH3_CACHE_DIR")


def read(
    url: str,
    start: int,
    end: int,
    max_retries: int = 5,
    backoff_factor: float = 0.5,
    timeout: float = 30.0,
    chunk_size: int = 64 * 1024,
    headers: Optional[dict[str, str]] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> bytes:
    """Read a byte slice [start, end) from a remote URL or local file.

    Args:
        url: URL (http/https/file) or local filesystem path.
        start: Starting byte index (inclusive, 0-based).
        end: Ending byte index (exclusive, 0-based).
        max_retries: Maximum number of retry attempts on network error or 5xx/429.
        backoff_factor: Initial backoff multiplier in seconds (doubles each attempt).
        timeout: Socket connect / read timeout in seconds.
        chunk_size: Size in bytes for streaming chunks.
        headers: Optional extra HTTP headers.

    Returns:
        Exact byte slice requested or available up to file EOF.

    Raises:
        ValueError: If start < 0 or start > end.
        requests.HTTPError: If an unrecoverable HTTP status code is received.
        requests.RequestException: If network failures persist after retries.
    """
    if start < 0:
        raise ValueError(f"start ({start}) must be non-negative")
    if start > end:
        raise ValueError(f"start ({start}) cannot exceed end ({end})")
    if start == end:
        return b""

    expected_len = end - start

    # Handle local filesystem paths or file:// URLs
    parsed = urlparse(url)
    is_remote = parsed.scheme in ("http", "https")
    if not is_remote:
        file_path = url
        if parsed.scheme == "file":
            file_path = parsed.path
            if os.name == "nt" and file_path.startswith("/") and len(file_path) > 2 and file_path[2] == ":":
                file_path = file_path.lstrip("/")
        elif not os.path.exists(file_path) and os.path.exists(parsed.path):
            file_path = parsed.path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file does not exist: {file_path}")

        file_size = os.path.getsize(file_path)
        if start >= file_size:
            return b""
        to_read = min(expected_len, file_size - start)
        with open(file_path, "rb") as f:
            f.seek(start)
            return f.read(to_read)

    # Check disk cache for remote slice if enabled
    cache_path = None
    effective_cache_dir = cache_dir or DEFAULT_CACHE_DIR
    if use_cache and effective_cache_dir:
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        c_dir = Path(effective_cache_dir)
        cache_path = c_dir / f"{url_hash}_{start}_{end}.bin"
        if cache_path.exists() and cache_path.stat().st_size == expected_len:
            try:
                with open(cache_path, "rb") as cf:
                    return cf.read()
            except OSError:
                pass

    # Remote HTTP / HTTPS fetching
    req_headers = dict(headers or {})
    # Propagate Hugging Face auth token if available in environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and "Authorization" not in req_headers:
        req_headers["Authorization"] = f"Bearer {hf_token}"

    buf = bytearray()
    cur_start = start
    attempt = 0
    last_err: Optional[Exception] = None

    while cur_start < end:
        range_header = f"bytes={cur_start}-{end - 1}"
        req_headers["Range"] = range_header

        try:
            with requests.get(
                url,
                headers=req_headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            ) as resp:
                if resp.status_code in RETRYABLE_STATUS_CODES:
                    attempt += 1
                    if attempt > max_retries:
                        resp.raise_for_status()
                    sleep_time = backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "HTTP %d on %s (range %s), retrying in %.2fs (attempt %d/%d)",
                        resp.status_code,
                        url,
                        range_header,
                        sleep_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(sleep_time)
                    continue

                # 206 Partial Content (expected) or 200 OK (server ignored Range on first fetch)
                if resp.status_code == 200:
                    # If whole file returned on initial request, slice the relevant window
                    if cur_start == start:
                        full_content = resp.content
                        if len(full_content) < end:
                            return full_content[start:]
                        return full_content[start:end]
                    else:
                        raise requests.HTTPError(
                            f"Server returned 200 OK during mid-stream resume: {url}",
                            response=resp,
                        )

                if resp.status_code == 416:
                    # Range not satisfiable (reached EOF)
                    break

                if resp.status_code != 206:
                    resp.raise_for_status()

                # Stream chunks into buffer
                bytes_received_this_req = 0
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    # Clamp chunk to remaining bytes to avoid any overfill
                    remaining = end - cur_start
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                    buf.extend(chunk)
                    cur_start += len(chunk)
                    bytes_received_this_req += len(chunk)
                    if cur_start >= end:
                        break

                if bytes_received_this_req == 0:
                    # Server returned 206 with empty body (EOF)
                    break

        except (requests.RequestException, ConnectionError, TimeoutError, IOError) as exc:
            last_err = exc
            attempt += 1
            if attempt > max_retries:
                logger.error("Max retries exceeded fetching %s range %s: %s", url, range_header, exc)
                raise
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            logger.warning(
                "Network error on %s (range %s): %s; retrying in %.2fs (attempt %d/%d, resumed at byte %d)",
                url,
                range_header,
                exc,
                sleep_time,
                attempt,
                max_retries,
                cur_start,
            )
            time.sleep(sleep_time)

    res = bytes(buf)
    if cache_path and len(res) == expected_len:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_cache = cache_path.with_suffix(".tmp")
            with open(temp_cache, "wb") as cf:
                cf.write(res)
            temp_cache.replace(cache_path)
        except OSError:
            pass
    return res
