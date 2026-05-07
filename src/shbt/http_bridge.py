from __future__ import annotations

"""Minimal HTTP helpers for staging remote JSON and CSV payloads.

The project's observational tooling already knows how to ingest local JSON and
CSV trigger drops. This module provides a tiny stdlib-only bridge for fetching
remote payloads and mirroring them into the same local file-based workflow.
"""

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_ACCEPT_HEADER = "application/json,text/csv,text/plain;q=0.9,*/*;q=0.1"
DEFAULT_USER_AGENT = "so10-312-reviewer-framework/http-bridge"
_SUPPORTED_TRIGGER_SUFFIXES = frozenset({".csv", ".json"})
_CONTENT_TYPE_SUFFIXES = {
    "application/csv": ".csv",
    "application/json": ".json",
    "application/vnd.ms-excel": ".csv",
    "text/csv": ".csv",
    "text/json": ".json",
    "text/x-csv": ".csv",
}


@dataclass(frozen=True)
class HTTPBridgeResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def content_type(self) -> str | None:
        raw_value = self.headers.get("Content-Type") or self.headers.get("content-type")
        if raw_value is None:
            return None
        return raw_value.split(";", 1)[0].strip().lower()

    def text(self, *, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding)


class HTTPBridgeError(RuntimeError):
    """Raised when remote payload staging fails."""


FetchResult = HTTPBridgeResponse | Mapping[str, Any] | tuple[Any, ...] | bytes | str
FetchHook = Callable[..., FetchResult]


def _coerce_bytes(payload: bytes | str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return payload.encode("utf-8")


def _normalized_headers(headers: Mapping[str, object] | None) -> dict[str, str]:
    return {str(key): str(value) for key, value in (headers or {}).items()}


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must use http or https.")
    if not parsed.netloc:
        raise ValueError("url must include a host.")


def _call_fetcher(fetcher: FetchHook, url: str, headers: Mapping[str, str], timeout: float) -> FetchResult:
    try:
        parameters = tuple(signature(fetcher).parameters.values())
    except (TypeError, ValueError):
        return fetcher(url, headers, timeout)

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if any(parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters) or len(positional_parameters) >= 3:
        return fetcher(url, headers, timeout)
    if len(positional_parameters) == 2:
        return fetcher(url, headers)
    return fetcher(url)


def _coerce_response(url: str, result: FetchResult) -> HTTPBridgeResponse:
    if isinstance(result, HTTPBridgeResponse):
        return result
    if isinstance(result, Mapping):
        body = result.get("body")
        if not isinstance(body, (bytes, str)):
            raise TypeError("fetcher mapping results must include a string or bytes 'body'.")
        return HTTPBridgeResponse(
            url=url,
            status=int(result.get("status", 200)),
            headers=_normalized_headers(result.get("headers") if isinstance(result.get("headers"), Mapping) else None),
            body=_coerce_bytes(body),
        )
    if isinstance(result, tuple):
        if len(result) == 2:
            status, body = result
            headers: Mapping[str, object] | None = None
        elif len(result) == 3:
            status, headers, body = result
            headers = headers if isinstance(headers, Mapping) else None
        else:
            raise TypeError("fetcher tuple results must have two or three items.")
        if not isinstance(body, (bytes, str)):
            raise TypeError("fetcher tuple results must end with a string or bytes body.")
        return HTTPBridgeResponse(
            url=url,
            status=int(status),
            headers=_normalized_headers(headers),
            body=_coerce_bytes(body),
        )
    if isinstance(result, (bytes, str)):
        return HTTPBridgeResponse(url=url, status=200, headers={}, body=_coerce_bytes(result))
    raise TypeError("fetcher must return bytes, str, a tuple, a mapping, or HTTPBridgeResponse.")


def _default_fetcher(url: str, headers: Mapping[str, str], timeout: float) -> HTTPBridgeResponse:
    request = Request(url, headers=dict(headers))
    with urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", response.getcode()))
        return HTTPBridgeResponse(
            url=url,
            status=status,
            headers={key: value for key, value in response.headers.items()},
            body=response.read(),
        )


def _remote_name_from_url(url: str) -> str:
    name = Path(urlsplit(url).path).name
    return name or "download"


def _safe_filename(filename: str) -> str:
    candidate = Path(filename).name
    if candidate != filename or candidate in {"", ".", ".."}:
        raise ValueError("filename must be a simple file name.")
    return candidate


def infer_suffix(url: str, *, content_type: str | None = None) -> str | None:
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_content_type in _CONTENT_TYPE_SUFFIXES:
        return _CONTENT_TYPE_SUFFIXES[normalized_content_type]
    remote_name = _remote_name_from_url(url)
    suffix = Path(remote_name).suffix.lower()
    if suffix:
        return suffix
    return None


def _resolve_target_path(
    url: str,
    destination: Path | str,
    *,
    response: HTTPBridgeResponse,
    filename: str | None,
) -> Path:
    destination_path = Path(destination)
    if destination_path.suffix and filename is None:
        return destination_path

    suffix = infer_suffix(url, content_type=response.content_type)
    remote_name = _remote_name_from_url(url)
    if filename is None:
        stem = Path(remote_name).stem or remote_name or "download"
        resolved_filename = _safe_filename(f"{stem}{suffix or ''}")
    else:
        resolved_filename = _safe_filename(filename)
        if Path(resolved_filename).suffix == "" and suffix is not None:
            resolved_filename = _safe_filename(f"{resolved_filename}{suffix}")
    return destination_path / resolved_filename


def _write_response(target_path: Path, response: HTTPBridgeResponse, *, overwrite: bool) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file {target_path}.")
    target_path.write_bytes(response.body)
    return target_path


def fetch_resource(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: FetchHook | None = None,
) -> HTTPBridgeResponse:
    _validate_url(url)
    resolved_timeout = float(timeout)
    if resolved_timeout <= 0:
        raise ValueError("timeout must be positive.")

    request_headers = {
        "Accept": DEFAULT_ACCEPT_HEADER,
        "User-Agent": DEFAULT_USER_AGENT,
    }
    request_headers.update(_normalized_headers(headers))

    resolved_fetcher = _default_fetcher if fetcher is None else fetcher
    try:
        response = _coerce_response(url, _call_fetcher(resolved_fetcher, url, request_headers, resolved_timeout))
    except (HTTPError, URLError, OSError, TypeError, ValueError) as exc:
        raise HTTPBridgeError(f"Failed to fetch remote payload from {url!r}.") from exc

    if not 200 <= response.status < 300:
        raise HTTPBridgeError(f"Remote payload request for {url!r} returned status {response.status}.")
    return response


def fetch_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: FetchHook | None = None,
) -> Any:
    response = fetch_resource(url, headers=headers, timeout=timeout, fetcher=fetcher)
    try:
        return json.loads(response.text())
    except json.JSONDecodeError as exc:
        raise HTTPBridgeError(f"Remote payload from {url!r} is not valid JSON.") from exc


def fetch_csv_rows(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: FetchHook | None = None,
) -> tuple[dict[str, str], ...]:
    response = fetch_resource(url, headers=headers, timeout=timeout, fetcher=fetcher)
    reader = csv.DictReader(StringIO(response.text()))
    return tuple({str(key): value for key, value in row.items()} for row in reader)


def stage_resource(
    url: str,
    destination: Path | str,
    *,
    filename: str | None = None,
    overwrite: bool = False,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: FetchHook | None = None,
) -> Path:
    response = fetch_resource(url, headers=headers, timeout=timeout, fetcher=fetcher)
    target_path = _resolve_target_path(url, destination, response=response, filename=filename)
    return _write_response(target_path, response, overwrite=overwrite)


def stage_trigger_payload(
    url: str,
    trigger_directory: Path | str,
    *,
    filename: str | None = None,
    overwrite: bool = False,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: FetchHook | None = None,
) -> Path:
    response = fetch_resource(url, headers=headers, timeout=timeout, fetcher=fetcher)
    normalized_filename = _safe_filename(filename) if filename is not None else None
    resolved_suffix = infer_suffix(url, content_type=response.content_type)
    candidate_suffix = Path(normalized_filename).suffix.lower() if normalized_filename is not None else (resolved_suffix or "")
    if candidate_suffix not in _SUPPORTED_TRIGGER_SUFFIXES:
        raise HTTPBridgeError(
            "Remote trigger payloads must resolve to .json or .csv content."
        )
    target_path = _resolve_target_path(url, trigger_directory, response=response, filename=normalized_filename)
    return _write_response(target_path, response, overwrite=overwrite)


__all__ = [
    "DEFAULT_ACCEPT_HEADER",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "FetchHook",
    "HTTPBridgeError",
    "HTTPBridgeResponse",
    "fetch_csv_rows",
    "fetch_json",
    "fetch_resource",
    "infer_suffix",
    "stage_resource",
    "stage_trigger_payload",
]
