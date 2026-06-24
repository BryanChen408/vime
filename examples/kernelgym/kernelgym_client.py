"""Async HTTP client for the kernelGym evaluation service.

Migrated from rllm-lilac's ``_HybridHttpWorker`` (sync ``httpx``) to ``aiohttp``
so it composes with vime's async rollout loop. The protocol is unchanged:

    POST /evaluate            submit a kernel for compile+correctness+perf eval
    GET  /status/{task_id}    poll until terminal (completed/failed/...)
    GET  /results/{task_id}   fetch the full EvaluationResponse

Key behaviours preserved from rllm:
  * A read timeout on POST /evaluate does NOT trigger a resubmit (the server
    may already have accepted the task under that task_id); we fall through to
    polling instead.
  * 429/503 are retried with exponential backoff.
  * Polling uses a fixed interval and an overall client-side timeout.

kernelGym defaults: host ``0.0.0.0``, port ``10907`` (see kernelGym-NPU
``kernelgym/config/settings.py``). Override via ``--custom-config-path``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp
else:
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - only needed when actually hitting the service
        aiohttp = None

logger = logging.getLogger(__name__)

# rllm prepends these imports to every generated kernel before evaluation so the
# model does not need to emit boilerplate. Kept byte-for-byte identical.
KERNEL_CODE_IMPORT_PREFIX = (
    "import triton \nimport triton.language as tl\nimport torch\nimport torch.nn as nn\n"
)

TERMINAL_STATUSES = ("completed", "failed", "timeout", "cancelled")

_shared_session: aiohttp.ClientSession | None = None


def get_session(request_timeout: float = 120.0, connect_timeout: float = 10.0) -> aiohttp.ClientSession:
    """Return a process-wide shared aiohttp session (lazily created on the
    running event loop). ``request_timeout`` bounds a single POST/GET; the
    overall evaluation wait is bounded separately by ``client_timeout`` in
    :func:`submit_and_poll`."""
    global _shared_session
    if aiohttp is None:
        raise RuntimeError(
            "aiohttp is required to talk to the kernelGym service but is not installed. "
            "Install it in the training environment (pip install aiohttp)."
        )
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=128,
            limit_per_host=128,
            keepalive_timeout=30.0,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=request_timeout, sock_connect=connect_timeout)
        _shared_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,
            headers={"Content-Type": "application/json"},
        )
    return _shared_session


async def close_session() -> None:
    global _shared_session
    if _shared_session is not None and not _shared_session.closed:
        await _shared_session.close()
    _shared_session = None


def _backoff(attempt: int, base: int = 2, cap: int = 30) -> float:
    return float(min(base**attempt, cap))


def preflight_validate(reference_code: str, kernel_code: str, entry_point: str) -> tuple[bool, str]:
    """Cheap local check that the reference defines ``class {ep}(...Module)`` and
    the kernel defines ``class {ep}New(...Module)``, so we skip a doomed HTTP
    round-trip. Mirrors rllm ``_preflight_validate`` (regex-based)."""
    ep = re.escape(entry_point or "Model")
    try:
        ref_ok = bool(re.search(rf"class {ep}\s*\(.*Module\s*\)", reference_code or ""))
        ker_ok = bool(re.search(rf"class {ep}New\s*\(.*Module\s*\)", kernel_code or ""))
    except re.error as exc:  # pragma: no cover - defensive: never block on regex
        logger.debug("preflight skipped due to regex error: %s", exc)
        return True, ""
    if ref_ok and ker_ok:
        return True, ""
    missing = []
    if not ref_ok:
        missing.append(f"class {entry_point}(nn.Module)")
    if not ker_ok:
        missing.append(f"class {entry_point}New(nn.Module)")
    return False, ", ".join(missing)


def build_eval_payload(
    *,
    task_id: str,
    reference_code: str,
    kernel_code: str,
    entry_point: str,
    backend: str,
    train_id: str = "",
    eval_tag: str = "train",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    server_timeout: int = 600,
    is_valid: bool = False,
    enable_profiling: bool = True,
    verbose_errors: bool = True,
    detect_decoy_kernel: bool = True,
    reference_backend: str | None = None,
    llm_messages: str = "",
) -> dict[str, Any]:
    """Build the POST /evaluate body (kernelGym ``EvaluationRequest``).

    Note: ``backend`` is the kernel backend ("triton"|"cuda"); ``reference_backend``
    controls how the reference is run ("pytorch"|"compile"|None). When validating
    (is_valid), decoy detection is forced on, matching rllm.
    """
    if is_valid:
        detect_decoy_kernel = True
    return {
        "task_id": task_id,
        "train_id": train_id,
        "eval_tag": eval_tag,
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "backend": backend,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "timeout": server_timeout,
        "priority": "normal",
        "entry_point": entry_point,
        "is_valid": is_valid,
        "verbose_errors": verbose_errors,
        "enable_profiling": enable_profiling,
        "detect_decoy_kernel": detect_decoy_kernel,
        "reference_backend": reference_backend,
        "llm_messages": llm_messages,
    }


async def _submit(session: aiohttp.ClientSession, server_url: str, payload: dict, max_retries: int | None) -> bool:
    """POST the task. Returns True if accepted (or read-timed-out after the
    server likely accepted), False on terminal submission failure."""
    url = f"{server_url}/evaluate"
    task_id = payload.get("task_id", "")
    attempt = 0
    unlimited = max_retries is None or max_retries == -1
    while unlimited or attempt < (max_retries or 0):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                if resp.status in (429, 503):
                    await asyncio.sleep(_backoff(attempt, base=2 if resp.status == 429 else 5))
                    attempt += 1
                    continue
                resp.raise_for_status()
        except asyncio.TimeoutError:
            # Server may have accepted the synchronous /evaluate; do not resubmit
            # the same task_id — fall through to polling.
            logger.warning("kernelgym POST /evaluate timeout task_id=%s; polling existing task", task_id)
            return True
        except aiohttp.ClientConnectionError as exc:
            if unlimited or attempt < (max_retries or 0) - 1:
                await asyncio.sleep(_backoff(attempt))
                attempt += 1
                continue
            logger.warning("kernelgym submit connection failure task_id=%s: %s", task_id, exc)
            return False
        except Exception as exc:  # noqa: BLE001 - any other HTTP error is terminal for this submit
            logger.warning("kernelgym submit error task_id=%s: %s", task_id, exc)
            return False
    return False


async def submit_and_poll(
    session: aiohttp.ClientSession,
    server_url: str,
    payload: dict,
    *,
    client_timeout: float,
    max_retries: int | None = 2,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    """Submit ``payload`` and poll until a terminal status or ``client_timeout``.

    Returns the merged result dict (kernelGym ``EvaluationResponse`` with the
    final ``status`` set), or a synthetic ``{"status": "failed"|"timeout", ...}``
    dict on submission failure / client-side timeout.
    """
    task_id = payload.get("task_id", "")
    start = time.monotonic()

    submitted = await _submit(session, server_url, payload, max_retries)
    if not submitted:
        return {"status": "failed", "error_message": f"Failed to submit task {task_id}"}

    last_status = None
    while time.monotonic() - start < client_timeout:
        try:
            async with session.get(f"{server_url}/status/{task_id}") as s:
                if s.status == 200:
                    data = await s.json()
                    status = data.get("status", "unknown")
                    if status != last_status:
                        last_status = status
                        logger.debug("kernelgym status task_id=%s -> %s", task_id, status)
                    if status in TERMINAL_STATUSES:
                        async with session.get(f"{server_url}/results/{task_id}") as r:
                            if r.status == 200:
                                result = await r.json()
                                result["status"] = status
                                return result
                            return {
                                "status": status,
                                "error_message": f"Failed to fetch results: HTTP {r.status}",
                            }
        except Exception as exc:  # noqa: BLE001 - transient poll errors are retried
            logger.debug("kernelgym poll error task_id=%s: %s", task_id, exc)
        await asyncio.sleep(poll_interval)

    return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"}


async def evaluate_kernel(
    server_url: str,
    payload: dict,
    *,
    client_timeout: float,
    request_timeout: float = 120.0,
    max_retries: int | None = 2,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    """High-level entry: submit a prepared payload and return the result dict."""
    session = get_session(request_timeout=request_timeout)
    return await submit_and_poll(
        session,
        server_url,
        payload,
        client_timeout=client_timeout,
        max_retries=max_retries,
        poll_interval=poll_interval,
    )


async def check_health(server_url: str, request_timeout: float = 10.0) -> dict[str, Any]:
    """GET /health — handy for startup sanity checks and tests."""
    session = get_session(request_timeout=request_timeout)
    async with session.get(f"{server_url}/health") as r:
        return await r.json()
