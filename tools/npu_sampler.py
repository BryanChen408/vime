#!/usr/bin/env python3
"""Single-flight npu-smi sampler with an atomic local cache.

The sampler is deliberately separated from the HTTP exporter.  If npu-smi is
blocked in an uninterruptible driver call, this process may wait for it, but no
HTTP request can start another probe and the last successful sample remains
available to readers.
"""

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time


DEFAULT_CACHE = "/tmp/vime_npu_metrics.json"
DEFAULT_LOCK = "/tmp/vime_npu_sampler.lock"
_STOP = False
_ACTIVE_CHILD = None


def parse_npu_info(out):
    """Parse both single-die 910B and dual-die A3 npu-smi tables."""
    cards, module_temp = {}, {}
    r1 = re.compile(r"\|\s*(\d+)\s+\S+\s*\|\s*\w+\s*\|\s*(?:[\d.]+|-)\s+(\d+)\s")
    r2 = re.compile(r"\|\s*(\d+)(?:\s+(\d+))?\s*\|\s*[\w:.]+\s*\|"
                    r"\s*([\d.]+)\s+[\d.]+\s*/\s*[\d.]+\s+(\d+)\s*/\s*(\d+)")
    last_npu_id = None
    for line in out.splitlines():
        match = r1.search(line)
        if match:
            last_npu_id = int(match.group(1))
            module_temp[last_npu_id] = int(match.group(2))
            continue
        match = r2.search(line)
        if not match:
            continue
        if match.group(2) is not None:
            card_id = int(match.group(2))
            temp = module_temp.get(card_id // 2, 0)
        else:
            card_id = last_npu_id if last_npu_id is not None else int(match.group(1))
            temp = module_temp.get(card_id, 0)
        cards[card_id] = {
            "id": card_id,
            "power": 0.0,
            "temp": temp,
            "aicore": int(float(match.group(3))),
            "hbm_used": int(match.group(4)),
            "hbm_total": int(match.group(5)),
        }
    return [cards[i] for i in sorted(cards)]


def _load_cache(path):
    try:
        with open(path, encoding="utf-8") as cache_file:
            data = json.load(cache_file)
        if not isinstance(data, dict):
            raise ValueError("cache root is not an object")
    except Exception:
        data = {}
    if not isinstance(data.get("npu"), list):
        data["npu"] = []
    return data


def _atomic_write(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".vime_npu_metrics.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, separators=(",", ":"), sort_keys=True)
            cache_file.write("\n")
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _publish(path, state, *, status, error=None, child_pid=None, attempt_ts=None):
    payload = dict(state)
    payload.update({
        "schema_version": 1,
        "collector_pid": os.getpid(),
        "child_pid": child_pid,
        "status": status,
        "ok": status == "ok",
        "stale": status != "ok",
        "error": error,
        "updated_at": time.time(),
    })
    if attempt_ts is not None:
        payload["last_attempt_ts"] = attempt_ts
    _atomic_write(path, payload)
    state.clear()
    state.update(payload)


def _acquire_singleton(path):
    lock_file = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def _handle_stop(_signum, _frame):
    global _STOP
    _STOP = True
    if _ACTIVE_CHILD is not None and _ACTIVE_CHILD.poll() is None:
        try:
            _ACTIVE_CHILD.terminate()
        except ProcessLookupError:
            pass


def _sleep(seconds):
    deadline = time.monotonic() + max(0.0, seconds)
    while not _STOP and time.monotonic() < deadline:
        time.sleep(min(0.5, deadline - time.monotonic()))


def collect_once(command, cache_path, timeout, state):
    """Run one probe; a slow child is observed but never killed automatically."""
    global _ACTIVE_CHILD
    attempt_ts = time.time()
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception as exc:
        _publish(cache_path, state, status="spawn_error",
                 error=type(exc).__name__, attempt_ts=attempt_ts)
        return False, False

    _ACTIVE_CHILD = proc
    _publish(cache_path, state, status="collecting", child_pid=proc.pid,
             attempt_ts=attempt_ts)
    was_slow = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        was_slow = True
        # This is a soft deadline. Killing npu-smi while it owns driver handles
        # can itself block in devdrv_host_notice_dev_process_exit. Keep waiting
        # for this one child instead; the singleton lock prevents another probe.
        _publish(cache_path, state, status="slow",
                 error=f"running>{timeout:g}s", child_pid=proc.pid,
                 attempt_ts=attempt_ts)
        stdout, stderr = proc.communicate()
        _ACTIVE_CHILD = None
    finally:
        if proc.poll() is not None:
            _ACTIVE_CHILD = None

    if proc.returncode != 0:
        detail = (stderr or "").strip().replace("\n", " ")[-300:]
        error = f"exit={proc.returncode}" + (f": {detail}" if detail else "")
        _publish(cache_path, state, status="command_error", error=error,
                 attempt_ts=attempt_ts)
        return False, was_slow

    cards = parse_npu_info(stdout)
    if not cards:
        _publish(cache_path, state, status="parse_empty", error="no cards parsed",
                 attempt_ts=attempt_ts)
        return False, was_slow

    state["npu"] = cards
    state["last_success_ts"] = time.time()
    _publish(cache_path, state, status="ok", attempt_ts=attempt_ts)
    return True, was_slow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.environ.get("NPU_METRICS_CACHE", DEFAULT_CACHE))
    parser.add_argument("--lock", default=os.environ.get("NPU_SAMPLER_LOCK", DEFAULT_LOCK))
    parser.add_argument("--interval", type=float,
                        default=float(os.environ.get("NPU_SAMPLER_INTERVAL", "60")))
    parser.add_argument("--timeout", type=float,
                        default=float(os.environ.get("NPU_SAMPLER_TIMEOUT", "30")))
    parser.add_argument("--failure-backoff", type=float,
                        default=float(os.environ.get("NPU_SAMPLER_FAILURE_BACKOFF", "300")))
    parser.add_argument("--initial-delay", type=float,
                        default=float(os.environ.get("NPU_SAMPLER_INITIAL_DELAY", "0")))
    parser.add_argument("--command", default=os.environ.get("NPU_SMI_COMMAND", "npu-smi info"),
                        help="command used by the sampler; primarily useful for isolated tests")
    args = parser.parse_args()

    lock_file = _acquire_singleton(args.lock)
    if lock_file is None:
        print(f"[npu-sampler] another sampler owns {args.lock}; exiting", flush=True)
        return 0

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    command = shlex.split(args.command)
    if not command:
        raise SystemExit("--command must not be empty")
    state = _load_cache(args.cache)
    print(f"[npu-sampler] cache={args.cache} interval={args.interval:g}s "
          f"timeout={args.timeout:g}s", flush=True)
    _sleep(args.initial_delay)
    while not _STOP:
        _ok, was_slow = collect_once(command, args.cache, args.timeout, state)
        delay = args.failure_backoff if was_slow else args.interval
        # Count the interval from full child teardown, not probe start.  A D-state
        # cleanup may take minutes; starting again immediately after it recovers
        # would put pressure straight back onto the driver control plane.
        _sleep(delay)
    lock_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
