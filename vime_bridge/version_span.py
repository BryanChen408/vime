"""version-span guard (vime side): push the current weight version to the polar gateway
during the weight-sync pause.

The gateway stamps each rollout turn with this version and rejects a session continuation
whose next turn would cross a weight update (mixed-weight = v_N prefix + v_{N+1} suffix).
Complements cancellation and abort fencing as a defense in depth.  See polar-side
prefix_merging cross-version fallback + gateway entry interception
(docs/design/polar_version_span_exclusion.md).

Hard constraint: MUST be pushed DURING the pause, BEFORE resume -- otherwise a turn can be
generated with new weights but stamped with the old version, and the span goes undetected.

The caller treats False as a fatal, fail-closed boundary error.  A partial gateway update
must never be accepted because it would stamp the same rollout fleet with different versions.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_push_url(args: Any) -> str | None:
    """Target the same gateway the agents hit.  Prefer the rollout server (which routes
    ``/rollout/admin/...`` to the gateway, mirroring ``_pause_gateway_generation``); fall
    back to a direct gateway URL.  Returns None when neither is configured (feature off)."""
    rollout_url = getattr(args, "polar_url", None) or getattr(args, "polar_rollout_url", None)
    if rollout_url:
        return f"{str(rollout_url).rstrip('/')}/rollout/admin/policy_version"
    gateway_url = getattr(args, "polar_gateway_url", None)
    if gateway_url:
        return f"{str(gateway_url).rstrip('/')}/admin/policy_version"
    return None


def push_policy_version_to_gateway(args: Any, version: int) -> bool:
    """POST ``version`` and require an explicit acknowledgement from every gateway."""
    url = _resolve_push_url(args)
    if not url:
        return False
    try:
        import httpx

        timeout = float(getattr(args, "polar_gateway_control_timeout", 30.0))
        with httpx.Client(timeout=max(timeout, 5.0)) as client:
            response = client.post(url, params={"version": int(version)})
            response.raise_for_status()
            payload = response.json()
        rollout_url = getattr(args, "polar_url", None) or getattr(
            args, "polar_rollout_url", None
        )
        if rollout_url:
            acknowledged = (
                isinstance(payload, dict)
                and payload.get("all_updated") is True
                and payload.get("policy_version") == int(version)
            )
        else:
            acknowledged = (
                isinstance(payload, dict)
                and payload.get("policy_version") == int(version)
            )
        if not acknowledged:
            logger.error(
                "version-span: gateway did not acknowledge policy_version=%s: %r",
                version,
                payload,
            )
            return False
        logger.info("version-span: pushed policy_version=%s to gateway (%s)", version, url)
        return True
    except Exception:
        logger.warning(
            "version-span: failed to push policy_version=%s to every gateway",
            version,
            exc_info=True,
        )
        return False
