"""Regression for the version-span acknowledgement predicate.

``push_policy_version_to_gateway`` is a fail-closed boundary guard: a False
return aborts the training step rather than let a rollout fleet get stamped
with two different weight versions. It therefore has to be strict about
*partial* acknowledgement — and it also has to actually parse what the polar
rollout server sends back.

On 20260829 it did not. The vime side required a summarized
``{"all_updated": true, "policy_version": N}``, mirroring the ``all_paused``
field that ``/rollout/admin/inference/pause`` returns. But the sibling
``/rollout/admin/policy_version`` endpoint never grew that summary: it forwards
through ``_forward_gateway_admin`` and returns the raw fan-out. So a gateway
that *had* accepted the version::

    {"nodes": [{"node_id": "ascend-node-01", "status": "ok",
                "response": {"policy_version": 1}}]}

was read as a refusal, and the first training step died with
``Failed to publish Polar policy_version=1 while admission was paused``.

The fix accepts the raw shape by recomputing the same predicate the server
would have. The point of these cases is that widening the *shape* did not
widen the *guarantee*: every node must be reached and every node must report
the requested version.
"""

from __future__ import annotations

import pytest

from vime_bridge.rollout import _all_gateway_nodes_ok
from vime_bridge.version_span import _acknowledged_via_rollout_server as acknowledged


def _node(version, *, status="ok", node_id="ascend-node-01"):
    return {"node_id": node_id, "status": status, "response": {"policy_version": version}}


def test_raw_fanout_from_the_run_that_died_is_accepted():
    """The verbatim payload from the 20260829 failure must acknowledge."""
    payload = {
        "nodes": [
            {"node_id": "ascend-node-01", "status": "ok", "response": {"policy_version": 1}}
        ]
    }
    assert acknowledged(payload, 1) is True


def test_summarized_shape_still_works():
    """Kept so a future polar-side summary does not regress this path."""
    assert acknowledged({"all_updated": True, "policy_version": 1}, 1) is True


def test_summary_wins_when_it_says_not_all_updated():
    """An explicit negative summary is authoritative; do not fall through to nodes."""
    payload = {
        "all_updated": False,
        "policy_version": 1,
        "nodes": [_node(1)],
    }
    assert acknowledged(payload, 1) is False


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"nodes": [_node(1), {"node_id": "n2", "status": "error", "error": "boom"}]},
                     id="one_node_unreachable"),
        pytest.param({"nodes": [_node(1), _node(0, node_id="n2")]},
                     id="one_node_stamped_older_version"),
        pytest.param({"nodes": []}, id="no_nodes"),
        pytest.param({"status": "ok"}, id="no_nodes_key"),
        pytest.param({"nodes": [{"node_id": "n1", "status": "ok", "response": None}]},
                     id="response_not_a_dict"),
        pytest.param({"nodes": [{"node_id": "n1", "status": "ok"}]}, id="response_missing"),
        pytest.param(["not", "a", "dict"], id="payload_not_a_dict"),
        pytest.param(None, id="payload_none"),
    ],
)
def test_partial_or_malformed_acknowledgement_stays_fail_closed(payload):
    assert acknowledged(payload, 1) is False


def test_version_mismatch_is_rejected_in_both_shapes():
    assert acknowledged({"nodes": [_node(1)]}, 2) is False
    assert acknowledged({"all_updated": True, "policy_version": 1}, 2) is False


def test_every_node_must_match_not_just_one():
    """The whole point of the guard: a partial fleet update is not an update."""
    payload = {"nodes": [_node(2), _node(2, node_id="n2"), _node(1, node_id="n3")]}
    assert acknowledged(payload, 2) is False
    payload["nodes"][2] = _node(2, node_id="n3")
    assert acknowledged(payload, 2) is True


# ---------------------------------------------------------------------------
# The same defect, one endpoint over: /rollout/admin/inference/resume
#
# The rollout server summarizes exactly one of its three fan-out endpoints.
# ``/rollout/admin/inference/pause`` returns all_paused/all_drained/inflight;
# ``/resume`` and ``/policy_version`` return _forward_gateway_admin's raw
# ``{"nodes": [...]}``. vime asked all three for a summary, so a fully resumed
# fleet came back looking like a refusal and killed the step at
# ``finish_policy_update``. Fixing only policy_version left this one live, so
# both are pinned here.
# ---------------------------------------------------------------------------

def _resumed(response):
    return response.get("paused") is False


def _ok(payload):
    return _all_gateway_nodes_ok(payload, "all_resumed", _resumed)


def test_raw_resume_fanout_from_the_run_that_died_is_accepted():
    payload = {
        "nodes": [
            {
                "node_id": "ascend-node-01",
                "status": "ok",
                "response": {
                    "paused": False,
                    "drained": True,
                    "inflight": 0,
                    "base_url": "http://80.48.5.52:8001",
                    "engine": "vllm",
                    "request_timeout_seconds": 14400.0,
                },
            }
        ]
    }
    assert _ok(payload) is True


def test_resume_summary_is_authoritative_in_both_directions():
    assert _ok({"all_resumed": True}) is True
    # A false summary must not be overturned by the per-node fallback.
    assert _ok({"all_resumed": False, "nodes": [{"status": "ok", "response": {"paused": False}}]}) is False


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"nodes": [{"status": "ok", "response": {"paused": False}},
                                {"status": "ok", "response": {"paused": True}}]},
                     id="one_node_still_paused"),
        pytest.param({"nodes": [{"status": "ok", "response": {"paused": False}},
                                {"status": "error", "error": "unreachable"}]},
                     id="one_node_unreachable"),
        pytest.param({"nodes": []}, id="no_nodes"),
        pytest.param({"status": "ok"}, id="no_nodes_key"),
        pytest.param({"nodes": [{"status": "ok", "response": {}}]}, id="paused_field_missing"),
        pytest.param({"nodes": [{"status": "ok", "response": None}]}, id="response_not_a_dict"),
        pytest.param(["not", "a", "dict"], id="payload_not_a_dict"),
    ],
)
def test_resume_stays_fail_closed(payload):
    assert _ok(payload) is False
