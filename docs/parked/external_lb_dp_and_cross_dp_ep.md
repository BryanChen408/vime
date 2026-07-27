# Parked: external-LB DP and cross-DP EP

Split off `dev/ascend-integrated` on 2026-07-27. Neither feature pays for itself on the
current workload, so they are kept here instead of on the integration branch. Nothing on
this branch has been re-verified against the new baseline.

## Why they were parked

External-LB DP measured a 2.3x net throughput loss on bursty agentic rollout (15 -> 35
minutes; reverting the flag restored it). The idle-rank forward hypothesis behind that
number was never confirmed. Reference numbers published for this mode come from saturated
serving, which is a different regime.

Cross-DP EP was never separated from that measurement, so it inherits the same doubt.

## What belongs here

| Piece | Where it came from |
|---|---|
| `_allocate_external_lb_addr_and_ports` and its dispatch | `vime/ray/rollout.py` |
| `ResourceLayout.vllm_dp_size` plus the engine-count check in `_apply_resource_layout` | `vime/ray/resource_layout.py`, `vime/utils/arguments.py` |
| DP rank and size plumbing into the engine subprocess | `vime/backends/vllm_utils/vllm_engine.py` — **not ported yet**, see below |
| `test_rollout_dp_alloc.py` | `tests/unit/ray/` |
| Cross-DP EP | Launch scripts only, as `--vllm-additional-config` keys; no library code |

The LB proxy itself is **not** part of this. It runs with DP off (`FEAT_LB_PROXY=1`,
`FEAT_DP_EXTERNAL_LB=0`) and lives on the integration branch, because it replaces the Rust
router to keep `return_token_ids` intact.

## Before reviving

Re-measure on the workload that will actually run. If the rollout is bursty rather than
saturated, expect the loss to reproduce. The reviving change also has to re-apply the
non-2xx passthrough contract the LB proxy already carries, since a DP fan-out multiplies
the number of upstreams that can fail.

## What is still missing

The engine-side plumbing has not been brought over. On `feature/swe-tasks` it is roughly 26
lines inside `vime/backends/vllm_utils/vllm_engine.py`, interleaved with unrelated changes
from that branch:

- `_resolve_vllm_parallel_sizes` dividing by `pp * dp` under the flag
- `build_vllm_cmd_and_env` forwarding `--data-parallel-rank`, `--data-parallel-address` and
  `--data-parallel-rpc-port`
- the four `data_parallel_*` fields threaded through `_compute_server_args`

Whoever revives this has to separate those hunks from the rest of that diff rather than
applying it wholesale. Without them the allocator here produces DP addresses that never
reach the engine, so the feature is inert as it stands.
