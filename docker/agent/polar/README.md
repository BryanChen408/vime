# polar — ProRL-Agent-Server

Polar is the **agentic RL server** for SWE / operator / math scenarios. It is a
separate repo (a fork of NVIDIA-NeMo/ProRL-Agent-Server with the NPU adaptation
layer), so it is **not vendored here** — this dir only documents how to pull and
deploy it alongside the vime training image.

- package: `polar` (pure Python; `src/{polar, ascend_operator, slime_bridge}`)
- role: drives task rollouts and bridges to the vime trainer via `slime_bridge`
  (weight sync, task request/response); co-located with training (reserves a
  slice of NPUs via `polar_reserved`).
- runs each task inside the **sandbox** image (`ascendc-tilelang`, see
  `../sandbox/`).

## Pull

```
git clone https://github.com/BryanChen408/ProRL-Agent-Server.git polar
cd polar
git checkout feat/ascendc-rl-t2a   # tentative branch for the ascendc/operator + t2a scenario
pip install -e .                   # pure-python; no NPU compile needed
```

## Deploy

Polar is launched from its own `deploy/ascend_operator/` scripts. After building
the sandbox image and pointing the profile at it (see `../sandbox/README.md`):

```
cd polar
POLAR_PROFILE=deploy/ascend_operator/profile.ascendc.yaml \
POLAR_RUN_ID=polar_$(date +%Y%m%d_%H%M%S) \
NO_PROXY=127.0.0.1,localhost,<hosts> no_proxy=127.0.0.1,localhost,<hosts> \
  bash deploy/ascend_operator/restart_polar_host.sh
```

## Scenarios (profile + runtime)

`feat/ascendc-rl-t2a` ships three operator scenarios; pick the matching polar
profile + runtime dir:

| scenario | profile | runtime dir | sandbox |
|---|---|---|---|
| triton  | `deploy/ascend_operator/profile.vime.yaml` | `operator_runtime` | triton sandbox |
| ascendc (t2a) | `deploy/ascend_operator/profile.t2a.yaml` | `operator_runtime_t2a` | `ascendc-tilelang` |

See `../RUNBOOK.md` for the full launch flow (training container + polar host +
resource layout). The polar profile's inference endpoint (`sglang_router_url`
field, legacy name) points at vime's vLLM router `:8001`.

## Pin

Tentative branch `feat/ascendc-rl-t2a` — currently at `4f0e1a4e`. Pin an exact
commit for reproducibility once it stabilizes.
