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

## Open item

Branch `feat/ascendc-rl-t2a` is TENTATIVE and does **not exist yet** on
`BryanChen408/ProRL-Agent-Server` (which currently has only `feat/ascend-smoke`
and `feat/swe-tasks`). It needs to be created from the curated reorg work and
pushed before `git checkout feat/ascendc-rl-t2a` above will work. Pin an exact
commit here once it lands.
