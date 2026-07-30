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
# branch: TBD — the repo is being reorganized into curated scenario branches
#   (main-npu + feat/operator|swe|math). Pin the intended branch/commit here
#   once finalized.
pip install -e .          # pure-python; no NPU compile needed
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

The exact branch to pin and the polar deployment profiles/images are
team-specific and still in flux (curated reorg in progress). Fill in the pinned
ref + the concrete profile once decided, so this is reproducible.
