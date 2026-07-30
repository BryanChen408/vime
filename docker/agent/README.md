# vime — Docker / deployment

Container images and build scripts for the vime stack on **Ascend A3
(910_93-class) / aarch64, CANN 9.0.0**. Three pieces:

| dir | image | what it is |
|-----|-------|------------|
| [`training/`](training/) | `vime:a3-release` | the RL **training** image — vllm / vllm-ascend / Megatron / MindSpeed / fla_npu, each cloned from its **official upstream** at a pinned ref + a small vime patch. NPU custom kernels are compiled on first run (no NPU in `docker build`). |
| [`sandbox/`](sandbox/) | `ascendc-tilelang:v1` | the **sandbox** image the Polar agent runs tasks in — TileLang (compiled from source) + torch/torch_npu + cmake<4 + Claude Code CLI, for AscendC kernel-generation tasks. |
| [`polar/`](polar/) | — | **Polar** (`ProRL-Agent-Server`) — the agentic RL server (SWE / operator / math). Pulled from its own repo; drives training via slime_bridge and runs task rollouts inside the sandbox image. |

## How they fit together

```
                 ┌─────────────────────────┐
  vime training  │  training/  vime:a3-release   (RL trainer: actor + rollout)
                 └───────────┬─────────────┘
                             │ slime_bridge (weight sync, task I/O)
                 ┌───────────┴─────────────┐
   agentic RL    │  polar/   ProRL-Agent-Server  (agent server: SWE/operator/math)
                 └───────────┬─────────────┘
                             │ launches task containers
                 ┌───────────┴─────────────┐
   task sandbox  │  sandbox/  ascendc-tilelang   (per-task exec env: tilelang/AscendC)
                 └─────────────────────────┘
```

Each subdir has its own README with exact build/run steps. Start with
`training/README.md`.

## Notes for committing to the vime repo

- `training/wheels/*.whl` (the triton-ascend wheel, ~188 MB) is **git-ignored** —
  it is downloaded per `training/README.md`, not committed.
- `training/patches/*.patch` (small text) and all Dockerfiles/scripts are committed.
- Polar itself is a separate repo (not vendored here) — `polar/` only documents
  how to pull and deploy it.
