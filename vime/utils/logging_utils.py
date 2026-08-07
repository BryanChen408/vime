import logging

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not wandb_utils.should_init_wandb(args):
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if wandb_utils.should_init_wandb(args):
        # wandb "shared" 模式下,非 primary 进程(RolloutManager 用 primary=False)不能读全局
        # step 去自增 → wandb.log(metrics) 会抛 "Cannot read the W&B step in shared mode",
        # rollout/eval 指标被静默丢掉(实测看板缺 rollout/rewards、eval/aime)。显式传 step
        # (每进程单调:rollout/eval=compute_rollout_step(rollout_id),train=accumulated_step_id;
        # 就是下面 TensorBoard 用的同一个值)就不需要读全局 step,secondary 也能正常写。
        step = metrics.get(step_key)
        if step is not None:
            wandb.log(metrics, step=int(step))
        else:
            wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
