import logging

import ray

from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from vime.utils.misc import should_run_periodic_action


logger = logging.getLogger(__name__)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with vLLM engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with vLLM metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        # [vime 2026-08-24 定案] 同步窗口 KV 不驻留(verl/slime 同款):只醒权重壳
        # 灌新权重,KV 同步后才醒。同步窗口瞬时最重(all_gather/reload 搅动),
        # 必须让它空 —— 权重壳 35G+workspace 4G+trainer ~10G ≈ 50G,余量 11G+;
        # KV 重映射只是同步后一次 ~1.4G 小额申请,放在 clear_memory 之后做。
        # (此前"KV 先醒"把 KV 塞进同步窗口 → 顶格 OOM,141406/142800 实锤,回正。)
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
    # Advance the rollout manager's policy version so custom rollout functions that
    # track off-policy staleness (e.g. vime_bridge) see the initial weights.
    # No-op for rollout functions without an update_policy_version hook.
    ray.get(rollout_manager.update_policy_version.remote(args.start_rollout_id))

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        # 同步完成后醒 KV(与每步同序:权重壳 → 同步 → KV)。
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        # offload_rollout puts the engines to sleep for the whole training step, so from
        # here until onload_kv below there is nothing serving inference. Rollout functions
        # that drive an external agent gateway (vime_bridge/polar) keep admitting sessions
        # on their own thread and would fire them at sleeping engines; prepare_policy_update
        # pauses the gateway and drains what is in flight first. No-op for rollout functions
        # without the hook.
        next_policy_version = rollout_id + 1
        if args.offload_rollout:
            # Best-effort: the drain raises PolarRolloutSchedulerError on timeout
            # (vime_bridge/rollout.py wait_for_policy_update_drain), and an agent session
            # can legitimately outlive the timeout. Failing to drain only means a few
            # in-flight sessions hit sleeping engines and come back as ERROR trajectories,
            # which are excluded from training anyway -- not a reason to kill the run.
            try:
                ray.get(rollout_manager.prepare_policy_update.remote(next_policy_version))
            except Exception:
                logger.warning(
                    "prepare_policy_update failed before offloading rollout engines; "
                    "in-flight sessions may error out. Continuing.",
                    exc_info=True,
                )
        try:
            if args.offload_rollout:
                ray.get(rollout_manager.offload.remote())

            actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

            if args.use_critic:
                value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

            if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                save(rollout_id)

            offload_train(actor_trains_this_step)
            if args.offload_rollout:
                # 同步窗口 KV 不驻留(定案,见文件头注):只醒权重壳做同步。
                ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()
            # Advance policy version so off-policy staleness tracking (vime_bridge) stays
            # live; a frozen version makes staleness grow without bound and drops every
            # group as stale (see vime_bridge/rollout.py drain_completed ->
            # max_off_policy_steps), which hangs training a few rollouts in.
            ray.get(rollout_manager.update_policy_version.remote(next_policy_version))
            if args.offload_rollout:
                # 同步完成 + clear_memory 之后,才醒 KV(此时空闲最足,重映射最稳)。
                ray.get(rollout_manager.onload_kv.remote())
        finally:
            # Engines are serving again -> let the gateway resume. In finally so a failed
            # training step can't strand the external gateway in the paused state.
            if args.offload_rollout:
                ray.get(rollout_manager.finish_policy_update.remote(next_policy_version))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
