import logging

import ray

from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from vime.utils.misc import should_run_periodic_action


logger = logging.getLogger(__name__)


def _prepare_rollout_memory_handoff(args, actor_model, rollout_manager) -> None:
    """Quiesce colocated trainer allocators before restoring rollout weights."""
    if not args.offload_rollout:
        return
    actor_model.prepare_memory_handoff()
    ray.get(rollout_manager.onload_weights.remote())


def _finish_rollout_memory_handoff(args, actor_model, rollout_manager) -> None:
    """Restore rollout KV first, then return trainer allocators to train mode."""
    if not args.offload_rollout:
        return
    ray.get(rollout_manager.onload_kv.remote())
    actor_model.finish_memory_handoff()


def train(args):
    configure_logger()
    durable_polar_boundary = bool(
        getattr(args, "polar_policy_transition_enabled", False)
    )
    if durable_polar_boundary and not args.offload_rollout:
        raise RuntimeError(
            "--polar-policy-transition-enabled currently requires --offload-rollout; "
            "otherwise training can overlap an open serving engine"
        )
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

    initial_boundary_attempted = False
    try:
        if durable_polar_boundary:
            # Bootstrap is a real transaction: close the previous run namespace and
            # abort all old engine work before the first weight mutation.
            initial_boundary_attempted = True
            ray.get(
                rollout_manager.prepare_initial_policy.remote(args.start_rollout_id)
            )

        # [vime 2026-08-24 定案] 同步窗口 KV 不驻留(verl/slime 同款):只醒权重壳
        # 灌新权重,KV 同步后才醒。同步窗口瞬时最重(all_gather/reload 搅动),
        # 必须让它空 —— 权重壳 35G+workspace 4G+trainer ~10G ≈ 50G,余量 11G+;
        # KV 重映射只是同步后一次 ~1.4G 小额申请,放在 clear_memory 之后做。
        # (此前"KV 先醒"把 KV 塞进同步窗口 → 顶格 OOM,141406/142800 实锤,回正。)
        _prepare_rollout_memory_handoff(args, actor_model, rollout_manager)

        # Always push actor weights to rollout once weights are loaded.
        actor_model.update_weights()
        if not durable_polar_boundary:
            # Preserve the original hook order for every non-transactional rollout.
            ray.get(
                rollout_manager.update_policy_version.remote(args.start_rollout_id)
            )

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        # 同步完成后醒 KV(与每步同序:权重壳 → 同步 → KV)。
        _finish_rollout_memory_handoff(args, actor_model, rollout_manager)

        if durable_polar_boundary:
            ray.get(
                rollout_manager.finish_initial_policy.remote(args.start_rollout_id)
            )
            initial_boundary_attempted = False
    except Exception as exc:
        if initial_boundary_attempted:
            try:
                ray.get(
                    rollout_manager.fail_policy_update.remote(
                        args.start_rollout_id,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to persist Polar bootstrap failure; admission remains closed"
                )
        raise

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
        policy_update_attempted = False

        try:
            if args.offload_rollout:
                # Mark the attempt before the RPC: a partial remote prepare must also
                # be driven to a durable fail-closed state if the acknowledgement dies.
                policy_update_attempted = True
                ray.get(
                    rollout_manager.prepare_policy_update.remote(next_policy_version)
                )
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
            # 同步窗口 KV 不驻留(定案,见文件头注):只醒权重壳做同步。
            _prepare_rollout_memory_handoff(args, actor_model, rollout_manager)
            actor_model.update_weights()
            # Advance policy version so off-policy staleness tracking (vime_bridge) stays
            # live; a frozen version makes staleness grow without bound and drops every
            # group as stale (see vime_bridge/rollout.py drain_completed ->
            # max_off_policy_steps), which hangs training a few rollouts in.
            if not durable_polar_boundary:
                ray.get(
                    rollout_manager.update_policy_version.remote(next_policy_version)
                )
            if args.offload_rollout:
                # 同步完成 + clear_memory 之后,才醒 KV(此时空闲最足,重映射最稳)。
                _finish_rollout_memory_handoff(args, actor_model, rollout_manager)
                # Resume only after train, weight sync, KV restore, and an all-engine
                # weight-version proof succeeded.
                ray.get(rollout_manager.finish_policy_update.remote(next_policy_version))
                policy_update_attempted = False
        except Exception as exc:
            if policy_update_attempted:
                try:
                    ray.get(
                        rollout_manager.fail_policy_update.remote(
                            next_policy_version,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist Polar policy-update failure; admission remains closed"
                    )
            raise

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            eval_policy_version = (
                next_policy_version if durable_polar_boundary else rollout_id
            )
            ray.get(rollout_manager.eval.remote(eval_policy_version))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
