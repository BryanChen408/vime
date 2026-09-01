import ray

from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from vime.utils.misc import should_run_periodic_action


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    if bool(getattr(args, "polar_policy_transition_enabled", False)):
        raise RuntimeError(
            "--polar-policy-transition-enabled is supported by train.py only; "
            "train_async.py overlaps generation and weight mutation by design"
        )
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

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
    # Advance the rollout manager's policy version so custom rollout functions
    # that track off-policy staleness (e.g. vime_bridge) see the initial weights.
    # No-op for rollout functions without an update_policy_version hook.
    ray.get(rollout_manager.update_policy_version.remote(args.start_rollout_id))

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
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

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # Overlap mode (polar_allow_weight_update_overlap): generation keeps
            # running during the weight update, so prepare_policy_update pauses the
            # polar gateway and drains/aborts in-flight sessions — those come back
            # with trajectory.status=ERROR and are excluded from training so the
            # weight boundary can't pollute the trainset. Non-overlap (default):
            # sync the pending generation first so no session spans the update.
            allow_weight_update_overlap = bool(getattr(args, "polar_allow_weight_update_overlap", False))
            next_policy_version = rollout_id + 1
            if not allow_weight_update_overlap:
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            if allow_weight_update_overlap:
                # runs in the rollout_manager actor, where the polar async worker lives
                ray.get(rollout_manager.prepare_policy_update.remote(next_policy_version))
            try:
                actor_model.update_weights()
                # Advance policy version so off-policy staleness tracking (vime_bridge)
                # stays live; a frozen version drops every group as stale and hangs.
                version_ref = rollout_manager.update_policy_version.remote(next_policy_version)
                if not allow_weight_update_overlap:
                    ray.get(version_ref)
            finally:
                if allow_weight_update_overlap:
                    ray.get(rollout_manager.finish_policy_update.remote(next_policy_version))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
