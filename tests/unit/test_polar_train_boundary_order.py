from __future__ import annotations

from types import SimpleNamespace

import train as train_module


class _RemoteMethod:
    def __init__(self, calls, name, result=None):
        self.calls = calls
        self.name = name
        self.result = result

    def remote(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))
        return self.result


def test_initial_polar_bootstrap_precedes_first_weight_mutation(monkeypatch) -> None:
    calls = []
    manager = SimpleNamespace(
        get_metrics_router_addr=_RemoteMethod(calls, "metrics", None),
        prepare_initial_policy=_RemoteMethod(calls, "prepare_initial"),
        onload_weights=_RemoteMethod(calls, "onload_weights"),
        onload_kv=_RemoteMethod(calls, "onload_kv"),
        finish_initial_policy=_RemoteMethod(calls, "finish_initial"),
        fail_policy_update=_RemoteMethod(calls, "fail_policy_update"),
        dispose=_RemoteMethod(calls, "dispose"),
    )

    class Actor:
        def prepare_memory_handoff(self):
            calls.append(("prepare_memory", (), {}))

        def update_weights(self):
            calls.append(("update_weights", (), {}))

        def finish_memory_handoff(self):
            calls.append(("finish_memory", (), {}))

    monkeypatch.setattr(train_module.ray, "get", lambda value: value)
    monkeypatch.setattr(
        train_module,
        "create_placement_groups",
        lambda args: {"rollout": object()},
    )
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda args, pg: (manager, 1),
    )
    monkeypatch.setattr(
        train_module,
        "create_training_models",
        lambda args, pgs, rollout_manager: (Actor(), None),
    )
    for name in (
        "configure_logger",
        "init_tracking",
        "update_tracking_open_metrics",
        "finish_tracking",
    ):
        monkeypatch.setattr(train_module, name, lambda *args, **kwargs: None)

    args = SimpleNamespace(
        polar_policy_transition_enabled=True,
        offload_rollout=True,
        offload_train=True,
        start_rollout_id=0,
        check_weight_update_equal=False,
        num_rollout=0,
        eval_interval=None,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
    )
    train_module.train(args)

    names = [entry[0] for entry in calls]
    assert names.index("prepare_initial") < names.index("update_weights")
    assert names.index("update_weights") < names.index("finish_initial")
    assert names.index("onload_kv") < names.index("finish_initial")
    assert names[-1] == "dispose"
    assert "fail_policy_update" not in names
