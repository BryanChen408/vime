import math

import _cp_dist_helpers
import pytest
import torch

from vime.backends.megatron_utils.cp_utils import reduce_train_step_metrics, slice_log_prob_with_cp
from vime.utils.rollout_prob_metrics import STAT_KEYS, probability_diff_metrics, probability_diff_stats


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("per_token_loss", [False, True])
def test_probability_metrics_match_verl_after_cp_and_microbatch_reduction(monkeypatch, cp_size, per_token_loss):
    # Different lengths, masks and differences expose mean-of-means/std/max bugs.
    train = [torch.tensor([.1, .7, .2, .5, .9]).log(), torch.tensor([.4, .3, .8]).log()]
    rollout = [torch.tensor([.4, .1, .2, .7, .6]).log(), torch.tensor([.2, .9, .6]).log()]
    masks = [torch.tensor([1, 0, 1, 1, 1]), torch.tensor([1, 1, 0])]
    expected = torch.cat([(a.exp()-b.exp()).abs()[m.bool()] for a, b, m in zip(train, rollout, masks)])
    contributions = []
    for rank in range(cp_size):
        _cp_dist_helpers.stub_megatron_in_worker(cp_size, rank)
        for a, b, m in zip(train, rollout, masks):
            a, b, m = [slice_log_prob_with_cp(x, len(x)+3, len(x)) for x in (a, b, m)]
            stats = probability_diff_stats(a, b, m)
            contributions.append({"keys": list(STAT_KEYS), "values": torch.stack([a.new_tensor(1), *stats.values()])})
    _cp_dist_helpers.stub_megatron_in_worker(1, 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    result = reduce_train_step_metrics(contributions, calculate_per_token_loss=per_token_loss,
                                      step_global_batch_size=2, cp_size=cp_size, dp_with_cp_group=None)
    assert result["rollout_probs_diff_count"] == expected.numel()
    for name, value in (("mean", expected.mean()), ("std", expected.std()), ("max", expected.max())):
        assert result[f"rollout_probs_diff_{name}"] == pytest.approx(value.item(), abs=1e-6)


def test_empty_singleton_and_masked_invalid_values():
    result = probability_diff_metrics(0, 0, 0, 0)
    assert math.isnan(result["rollout_probs_diff_mean"])
    assert math.isnan(probability_diff_metrics(1, .2, .04, .2)["rollout_probs_diff_std"])
    stats = probability_diff_stats(torch.tensor([float("nan"), -.2]), torch.tensor([0., -.1]), torch.tensor([0, 1]))
    assert stats[STAT_KEYS[0]] == 1
    with pytest.raises(ValueError, match="non-finite"):
        probability_diff_stats(torch.tensor([float("nan")]), torch.tensor([0.]), torch.tensor([1]))


def _distributed_probability_worker(rank, port):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        probabilities = [torch.tensor([.1, .7])] if rank == 0 else [torch.tensor([.2]), torch.tensor([.8, .4])]
        batches = []
        for probs in probabilities:
            stats = probability_diff_stats(probs.log(), torch.full_like(probs, .2).log(), torch.ones_like(probs))
            batches.append({"keys": list(STAT_KEYS), "values": torch.stack([probs.new_tensor(1), *stats.values()])})
        result = reduce_train_step_metrics(batches, calculate_per_token_loss=False,
                                          step_global_batch_size=2, cp_size=1, dp_with_cp_group=None)
        expected = torch.tensor([.1, .5, 0., .6, .2])
        assert result["rollout_probs_diff_count"] == 5
        assert abs(result["rollout_probs_diff_max"] - expected.max().item()) < 1e-6
        assert abs(result["rollout_probs_diff_mean"] - expected.mean().item()) < 1e-6
        assert abs(result["rollout_probs_diff_std"] - expected.std().item()) < 1e-6
    finally:
        dist.destroy_process_group()


def test_real_distributed_reduction_with_uneven_microbatches():
    torch.multiprocessing.spawn(_distributed_probability_worker, args=(_cp_dist_helpers.free_port(),),
                                nprocs=2, join=True)
