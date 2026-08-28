"""Regression matrix for :mod:`vime.ray.engine_roles`.

The colocation semantic has two distinct halves and each one has already been
got wrong in production:

* *is this engine colocated?* — decided three different ways (by node, by total
  actor GPU count, by shared GPU count), which agreed only on the two topologies
  that had been run. See the module docstring of ``engine_roles``.
* *which actor rank owns each of the engine's cards?* — conflated with the
  engine's **rollout slot**, which is a different index space. On 20260828 a
  single-node partially-shared layout (actor ``0-15``, rollout ``4-15``) handed
  every engine the IPC handles of a card pair four positions away and died with
  ``IPC handle not found for GPU UUID ...``.

Both halves are cheap to assert off-hardware, so they are asserted here for
every topology rather than only for the one that happens to be running.
"""

from __future__ import annotations

import dataclasses
import types

import pytest

from vime.ray.engine_roles import (
    EngineRoleError,
    any_colocated,
    colocated_prefix_count,
    resolve_engine_roles,
)


@dataclasses.dataclass
class _Item:
    node: str
    devices: list[int]


def _layout_args(actor, rollout, *, per_engine=2, node="n0"):
    """Args namespace carrying a ``--resource-layout`` spec."""
    return types.SimpleNamespace(
        rollout_num_gpus_per_engine=per_engine,
        resource_layout_spec=types.SimpleNamespace(
            actor=[_Item(node, list(seg)) for seg in actor],
            rollout=[_Item(node, list(seg)) for seg in rollout],
            rollout_has_share=True,
        ),
    )


def _colocate_args(rollout_num_gpus, *, per_engine=2):
    """Args namespace for the ``--colocate`` path (no layout)."""
    return types.SimpleNamespace(
        rollout_num_gpus_per_engine=per_engine,
        rollout_num_gpus=rollout_num_gpus,
        colocate=True,
        resource_layout_spec=None,
    )


def _positional_args(actor_gpus, rollout_num_gpus, *, per_engine=2):
    """Args namespace for the positional (disaggregated) path."""
    return types.SimpleNamespace(
        rollout_num_gpus_per_engine=per_engine,
        rollout_num_gpus=rollout_num_gpus,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=actor_gpus,
        resource_layout_spec=None,
    )


# --- the five topologies, as (name, args, expected colocated flags) ----------

T2_HOMOGENEOUS = _layout_args([range(16)], [range(16)])
T3_HETERO_CROSS_NODE = types.SimpleNamespace(
    rollout_num_gpus_per_engine=2,
    resource_layout_spec=types.SimpleNamespace(
        actor=[_Item("n56", list(range(16)))],
        rollout=[_Item("n56", list(range(16))), _Item("n64", list(range(8)))],
        rollout_has_share=True,
    ),
)
T4_HETERO_SINGLE_NODE = _layout_args([range(4, 12)], [range(4, 8), range(8, 12), range(12, 16)])
T5_PARTIAL_SHARE = _layout_args([range(16)], [range(4, 8), range(8, 16)])


def test_t5_engine_cards_map_to_their_own_actor_ranks():
    """The 20260828 crash: rollout slot 0 is actor rank 4, not actor rank 0.

    actor ``0-15`` with rollout ``4-15`` means the rollout sequence starts four
    cards into the actor's. Reading the slot as a rank shifts every engine's IPC
    handles by four cards; the receiver keys on its own device UUID and finds
    the neighbour's pair instead.
    """
    roles = resolve_engine_roles(T5_PARTIAL_SHARE)

    assert [r.gpu_slot for r in roles] == [0, 2, 4, 6, 8, 10]
    assert [r.actor_ranks for r in roles] == [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    # Every engine is colocated here — the bug was never about *whether*.
    assert all(r.colocated for r in roles)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("T2 homogeneous", T2_HOMOGENEOUS),
        ("T3 heterogeneous cross-node", T3_HETERO_CROSS_NODE),
        ("T4 heterogeneous single-node", T4_HETERO_SINGLE_NODE),
        ("--colocate", _colocate_args(8)),
    ],
)
def test_slot_equals_rank_where_it_already_worked(name, args):
    """Guards the fix against regressing the topologies that were already fine.

    Each of these starts its rollout segment at the actor's first card, so the
    old slot-as-rank arithmetic was accidentally correct. The new mapping must
    reproduce it exactly, otherwise the fix trades one broken topology for four.
    """
    for role in resolve_engine_roles(args):
        if role.colocated:
            expected = tuple(range(role.gpu_slot, role.gpu_slot + role.num_gpus))
            assert role.actor_ranks == expected, f"{name}: slot {role.gpu_slot}"


def test_dedicated_engines_have_no_actor_ranks():
    roles = resolve_engine_roles(T4_HETERO_SINGLE_NODE)
    colocated, dedicated = roles[:4], roles[4:]
    assert all(r.colocated and r.actor_ranks for r in colocated)
    # A dedicated engine shares no card with the actor, so no rank owns it and
    # there is nothing for an IPC gather group to be built from.
    assert all(not r.colocated and r.actor_ranks == () for r in dedicated)


def test_positional_path_has_no_colocated_engines():
    roles = resolve_engine_roles(_positional_args(actor_gpus=8, rollout_num_gpus=4))
    assert not any_colocated(roles)
    assert all(r.actor_ranks == () for r in roles)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (T2_HOMOGENEOUS, 8),
        (T3_HETERO_CROSS_NODE, 8),
        (T4_HETERO_SINGLE_NODE, 4),
        (T5_PARTIAL_SHARE, 6),
        (_colocate_args(8), 4),
        (_positional_args(actor_gpus=8, rollout_num_gpus=4), 0),
    ],
)
def test_colocated_prefix_count(args, expected):
    assert colocated_prefix_count(resolve_engine_roles(args)) == expected


def test_interleaved_colocation_is_rejected():
    """Callers size IPC groups from a bare count, which cannot express a gap."""
    interleaved = _layout_args([range(4)], [range(2), range(8, 10), range(2, 4)])
    with pytest.raises(EngineRoleError, match="must form a prefix"):
        colocated_prefix_count(resolve_engine_roles(interleaved))


def test_rollout_cards_must_divide_into_whole_engines():
    with pytest.raises(EngineRoleError, match="not a multiple of"):
        resolve_engine_roles(_layout_args([range(8)], [range(3)]))


def test_actor_ranks_are_ordered_by_placement_not_sorted():
    """Rank order must follow the engine's own card order.

    An engine's cards are what ``_send_to_colocated_engine`` gathers over, and
    the gather source is the group's lowest rank. Re-sorting here would be
    invisible for contiguous segments and wrong for any layout that lists a
    card out of order.
    """
    args = _layout_args([[3, 1, 0, 2]], [[0, 2]], per_engine=2)
    (role,) = resolve_engine_roles(args)
    # Card 0 is at index 2 of the actor list, card 2 is at index 3.
    assert role.actor_ranks == (2, 3)
