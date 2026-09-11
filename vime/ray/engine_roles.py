"""Single source of truth for «is this rollout engine colocated with the trainer?».

Why this module exists
----------------------
That one fact used to be decided independently in three places, each with a
different proxy for it:

===========================================  ==========================================
consumer                                     proxy it used
===========================================  ==========================================
engine launch weight-transfer backend        the *node* the engine runs on
weight sync channel (IPC vs HCCL)            total actor GPU count
which engines sleep during training          the shared (``share: actor``) GPU count
===========================================  ==========================================

Each proxy is only equivalent to colocation under an assumption that happens to
hold for the two topologies that had been run in production:

* *node* assumes the dedicated segment lives on a different host — false as soon
  as a heterogeneous layout fits on one machine, which made the engine launch
  with the IPC backend while the trainer addressed it over HCCL
  (``NPUIPCWeightTransferInitInfo ... unexpected keyword argument
  'master_address'``).
* *total actor GPU count* assumes the shared segment covers every actor card —
  false for a partially shared layout, where a remote engine's slot still falls
  below the actor count and would be handed IPC.
* *shared GPU count* is only defined on the layout path; it is ``None`` under
  ``--colocate`` and under the positional (disaggregated) path.

Colocation is a set relation — every ``(node, device)`` an engine occupies also
belongs to the actor — and no single scalar can express it losslessly. This
module computes that relation once; every consumer reads the result.

See ``docs/design/colocate_topology_robustness_plan.md``.
"""

from __future__ import annotations

import dataclasses
from typing import Any


# A physical card: (node identifier, device index on that node).
Placement = tuple[str, int]

# Node identifier used when a path carries no real topology (``--colocate`` and
# the positional path describe placement by GPU *count* only). It never escapes
# this module's own bookkeeping, so it cannot collide with a real hostname.
_IMPLICIT_NODE = "<implicit>"


@dataclasses.dataclass(frozen=True)
class EngineRole:
    """What one rollout engine is, from the trainer's point of view."""

    index: int
    """Global engine index, matching the order engines are created in."""

    gpu_slot: int
    """Start offset of this engine inside the rollout GPU sequence."""

    placement: tuple[Placement, ...]
    """The physical cards this engine occupies."""

    colocated: bool
    """True when every card of this engine is also an actor card."""

    actor_ranks: tuple[int, ...] = ()
    """Actor ranks sharing this engine's cards, in ``placement`` order.

    Empty for a dedicated engine (its cards belong to no actor rank). The rank
    of a card is its index in the actor's expanded card list, which is the order
    the launcher assigns ranks in.

    This is *not* derivable from :attr:`gpu_slot`. A slot indexes the **rollout**
    GPU sequence; a rank indexes the **actor's**. They coincide only when the
    rollout segment starts at the actor's first card — true for the two
    production topologies, false for any partially shared layout (e.g. actor
    ``0-15`` with rollout ``4-15``, where slot 0 is actor rank 4). Comparing a
    slot against ``dist.get_rank()`` there hands each engine the IPC handles of
    a card pair four positions away, and the receiver — which correctly keys on
    its own device UUID — reports ``IPC handle not found for GPU UUID ...``.
    """

    @property
    def num_gpus(self) -> int:
        return len(self.placement)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(node for node, _ in self.placement)


class EngineRoleError(RuntimeError):
    """Raised when a topology violates an invariant the callers depend on."""


def _expand(items: Any) -> list[Placement]:
    return [(item.node, device) for item in items for device in item.devices]


def resolve_engine_roles(args: Any) -> tuple[EngineRole, ...]:
    """Classify every rollout engine. The only colocation judgement in the tree.

    Handles all three placement paths:

    * ``--resource-layout``: expand the layout and test set containment. This is
      the only path where the answer can vary per engine.
    * ``--colocate``: rollout is defined to reuse the actor's cards, so every
      engine is colocated.
    * positional (disaggregated): rollout cards start after the actor's, so no
      engine is colocated.

    Colocated engines must form a prefix — see :func:`colocated_prefix_count`.
    """
    per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    if per_engine <= 0:
        raise EngineRoleError(f"rollout_num_gpus_per_engine must be positive, got {per_engine}")

    spec = getattr(args, "resource_layout_spec", None)
    if spec is not None:
        actor_card_list = _expand(spec.actor)
        actor_cards = set(actor_card_list)
        rollout_cards = _expand(spec.rollout)
    else:
        rollout_num_gpus = int(getattr(args, "rollout_num_gpus", 0) or 0)
        if getattr(args, "colocate", False):
            # Rollout reuses the actor's cards one-for-one, so the two sequences
            # are the same cards and containment holds for every engine.
            rollout_cards = [(_IMPLICIT_NODE, i) for i in range(rollout_num_gpus)]
            actor_card_list = list(rollout_cards)
            actor_cards = set(rollout_cards)
        else:
            # Positional: the actor owns [0, actor_gpus) and rollout starts after
            # it, so the two sets are disjoint by construction.
            actor_gpus = int(getattr(args, "actor_num_nodes", 1) or 1) * int(
                getattr(args, "actor_num_gpus_per_node", 0) or 0
            )
            actor_card_list = [(_IMPLICIT_NODE, i) for i in range(actor_gpus)]
            actor_cards = set(actor_card_list)
            rollout_cards = [(_IMPLICIT_NODE, actor_gpus + i) for i in range(rollout_num_gpus)]

    # Rank of a card = its index in the actor's card sequence, which is the order
    # the launcher assigns ranks in.
    actor_rank_of = {card: rank for rank, card in enumerate(actor_card_list)}

    if len(rollout_cards) % per_engine:
        raise EngineRoleError(
            f"rollout has {len(rollout_cards)} cards, which is not a multiple of "
            f"rollout_num_gpus_per_engine={per_engine}"
        )

    roles = []
    for index in range(len(rollout_cards) // per_engine):
        slot = index * per_engine
        placement = tuple(rollout_cards[slot : slot + per_engine])
        # Partial overlap would mean an engine straddling the boundary, which no
        # caller can act on: it can be neither handed an IPC handle nor put to
        # sleep independently of the trainer.
        colocated = all(card in actor_cards for card in placement)
        roles.append(
            EngineRole(
                index=index,
                gpu_slot=slot,
                placement=placement,
                colocated=colocated,
                actor_ranks=tuple(actor_rank_of[card] for card in placement) if colocated else (),
            )
        )
    return tuple(roles)


def colocated_prefix_count(roles: tuple[EngineRole, ...]) -> int:
    """Number of leading colocated engines; errors if colocation is interleaved.

    Several callers pass this count around instead of a list of engines — the
    IPC weight send sizes its Gloo gather group from it, for one. A bare count
    only identifies *which* engines when the colocated ones come first, which is
    what the layout validator enforces by requiring ``share: actor`` entries
    before dedicated ones.

    That requirement used to be implicit, spread across arithmetic in each
    caller. Asserting it here turns it into a contract: if the layout rule is
    ever relaxed, this raises immediately instead of silently mismatching the
    trainer's channel against the engine's backend.
    """
    count = 0
    for role in roles:
        if not role.colocated:
            break
        count += 1
    interleaved = [role.index for role in roles[count:] if role.colocated]
    if interleaved:
        raise EngineRoleError(
            f"colocated engines must form a prefix, but engines {interleaved} are colocated "
            f"after non-colocated engine {count}. Callers size IPC groups from a count, which "
            "cannot express an interleaved layout. Order shared (share: actor) rollout entries "
            "before dedicated ones."
        )
    return count


def any_colocated(roles: tuple[EngineRole, ...]) -> bool:
    return any(role.colocated for role in roles)
