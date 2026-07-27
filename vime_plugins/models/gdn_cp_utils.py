"""Context-parallel helpers for the gated delta net.

Two groups:
  A. Pure-torch THD reordering for packed sequences under CP. Transformer Engine's
     ``tex.thd_get_partitioned_indices`` is not available on Ascend, so the index math is
     reimplemented here.
  B. High-level CP helpers (``tensor_a2a_cp2hp`` / ``tensor_a2a_hp2cp`` /
     ``get_parameter_local_cp``). Newer Megatron ships these in
     ``megatron.core.ssm.gated_delta_net``; the version we build against does not, so they
     are reimplemented on top of the primitives it does have. Signatures match the upstream
     ones so this module can be dropped once Megatron is new enough.

Group A mirrors ``get_batch_on_this_cp_rank`` / ``get_thd_batch_on_this_cp_rank``: each
sequence is cut into ``2 * cp_size`` chunks and rank ``r`` owns chunk ``r`` and chunk
``2 * cp_size - 1 - r``. Only used for packed sequences with CP > 1.
"""
from typing import List, Optional

import torch

# Private Megatron symbols: no public equivalent exists in the version we build against.
# Revisit on every Megatron upgrade.
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _undo_attention_load_balancing,
    _redo_attention_load_balancing,
)


# ---------------------------------------------------------------------------
# A. Packed (THD) index reordering under CP
# ---------------------------------------------------------------------------
def _cu_seqlens_to_list(cu_seqlens) -> List[int]:
    """Materialise cu_seqlens on the host. This synchronises, so do it once per call site."""
    if isinstance(cu_seqlens, torch.Tensor):
        return [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    return [int(x) for x in cu_seqlens]


def _partitioned_indices_per_rank(cu: List[int], cp_size: int) -> List[torch.Tensor]:
    """Token indices owned by each CP rank, for an already-materialised cu_seqlens."""
    two_cp = 2 * cp_size
    per_rank = [[] for _ in range(cp_size)]
    for i in range(len(cu) - 1):
        start, end = cu[i], cu[i + 1]
        length = end - start
        if length <= 0:
            continue
        assert length % two_cp == 0, (
            f"sequence {i} has length {length}, which is not divisible by 2*cp={two_cp}; "
            "packed CP requires the padded cu_seqlens"
        )
        chunk = length // two_cp
        for rank in range(cp_size):
            early = start + rank * chunk
            late = start + (two_cp - 1 - rank) * chunk
            per_rank[rank].append(torch.arange(early, early + chunk, dtype=torch.long))
            per_rank[rank].append(torch.arange(late, late + chunk, dtype=torch.long))
    return [torch.cat(idx) if idx else torch.empty(0, dtype=torch.long) for idx in per_rank]


def thd_get_partitioned_indices_torch(cu_seqlens, total_tokens: int, cp_size: int, cp_rank: int):
    """Indices, in the packed buffer, of the tokens owned by `cp_rank`.

    `cu_seqlens` must be the padded variant: every sequence length has to be divisible by
    ``2 * cp_size``, which the data path guarantees by padding each sequence before slicing.
    """
    return _partitioned_indices_per_rank(_cu_seqlens_to_list(cu_seqlens), cp_size)[cp_rank]


def undo_attention_load_balancing_thd(input_, cu_seqlens, cp_size):
    """Zigzag load-balanced layout -> natural order, per sequence.

    `input_` is the full [total_tokens, ...] buffer with the per-rank slices concatenated
    in rank order.
    """
    if cp_size == 1:
        return input_
    total = input_.size(0)
    assert total % cp_size == 0
    per_rank = total // cp_size
    indices = _partitioned_indices_per_rank(_cu_seqlens_to_list(cu_seqlens), cp_size)
    output = torch.empty_like(input_)
    for r in range(cp_size):
        output[indices[r].to(input_.device)] = input_[r * per_rank : (r + 1) * per_rank]
    return output


def redo_attention_load_balancing_thd(input_, cu_seqlens, cp_size):
    """Inverse of :func:`undo_attention_load_balancing_thd`."""
    if cp_size == 1:
        return input_
    total = input_.size(0)
    assert total % cp_size == 0
    per_rank = total // cp_size
    indices = _partitioned_indices_per_rank(_cu_seqlens_to_list(cu_seqlens), cp_size)
    index = torch.empty(total, dtype=torch.long, device=input_.device)
    for r in range(cp_size):
        index[r * per_rank : (r + 1) * per_rank] = indices[r].to(input_.device)
    return input_.index_select(0, index)


# ---------------------------------------------------------------------------
# B. High-level CP helpers
#    Signatures mirror the same-named functions in newer Megatron's
#    megatron.core.ssm.gated_delta_net:
#      get_parameter_local_cp(param, dim, cp_group, split_sections=None)
#      tensor_a2a_cp2hp(tensor, seq_dim, head_dim, cp_group, split_sections=None,
#                       undo_attention_load_balancing=True)
#      tensor_a2a_hp2cp(tensor, seq_dim, head_dim, cp_group, split_sections=None,
#                       redo_attention_load_balancing=True)
# ============================================================================
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: "torch.distributed.ProcessGroup",
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """This CP rank's slice of `param`.

    A no-op when cp_size == 1. With `split_sections`, each segment is sliced separately
    along `dim` and the results are concatenated back, so segment boundaries stay aligned.
    """
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    if cp_size == 1:
        return param

    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: "torch.distributed.ProcessGroup",
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """Sequence-parallel -> head-parallel all-to-all.

    Expects (seq_len, batch, hidden); only seq_dim=0 and head_dim in {-1, 2} are supported.
    With `split_sections` each segment is exchanged on its own so segment boundaries stay
    aligned, and the zigzag load balancing is undone once over the concatenated result.
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert tensor.dim() == 3, f"tensor_a2a_cp2hp only supports 3-d input tensor, but got {tensor.dim()=}"

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x, seq_dim=seq_dim, head_dim=head_dim, cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: "torch.distributed.ProcessGroup",
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """Head-parallel -> sequence-parallel all-to-all, the inverse of :func:`tensor_a2a_cp2hp`.

    The zigzag load balancing is redone once up front, then each segment is exchanged
    separately when `split_sections` is given.
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert tensor.dim() == 3, f"tensor_a2a_hp2cp only supports 3-d input tensor, but got {tensor.dim()=}"

    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x, seq_dim=seq_dim, head_dim=head_dim, cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor
