"""Explicit node/card resource layout parsing for Ray placement."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class NodeDevices:
    node: str
    devices: tuple[int, ...]
    # share="actor": 与 actor 角色共享同一批物理卡 —— 不为这些设备新建 bundle,
    # rollout 槽位直接映射到 actor 的 bundle 上(训推共卡)。目前仅支持 share="actor"
    # 且只用于 rollout 角色。共卡段必须写在 rollout 列表最前(槽位 0..N-1),
    # 使 needs_offload 正确命中(见 rollout.py _compute_rollout_offset)。
    share: str | None = None


@dataclasses.dataclass(frozen=True)
class ResourceLayout:
    actor: tuple[NodeDevices, ...] = ()
    # critic 独立卡位:默认空 () → critic 复用 actor bundle(共卡老路径,行为不变);
    # 仅当 yaml 显式给 roles.critic 才建独立 placement(分卡 / 跨节点),让每个节点只放
    # 一个模型的优化器,避开 host 内存天花板。见 placement_group._create_placement_groups_from_layout。
    critic: tuple[NodeDevices, ...] = ()
    rollout: tuple[NodeDevices, ...] = ()
    polar_reserved: tuple[NodeDevices, ...] = ()
    rollout_num_gpus_per_engine: int | None = None
    vllm_dp_size: int | None = None

    @property
    def actor_num_gpus(self) -> int:
        return sum(len(item.devices) for item in self.actor)

    @property
    def rollout_num_gpus(self) -> int:
        return sum(len(item.devices) for item in self.rollout)

    @property
    def rollout_shared_num_gpus(self) -> int:
        """共卡(share)rollout 设备数 —— 复用 actor bundle,不占额外 Ray 资源。"""
        return sum(len(item.devices) for item in self.rollout if item.share)

    @property
    def rollout_dedicated_num_gpus(self) -> int:
        """专用(非共卡)rollout 设备数 —— 需要独立 bundle 的部分。"""
        return sum(len(item.devices) for item in self.rollout if not item.share)

    @property
    def rollout_has_share(self) -> bool:
        return any(item.share for item in self.rollout)

    @property
    def rollout_num_gpus_per_node(self) -> int:
        if not self.rollout:
            return 0
        if self.rollout_has_share:
            # 混合(共卡+专用)布局:共卡段约定写在最前(槽位 0..N-1 与 actor 同卡),
            # 端口分配器的"每节点引擎数"桶宽必须等于 actor 所在节点的 rollout 卡数
            # (含共卡),否则跨节点的引擎会被分到错误的地址桶里。
            counts_by_node = _device_counts_by_node(self.rollout)
            for item in self.actor:
                if item.node in counts_by_node:
                    return counts_by_node[item.node]
            raise ValueError(
                "resource layout with share: actor's node must also appear in rollout entries "
                f"(actor nodes: {sorted({i.node for i in self.actor})}, "
                f"rollout nodes: {sorted(counts_by_node)})"
            )
        counts = set(_device_counts_by_node(self.rollout).values())
        if len(counts) != 1:
            raise ValueError(
                "--resource-layout requires equal rollout device counts per node for the current vLLM path; "
                f"got {sorted(counts)}"
            )
        return counts.pop()

    @property
    def ray_num_gpus(self) -> int:
        # critic 独立卡位时须计入(空 critic → critic_num_gpus=0,与共卡老路径一致);
        # rollout 的 share 段复用 actor bundle,不计入(否则 Ray 资源不可满足)。
        return self.actor_num_gpus + self.critic_num_gpus + self.rollout_dedicated_num_gpus

    @property
    def actor_num_nodes(self) -> int:
        return len(_device_counts_by_node(self.actor))

    @property
    def actor_num_gpus_per_node(self) -> int:
        if not self.actor:
            return 0
        counts = set(_device_counts_by_node(self.actor).values())
        if len(counts) != 1:
            raise ValueError(
                "--resource-layout requires equal actor device counts per node for the current Megatron path; "
                f"got {sorted(counts)}"
            )
        return counts.pop()

    @property
    def critic_num_gpus(self) -> int:
        return sum(len(item.devices) for item in self.critic)

    @property
    def critic_num_nodes(self) -> int:
        return len(_device_counts_by_node(self.critic))

    @property
    def critic_num_gpus_per_node(self) -> int:
        if not self.critic:
            return 0
        counts = set(_device_counts_by_node(self.critic).values())
        if len(counts) != 1:
            raise ValueError(
                "--resource-layout requires equal critic device counts per node for the current Megatron path; "
                f"got {sorted(counts)}"
            )
        return counts.pop()


def parse_devices(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        devices = (value,)
    elif isinstance(value, str):
        devices = _parse_device_string(value)
    elif isinstance(value, (list, tuple)):
        devices = tuple(_parse_device_int(item) for item in value)
    else:
        raise TypeError(f"devices must be an int, string, or list of ints; got {type(value).__name__}")

    seen: set[int] = set()
    out: list[int] = []
    for device in devices:
        if device < 0:
            raise ValueError(f"device id must be >= 0, got {device}")
        if device in seen:
            raise ValueError(f"duplicate device id {device}")
        seen.add(device)
        out.append(device)
    return tuple(out)


def _device_counts_by_node(entries: tuple[NodeDevices, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in entries:
        counts[item.node] = counts.get(item.node, 0) + len(item.devices)
    return counts


def load_resource_layout(path: str | Path) -> ResourceLayout:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return resource_layout_from_dict(data)


def resource_layout_from_dict(data: dict[str, Any]) -> ResourceLayout:
    roles = data.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("resource layout must contain a 'roles' mapping")

    layout = ResourceLayout(
        actor=_parse_role(roles, "actor"),
        critic=_parse_role(roles, "critic", required=False),
        rollout=_parse_role(roles, "rollout"),
        polar_reserved=_parse_role(roles, "polar_reserved", required=False),
        rollout_num_gpus_per_engine=_optional_positive_int(data, ("rollout", "num_gpus_per_engine")),
        vllm_dp_size=_optional_positive_int(data, ("rollout", "vllm_dp_size")),
    )
    _validate_layout(layout)
    return layout


def select_role_bundles(
    bundle_infos: list[tuple[int, str, Any]],
    role: tuple[NodeDevices, ...],
    *,
    role_name: str,
) -> tuple[list[int], list[int]]:
    """Select placement-group bundle indices for a role.

    Args:
        bundle_infos: ``(bundle_index, node_ip, device_id)`` triples from Ray.
        role: ordered node/device role spec.
        role_name: name used in error messages.

    Returns:
        ``(bundle_indices, device_ids)`` in role/YAML order.
    """

    by_node_device: dict[tuple[str, int], int] = {}
    for bundle_index, node, device in bundle_infos:
        key = (str(node), int(device))
        if key in by_node_device:
            raise ValueError(f"duplicate Ray bundle for node/device {key}: {by_node_device[key]} and {bundle_index}")
        by_node_device[key] = int(bundle_index)

    bundle_indices: list[int] = []
    device_ids: list[int] = []
    missing: list[str] = []
    for item in role:
        for device in item.devices:
            key = (item.node, device)
            bundle_index = by_node_device.get(key)
            if bundle_index is None:
                missing.append(f"{item.node}:{device}")
                continue
            bundle_indices.append(bundle_index)
            device_ids.append(device)

    if missing:
        available = ", ".join(
            f"{node}:{device}" for _, node, device in sorted(bundle_infos, key=lambda x: (x[1], int(x[2])))
        )
        raise ValueError(
            f"Ray cluster is missing requested {role_name} devices: {', '.join(missing)}. "
            f"Available bundles: {available}"
        )

    return bundle_indices, device_ids


def _parse_device_string(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        raise ValueError("devices string must not be empty")

    devices: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"empty device item in {value!r}")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = _parse_device_int(start_text.strip())
            end = _parse_device_int(end_text.strip())
            if end < start:
                raise ValueError(f"invalid descending device range {part!r}")
            devices.extend(range(start, end + 1))
        else:
            devices.append(_parse_device_int(part))
    return tuple(devices)


def _parse_device_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("device id must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"device id must be an integer, got {value!r}") from exc


def _parse_role(roles: dict[str, Any], name: str, *, required: bool = True) -> tuple[NodeDevices, ...]:
    raw_entries = roles.get(name)
    if raw_entries is None:
        if required:
            raise ValueError(f"resource layout roles.{name} is required")
        return ()
    if not isinstance(raw_entries, list):
        raise TypeError(f"resource layout roles.{name} must be a list")

    entries: list[NodeDevices] = []
    for idx, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"resource layout roles.{name}[{idx}] must be a mapping")
        node = str(raw_entry.get("node") or "").strip()
        if not node:
            raise ValueError(f"resource layout roles.{name}[{idx}].node is required")
        if "devices" not in raw_entry:
            raise ValueError(f"resource layout roles.{name}[{idx}].devices is required")
        share = raw_entry.get("share")
        if share is not None:
            share = str(share).strip()
            if share != "actor":
                raise ValueError(
                    f"resource layout roles.{name}[{idx}].share only supports \"actor\" for now, got {share!r}"
                )
            if name != "rollout":
                raise ValueError(
                    f"resource layout roles.{name}[{idx}].share is only allowed on the rollout role"
                )
        entries.append(NodeDevices(node=node, devices=parse_devices(raw_entry["devices"]), share=share))
    return tuple(entries)


def _optional_positive_int(data: dict[str, Any], path: tuple[str, str]) -> int | None:
    section = data.get(path[0])
    if section is None:
        return None
    if not isinstance(section, dict):
        raise TypeError(f"resource layout {path[0]} must be a mapping")
    value = section.get(path[1])
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"resource layout {'.'.join(path)} must be an integer, got bool")
    out = int(value)
    if out <= 0:
        raise ValueError(f"resource layout {'.'.join(path)} must be > 0, got {out}")
    return out


def _validate_layout(layout: ResourceLayout) -> None:
    if layout.actor_num_gpus <= 0:
        raise ValueError("resource layout roles.actor must request at least one device")
    if layout.rollout_num_gpus <= 0:
        raise ValueError("resource layout roles.rollout must request at least one device")
    # Force current Megatron-compatible actor shape during parsing.
    _ = layout.actor_num_gpus_per_node
    # Force current vLLM-compatible rollout shape during parsing.
    _ = layout.rollout_num_gpus_per_node

    used: dict[tuple[str, int], str] = {}
    actor_devices: set[tuple[str, int]] = {
        (item.node, device) for item in layout.actor for device in item.devices
    }
    for role_name, entries in (
        ("actor", layout.actor),
        ("critic", layout.critic),
        ("rollout", layout.rollout),
        ("polar_reserved", layout.polar_reserved),
    ):
        for item in entries:
            for device in item.devices:
                key = (item.node, device)
                prior = used.get(key)
                if prior is not None:
                    # rollout share="actor" 的共卡设备允许与 actor 重叠(且必须 ⊆ actor),
                    # 其它一切角色间重叠照旧禁止。
                    if item.share == "actor" and role_name == "rollout" and prior == "actor":
                        continue
                    raise ValueError(
                        f"resource layout device overlap: {item.node}:{device} is in both {prior} and {role_name}"
                    )
                used[key] = role_name
    # share 语义校验:共卡设备必须确实在 actor 卡集合内;共卡段必须排在专用段之前
    # (槽位顺序决定 needs_offload,见 rollout.py _compute_rollout_offset)。
    seen_dedicated = False
    for item in layout.rollout:
        if item.share:
            if seen_dedicated:
                raise ValueError(
                    "resource layout: shared (share=actor) rollout entries must be listed before dedicated ones"
                )
            for device in item.devices:
                if (item.node, device) not in actor_devices:
                    raise ValueError(
                        f"resource layout: shared rollout device {item.node}:{device} is not in roles.actor"
                    )
        else:
            seen_dedicated = True
