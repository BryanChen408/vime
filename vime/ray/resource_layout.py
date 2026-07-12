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


@dataclasses.dataclass(frozen=True)
class ResourceLayout:
    actor: tuple[NodeDevices, ...] = ()
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
    def rollout_num_gpus_per_node(self) -> int:
        if not self.rollout:
            return 0
        counts = set(_device_counts_by_node(self.rollout).values())
        if len(counts) != 1:
            raise ValueError(
                "--resource-layout requires equal rollout device counts per node for the current vLLM path; "
                f"got {sorted(counts)}"
            )
        return counts.pop()

    @property
    def ray_num_gpus(self) -> int:
        return self.actor_num_gpus + self.rollout_num_gpus

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
        entries.append(NodeDevices(node=node, devices=parse_devices(raw_entry["devices"])))
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
    for role_name, entries in (
        ("actor", layout.actor),
        ("rollout", layout.rollout),
        ("polar_reserved", layout.polar_reserved),
    ):
        for item in entries:
            for device in item.devices:
                key = (item.node, device)
                prior = used.get(key)
                if prior is not None:
                    raise ValueError(
                        f"resource layout device overlap: {item.node}:{device} is in both {prior} and {role_name}"
                    )
                used[key] = role_name
