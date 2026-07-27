"""Explicit node/card resource layout parsing for Ray placement."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

ROLE_NAMES = ("actor", "critic", "rollout", "polar_reserved")


@dataclasses.dataclass(frozen=True)
class NodeDevices:
    node: str
    devices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ResourceLayout:
    """Which cards on which nodes each role runs on.

    An empty ``critic`` means the critic shares the actor's bundles; giving it its own
    entries places it on separate cards so a node holds only one model's optimizer.
    """

    actor: tuple[NodeDevices, ...] = ()
    critic: tuple[NodeDevices, ...] = ()
    rollout: tuple[NodeDevices, ...] = ()
    polar_reserved: tuple[NodeDevices, ...] = ()
    rollout_num_gpus_per_engine: int | None = None

    @property
    def actor_num_gpus(self) -> int:
        return _num_gpus(self.actor)

    @property
    def actor_num_nodes(self) -> int:
        return len(_device_counts_by_node(self.actor))

    @property
    def actor_num_gpus_per_node(self) -> int:
        return _uniform_gpus_per_node(self.actor, "actor")

    @property
    def critic_num_gpus(self) -> int:
        return _num_gpus(self.critic)

    @property
    def critic_num_nodes(self) -> int:
        return len(_device_counts_by_node(self.critic))

    @property
    def critic_num_gpus_per_node(self) -> int:
        return _uniform_gpus_per_node(self.critic, "critic")

    @property
    def rollout_num_gpus(self) -> int:
        return _num_gpus(self.rollout)

    @property
    def rollout_num_nodes(self) -> int:
        return len(_device_counts_by_node(self.rollout))

    @property
    def rollout_num_gpus_per_node(self) -> int:
        return _uniform_gpus_per_node(self.rollout, "rollout")

    @property
    def ray_num_gpus(self) -> int:
        # polar_reserved is deliberately excluded: those cards are held by an external
        # service and never handed to Ray.
        return self.actor_num_gpus + self.critic_num_gpus + self.rollout_num_gpus


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
        rollout_num_gpus_per_engine=_optional_positive_int(data, "rollout", "num_gpus_per_engine"),
    )
    _validate_layout(layout)
    return layout


def select_role_bundles(
    bundle_infos: list[tuple[int, str, Any]],
    role: tuple[NodeDevices, ...],
    *,
    role_name: str,
) -> tuple[list[int], list[int]]:
    """Pick the placement-group bundles backing a role, in the order the layout lists them.

    ``bundle_infos`` holds ``(bundle_index, node_ip, device_id)`` triples as reported by Ray.
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
            bundle_index = by_node_device.get((item.node, device))
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


def _num_gpus(entries: tuple[NodeDevices, ...]) -> int:
    return sum(len(item.devices) for item in entries)


def _device_counts_by_node(entries: tuple[NodeDevices, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in entries:
        counts[item.node] = counts.get(item.node, 0) + len(item.devices)
    return counts


def _uniform_gpus_per_node(entries: tuple[NodeDevices, ...], role_name: str) -> int:
    """Device count per node, which the Megatron and vLLM paths require to be uniform."""
    if not entries:
        return 0
    counts = set(_device_counts_by_node(entries).values())
    if len(counts) != 1:
        raise ValueError(
            f"resource layout requires equal {role_name} device counts per node, got {sorted(counts)}"
        )
    return counts.pop()


def _parse_devices(value: Any) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        devices: tuple[int, ...] = (value,)
    elif isinstance(value, str):
        devices = _parse_device_string(value)
    elif isinstance(value, (list, tuple)):
        devices = tuple(_parse_device_int(item) for item in value)
    else:
        raise TypeError(f"devices must be an int, string, or list of ints; got {type(value).__name__}")

    seen: set[int] = set()
    for device in devices:
        if device < 0:
            raise ValueError(f"device id must be >= 0, got {device}")
        if device in seen:
            raise ValueError(f"duplicate device id {device}")
        seen.add(device)
    return devices


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
        if raw_entry.get("node") is None:
            raise ValueError(f"resource layout roles.{name}[{idx}].node is required")
        node = str(raw_entry["node"]).strip()
        if not node:
            raise ValueError(f"resource layout roles.{name}[{idx}].node must not be empty")
        if "devices" not in raw_entry:
            raise ValueError(f"resource layout roles.{name}[{idx}].devices is required")
        entries.append(NodeDevices(node=node, devices=_parse_devices(raw_entry["devices"])))
    return tuple(entries)


def _optional_positive_int(data: dict[str, Any], section_name: str, key: str) -> int | None:
    section = data.get(section_name)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise TypeError(f"resource layout {section_name} must be a mapping")
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"resource layout {section_name}.{key} must be an integer, got bool")
    out = int(value)
    if out <= 0:
        raise ValueError(f"resource layout {section_name}.{key} must be > 0, got {out}")
    return out


def _validate_layout(layout: ResourceLayout) -> None:
    if layout.actor_num_gpus <= 0:
        raise ValueError("resource layout roles.actor must request at least one device")
    if layout.rollout_num_gpus <= 0:
        raise ValueError("resource layout roles.rollout must request at least one device")

    # Reject non-uniform shapes here rather than when the placement group is built.
    for entries, role_name in ((layout.actor, "actor"), (layout.critic, "critic"), (layout.rollout, "rollout")):
        _uniform_gpus_per_node(entries, role_name)

    used: dict[tuple[str, int], str] = {}
    for role_name in ROLE_NAMES:
        for item in getattr(layout, role_name):
            for device in item.devices:
                key = (item.node, device)
                prior = used.get(key)
                if prior is not None:
                    raise ValueError(
                        f"resource layout device overlap: {item.node}:{device} is in both {prior} and {role_name}"
                    )
                used[key] = role_name
