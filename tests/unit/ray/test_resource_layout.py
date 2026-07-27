"""Unit tests for ``vime.ray.resource_layout``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vime.ray.resource_layout import NodeDevices, load_resource_layout, resource_layout_from_dict, select_role_bundles

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_LAYOUTS = REPO_ROOT / "scripts"


def _layout(**roles):
    return resource_layout_from_dict({"roles": roles})


@pytest.mark.unit
def test_device_counts_are_derived_per_role():
    layout = _layout(
        actor=[{"node": "n1", "devices": "0-7"}],
        critic=[{"node": "n2", "devices": "0-7"}],
        rollout=[{"node": "n2", "devices": "8-15"}],
        polar_reserved=[{"node": "n1", "devices": "8-15"}],
    )
    assert (layout.actor_num_gpus, layout.actor_num_nodes, layout.actor_num_gpus_per_node) == (8, 1, 8)
    assert (layout.critic_num_gpus, layout.critic_num_nodes, layout.critic_num_gpus_per_node) == (8, 1, 8)
    assert (layout.rollout_num_gpus, layout.rollout_num_nodes, layout.rollout_num_gpus_per_node) == (8, 1, 8)


@pytest.mark.unit
def test_ray_num_gpus_excludes_polar_reserved():
    # polar_reserved cards belong to an external service and are never handed to Ray.
    layout = _layout(
        actor=[{"node": "n1", "devices": "0-3"}],
        rollout=[{"node": "n1", "devices": "4-7"}],
        polar_reserved=[{"node": "n1", "devices": "8-15"}],
    )
    assert layout.ray_num_gpus == 8


@pytest.mark.unit
def test_ray_num_gpus_counts_an_independent_critic():
    colocated = _layout(actor=[{"node": "n1", "devices": "0-3"}], rollout=[{"node": "n1", "devices": "4-7"}])
    assert colocated.critic_num_gpus == 0
    assert colocated.ray_num_gpus == 8

    split = _layout(
        actor=[{"node": "n1", "devices": "0-3"}],
        critic=[{"node": "n2", "devices": "0-3"}],
        rollout=[{"node": "n1", "devices": "4-7"}],
    )
    assert split.ray_num_gpus == 12


@pytest.mark.unit
@pytest.mark.parametrize("role", ["actor", "critic", "rollout"])
def test_uneven_device_counts_per_node_are_rejected_at_parse_time(role):
    roles = {
        "actor": [{"node": "n1", "devices": "0-3"}],
        "rollout": [{"node": "n1", "devices": "4-7"}],
    }
    roles[role] = [{"node": "n1", "devices": "0-1"}, {"node": "n2", "devices": "0-3"}]
    if role != "rollout":
        roles["rollout"] = [{"node": "n3", "devices": "0-1"}]
    if role != "actor":
        roles["actor"] = [{"node": "n4", "devices": "0-1"}]

    with pytest.raises(ValueError, match=f"equal {role} device counts"):
        resource_layout_from_dict({"roles": roles})


@pytest.mark.unit
def test_overlapping_devices_are_rejected():
    with pytest.raises(ValueError, match="device overlap"):
        _layout(actor=[{"node": "n1", "devices": "0-3"}], rollout=[{"node": "n1", "devices": "3-5"}])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("devices", "expected"),
    [("0-3", (0, 1, 2, 3)), ("0,2,4", (0, 2, 4)), ("1, 3-5", (1, 3, 4, 5)), ([2, 0], (2, 0)), (7, (7,))],
)
def test_device_spellings(devices, expected):
    layout = _layout(actor=[{"node": "n1", "devices": devices}], rollout=[{"node": "n2", "devices": "0"}])
    assert layout.actor[0].devices == expected


@pytest.mark.unit
def test_node_zero_is_a_valid_name():
    # A falsy-but-present node must not be reported as missing.
    layout = _layout(actor=[{"node": 0, "devices": "0-1"}], rollout=[{"node": "n1", "devices": "0"}])
    assert layout.actor[0].node == "0"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"devices": "0-1"}, "node is required"),
        ({"node": "  ", "devices": "0-1"}, "must not be empty"),
        ({"node": "n1"}, "devices is required"),
        ({"node": "n1", "devices": ""}, "must not be empty"),
        ({"node": "n1", "devices": "3-1"}, "descending"),
        ({"node": "n1", "devices": "0,0"}, "duplicate device id"),
        ({"node": "n1", "devices": True}, "int, string, or list"),
    ],
)
def test_malformed_entries_are_rejected(entry, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _layout(actor=[entry], rollout=[{"node": "n9", "devices": "0"}])


@pytest.mark.unit
def test_error_messages_do_not_name_the_cli_flag():
    # resource_layout_from_dict is a programmatic entry point; naming a flag the caller never
    # passed would misdirect.
    with pytest.raises(ValueError) as excinfo:
        _layout(
            actor=[{"node": "a", "devices": "0-1"}, {"node": "b", "devices": "0-2"}],
            rollout=[{"node": "c", "devices": "0"}],
        )
    assert "--resource-layout" not in str(excinfo.value)


@pytest.mark.unit
def test_rollout_section_is_read_from_yaml(tmp_path):
    path = tmp_path / "layout.yaml"
    path.write_text(
        textwrap.dedent(
            """
            roles:
              actor:
                - {node: n1, devices: "0-7"}
              rollout:
                - {node: n2, devices: "0-15"}
            rollout:
              num_gpus_per_engine: 4
            """
        )
    )
    layout = load_resource_layout(path)
    assert layout.rollout_num_gpus_per_engine == 4


@pytest.mark.unit
def test_absent_rollout_section_leaves_optionals_none():
    layout = _layout(actor=[{"node": "n1", "devices": "0-7"}], rollout=[{"node": "n2", "devices": "0-15"}])
    assert layout.rollout_num_gpus_per_engine is None


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, -1, True])
def test_non_positive_rollout_options_are_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        resource_layout_from_dict(
            {
                "roles": {
                    "actor": [{"node": "n1", "devices": "0-7"}],
                    "rollout": [{"node": "n2", "devices": "0-15"}],
                },
                "rollout": {"num_gpus_per_engine": value},
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize("path", sorted(SHIPPED_LAYOUTS.glob("resource_layout*.yaml")), ids=lambda p: p.name)
def test_shipped_layouts_parse_and_divide_evenly_into_engines(path):
    layout = load_resource_layout(path)
    assert layout.ray_num_gpus > 0
    if layout.rollout_num_gpus_per_engine:
        assert layout.rollout_num_gpus % layout.rollout_num_gpus_per_engine == 0


@pytest.mark.unit
def test_select_role_bundles_follows_layout_order():
    bundle_infos = [(0, "n1", 4), (1, "n1", 5), (2, "n2", 0)]
    role = (NodeDevices("n1", (5, 4)), NodeDevices("n2", (0,)))
    assert select_role_bundles(bundle_infos, role, role_name="actor") == ([1, 0, 2], [5, 4, 0])


@pytest.mark.unit
def test_select_role_bundles_reports_missing_devices():
    bundle_infos = [(0, "n1", 4)]
    role = (NodeDevices("n1", (4, 9)),)
    with pytest.raises(ValueError, match=r"missing requested actor devices: n1:9"):
        select_role_bundles(bundle_infos, role, role_name="actor")
