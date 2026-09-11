from pathlib import Path

from vime.ray.placement_group import _build_layout_bundles
from vime.ray.resource_layout import resource_layout_from_dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sparse_hybrid_layout_pads_physical_device_ids(monkeypatch):
    layout = resource_layout_from_dict(
        {
            "roles": {
                "actor": [{"node": "node-a", "devices": "4-11"}],
                "rollout": [
                    {"node": "node-a", "devices": "4-11", "share": "actor"},
                    {"node": "node-a", "devices": "12-15"},
                ],
                "polar_reserved": [{"node": "node-a", "devices": "0-3"}],
            }
        }
    )
    monkeypatch.setattr(
        "vime.ray.placement_group._node_resource_keys_by_ip",
        lambda: {"node-a": "node:node-a"},
    )

    bundles = _build_layout_bundles(layout, "NPU")

    # Ray must expose accelerator ids 0..15 so the role selectors can refer to
    # the physical ids in the YAML instead of a compact 0..11 renumbering.
    assert len(bundles) == 16
    assert [bundle["node:node-a"] for bundle in bundles] == [0.001] * 16
    assert all(bundle["NPU"] == 1 and bundle["CPU"] == 1 for bundle in bundles)


def test_sparse_layout_pads_each_node_independently(monkeypatch):
    layout = resource_layout_from_dict(
        {
            "roles": {
                "actor": [{"node": "node-a", "devices": "8-15"}],
                "rollout": [{"node": "node-b", "devices": "4-11"}],
            }
        }
    )
    monkeypatch.setattr(
        "vime.ray.placement_group._node_resource_keys_by_ip",
        lambda: {"node-a": "node:node-a", "node-b": "node:node-b"},
    )

    bundles = _build_layout_bundles(layout, "NPU")

    # Node-local accelerator ids need their own zero-based padding.  The
    # ordering is deterministic so bundle probing remains reproducible.
    assert len(bundles) == 16 + 12
    assert [bundle.get("node:node-a") for bundle in bundles[:16]] == [0.001] * 16
    assert [bundle.get("node:node-b") for bundle in bundles[16:]] == [0.001] * 12
