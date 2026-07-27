import copy
import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vime.ray.resource_layout import select_role_bundles
from vime.utils.common import is_npu

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)

# Weight of the ``node:<ip>`` resource a layout bundle asks for. Any non-zero amount pins the
# bundle to that node; keep it small so it never competes with real placements.
_NODE_PIN_RESOURCE_AMOUNT = 0.001

# The RolloutManager shares a bundle with the engine it fronts, so it takes a token CPU slice
# rather than the whole CPU the position path gives it.
_LAYOUT_ROLLOUT_MANAGER_NUM_CPUS = 0.01


# @ray.remote(num_gpus=1)
@ray.remote
class InfoActor:
    def get_ip_and_gpu_id(self):
        try:
            import torch_npu  # noqa: F401

            has_npu = True
        except ImportError:
            has_npu = False

        if has_npu or is_npu():
            npu_ids = ray.get_runtime_context().get_accelerator_ids().get("NPU", [])
            if npu_ids:
                return ray.util.get_node_ip_address(), npu_ids[0]

        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return ray.util.get_node_ip_address(), gpu_ids[0]

        raise RuntimeError(
            "No GPU/NPU IDs found. "
            f"Accelerator IDs: {ray.get_runtime_context().get_accelerator_ids()}, GPU IDs: {gpu_ids}"
        )


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, int(gpu_id))


def _create_and_probe_placement_group(bundles):
    """Create a PACK placement group and probe which ``(node, gpu)`` each bundle landed on.

    Returns ``(pg, bundle_infos)`` with ``bundle_infos`` a list of unsorted
    ``(bundle_index, node_ip, gpu_id)`` triples.
    """
    device_name = "NPU" if is_npu() else "GPU"
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
                resources={device_name: 1},
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    return pg, bundle_infos


def _log_bundle_order(bundle_infos, reordered_bundle_indices):
    by_index = {index: (node, gpu_id) for index, node, gpu_id in bundle_infos}
    for i, actual_bundle_index in enumerate(reordered_bundle_indices):
        node, gpu_id = by_index[actual_bundle_index]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, node: {node}, gpu: {gpu_id}"
        )


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    device_name = "NPU" if is_npu() else "GPU"
    pg, bundle_infos = _create_and_probe_placement_group([{device_name: 1, "CPU": 1} for _ in range(num_gpus)])

    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [info[2] for info in sorted_bundle_infos]
    _log_bundle_order(bundle_infos, pg_reordered_bundle_indices)

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _alive_node_resource_keys():
    """Map each alive Ray node IP to its built-in ``node:<ip>`` resource key.

    Connects first: unlike ``placement_group()``, ``ray.nodes()`` does not attach to a
    running cluster on its own, and this is reached before anything else has.
    """
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    keys = {}
    alive_ips = set()
    for node in ray.nodes():
        if not node.get("Alive", True):
            continue
        ip = str(node.get("NodeManagerAddress") or "")
        if not ip:
            continue
        alive_ips.add(ip)
        resource_key = f"node:{ip}"
        if resource_key in (node.get("Resources") or {}):
            keys[ip] = resource_key
    return keys, alive_ips


def _build_layout_bundles(layout, device_name):
    """Build one PACK bundle per requested device, pinned to the node the layout names."""
    node_resource_keys, alive_ips = _alive_node_resource_keys()
    bundles = []
    for role in (layout.actor, layout.critic, layout.rollout):
        for item in role:
            node_resource_key = node_resource_keys.get(item.node)
            if node_resource_key is None:
                if item.node in alive_ips:
                    raise ValueError(
                        f"Ray node {item.node!r} is alive but exposes no 'node:{item.node}' resource, "
                        "so bundles cannot be pinned to it."
                    )
                available = ", ".join(sorted(alive_ips)) or "<none>"
                raise ValueError(
                    f"Resource layout requested node {item.node!r}, but it is not an active Ray node. "
                    f"Available Ray nodes: {available}"
                )
            for _ in item.devices:
                bundles.append({device_name: 1, "CPU": 1, node_resource_key: _NODE_PIN_RESOURCE_AMOUNT})
    return bundles


def _create_placement_groups_from_layout(args):
    """Pin each role to the exact (node, devices) the layout spells out."""
    layout = args.resource_layout_spec
    logger.info(f"Creating placement group from resource layout with {layout.ray_num_gpus} GPUs...")
    device_name = "NPU" if is_npu() else "GPU"
    pg, bundle_infos = _create_and_probe_placement_group(_build_layout_bundles(layout, device_name))

    actor_bundle_indices, actor_gpu_ids = select_role_bundles(bundle_infos, layout.actor, role_name="actor")
    rollout_bundle_indices, rollout_gpu_ids = select_role_bundles(bundle_infos, layout.rollout, role_name="rollout")

    logger.info("Actor placement from resource layout:")
    _log_bundle_order(bundle_infos, actor_bundle_indices)
    logger.info("Rollout placement from resource layout:")
    _log_bundle_order(bundle_infos, rollout_bundle_indices)

    if not args.use_critic:
        critic_placement = None
    elif layout.critic:
        critic_bundle_indices, critic_gpu_ids = select_role_bundles(bundle_infos, layout.critic, role_name="critic")
        logger.info("Critic placement from resource layout:")
        _log_bundle_order(bundle_infos, critic_bundle_indices)
        critic_placement = (pg, critic_bundle_indices, critic_gpu_ids)
    else:
        # No critic entries: share the actor's bundles, as the position path does.
        critic_placement = (pg, actor_bundle_indices, actor_gpu_ids)

    return {
        "actor": (pg, actor_bundle_indices, actor_gpu_ids),
        "critic": critic_placement,
        "rollout": (pg, rollout_bundle_indices, rollout_gpu_ids),
    }


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    if getattr(args, "resource_layout_spec", None) is not None:
        return _create_placement_groups_from_layout(args)

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
    )


def create_training_models(args, pgs, rollout_manager):
    actor_args = args
    if args.megatron_config_path is not None:
        from vime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )

    critic_model = None
    if args.use_critic:
        from vime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else copy.deepcopy(args)
        )
        if args.megatron_config_path is None:
            critic_args.disable_param_buffers_cpu_backup = False

        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = ray.get(critic_model.async_init(critic_model.args, role="critic", with_ref=False))

    actor_start_rollout_ids = ray.get(
        actor_model.async_init(
            actor_args,
            role="actor",
            with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
            with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        )
    )
    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if args.use_critic:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    device_name = "NPU" if is_npu() else "GPU"
    options = dict(num_cpus=1, resources={device_name: 0})
    if getattr(args, "resource_layout_spec", None) is not None:
        # Put the manager on the first rollout bundle so the engines it spawns inherit the
        # layout's node and card placement.
        placement_pg, rollout_bundle_indices, _ = pg
        options["num_cpus"] = _LAYOUT_ROLLOUT_MANAGER_NUM_CPUS
        options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
            placement_group=placement_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=rollout_bundle_indices[0],
        )
    rollout_manager = RolloutManager.options(**options).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
