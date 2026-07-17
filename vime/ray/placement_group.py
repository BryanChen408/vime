import copy
import logging
import os
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

from vime.ray.resource_layout import select_role_bundles
from vime.utils.common import is_npu

logger = logging.getLogger(__name__)


# @ray.remote(num_gpus=1)
@ray.remote
class InfoActor:
    def get_ip_and_gpu_id(self):
        if is_npu():
            return ray.util.get_node_ip_address(), ray.get_runtime_context().get_accelerator_ids()["NPU"][0]
        else:
            return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


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


def _ensure_ray_initialized():
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)


def _create_and_probe_placement_group(num_gpus, bundles=None):
    """Create a PACK placement group and probe each bundle's ``(node_ip, gpu_id)``.

    When ``bundles`` is provided (resource-layout path), it pins bundles to
    specific nodes; otherwise a plain ``num_gpus``-bundle PACK group is used
    (position path). Returns ``(pg, bundle_infos)`` where ``bundle_infos`` is a
    list of ``(bundle_index, node_ip, gpu_id)`` triples (unsorted).
    """
    _ensure_ray_initialized()
    device_name = "NPU" if is_npu() else "GPU"
    if bundles is None:
        bundles = [{device_name: 1, "CPU": 1} for _ in range(num_gpus)]
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
                resources={device_name:1}
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    return pg, bundle_infos


def _node_resource_keys_by_ip():
    """Map each active Ray node IP to its built-in ``node:<ip>`` resource key."""
    _ensure_ray_initialized()
    out = {}
    for node in ray.nodes():
        if not node.get("Alive", True):
            continue
        ip = str(node.get("NodeManagerAddress") or "")
        resources = node.get("Resources") or {}
        exact = f"node:{ip}"
        if exact in resources:
            out[ip] = exact
            continue
        node_keys = [key for key in resources if key.startswith("node:")]
        if len(node_keys) == 1:
            out[ip] = node_keys[0]
    return out


def _build_layout_bundles(layout, device_name):
    """Build PACK bundles that pin each ``(node, device)`` to its node via a tiny
    ``node:<ip>: 0.001`` resource request (forces the bundle onto that node)."""
    node_resource_keys = _node_resource_keys_by_ip()
    bundles = []
    # layout.critic 默认空 () → 不产生额外 bundle(共卡老路径);非空则为独立 critic 卡位建 bundle。
    for role in (layout.actor, layout.critic, layout.rollout):
        for item in role:
            node_resource_key = node_resource_keys.get(item.node)
            if node_resource_key is None:
                available = ", ".join(sorted(node_resource_keys)) or "<none>"
                raise ValueError(
                    f"Resource layout requested node {item.node!r}, but it is not an active Ray node. "
                    f"Available Ray nodes: {available}"
                )
            for _ in item.devices:
                bundles.append({device_name: 1, "CPU": 1, node_resource_key: 0.001})
    return bundles


def _log_bundle_order(bundle_infos, reordered_bundle_indices):
    by_index = {index: (node, gpu_id) for index, node, gpu_id in bundle_infos}
    for i, actual_bundle_index in enumerate(reordered_bundle_indices):
        node, gpu_id = by_index[actual_bundle_index]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {node}, gpu: {gpu_id}"
        )


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs (position path)."""
    pg, bundle_infos = _create_and_probe_placement_group(num_gpus)
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [info[2] for info in sorted_bundle_infos]
    _log_bundle_order(bundle_infos, pg_reordered_bundle_indices)

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _create_placement_groups_from_layout(args):
    """Explicit-layout path: pin actor/rollout to the (node, devices) in the spec."""
    layout = args.resource_layout_spec
    logger.info(f"Creating placement group from resource layout with {layout.ray_num_gpus} GPUs...")
    device_name = "NPU" if is_npu() else "GPU"
    pg, bundle_infos = _create_and_probe_placement_group(
        layout.ray_num_gpus,
        bundles=_build_layout_bundles(layout, device_name),
    )

    actor_bundle_indices, actor_gpu_ids = select_role_bundles(bundle_infos, layout.actor, role_name="actor")
    rollout_bundle_indices, rollout_gpu_ids = select_role_bundles(bundle_infos, layout.rollout, role_name="rollout")

    logger.info("Actor placement from resource layout:")
    _log_bundle_order(bundle_infos, actor_bundle_indices)
    logger.info("Rollout placement from resource layout:")
    _log_bundle_order(bundle_infos, rollout_bundle_indices)

    # critic 卡位:
    #   - layout 无 critic 条目(layout.critic 空)→ 复用 actor bundle = 共卡老路径(行为不变,
    #     [F-PPO-2 步骤2/A1] critic 与 actor 共卡 offload 时分,与位置路径 result["critic"]=result["actor"] 一致)。
    #   - layout 显式给 critic 独立 (node, devices) → 建独立 placement(分卡 / 跨节点),每节点只放一个模型优化器。
    if not args.use_critic:
        critic_placement = None
    elif layout.critic:
        critic_bundle_indices, critic_gpu_ids = select_role_bundles(bundle_infos, layout.critic, role_name="critic")
        logger.info("Critic placement from resource layout (independent, not colocated with actor):")
        _log_bundle_order(bundle_infos, critic_bundle_indices)
        critic_placement = (pg, critic_bundle_indices, critic_gpu_ids)
    else:
        critic_placement = (pg, actor_bundle_indices, actor_gpu_ids)

    return {
        "actor": (pg, actor_bundle_indices, actor_gpu_ids),
        "critic": critic_placement,
        "rollout": (pg, rollout_bundle_indices, rollout_gpu_ids),
    }


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    if getattr(args, "resource_layout_spec", None) is not None:
        # [F-PPO-2 步骤2/A1] critic 与 actor 共卡(见 _create_placement_groups_from_layout),
        # 故 layout 路径支持 critic;原 "does not support critic" 限制取消。
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

    # [拓扑] VIME_ROLLOUT_LOW_CARDS=1:把 rollout 钉到升序前 rollout_num 张卡(如 4-7),actor 用其余
    # 高卡(如 8-15,同一 HCCS 域,训练集合通信不跨 7/8 边界 → 免 EI0013 跨域 RoCE)。走已验证稳的
    # 位置路径(plain bundle,dist-init 无 7/8 问题),只是把 actor/rollout 的卡切法对调。仅 normal
    # 模式(非 colocate/debug)。retro-compat:env 不设=原行为(actor 首 N 卡、rollout 尾卡)。
    if (
        os.environ.get("VIME_ROLLOUT_LOW_CARDS") == "1"
        and not args.colocate
        and not args.debug_train_only
        and not args.debug_rollout_only
        and not args.use_critic
    ):
        rn = args.rollout_num_gpus
        rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[:rn]
        rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[:rn]
        actor_indices = actor_pg_reordered_bundle_indices[rn:]
        actor_gpu_ids = actor_pg_reordered_gpu_ids[rn:]
        logger.info(
            f"[VIME_ROLLOUT_LOW_CARDS] rollout cards={rollout_pg_reordered_gpu_ids}, actor cards={actor_gpu_ids}"
        )
        return {
            "actor": (pg, actor_indices, actor_gpu_ids),
            "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
            "critic": None,
        }

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
        # Pin the RolloutManager actor onto the first rollout bundle so its child
        # engines inherit the layout's node/card placement.
        placement_pg, rollout_bundle_indices, _ = pg
        options["num_cpus"] = 0.01
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
