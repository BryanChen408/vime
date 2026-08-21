"""场景 D 的调试扩展:按指定名字顺序从 checkpoint 读权重喂给引擎。

复刻生产 .64 接收侧的引擎处理(start 会话 → 分块 load_weights → finish),
但传输替换为本地读盘,从而可以任意控制**投喂顺序/bucketing**——
用于验证"direct 迭代器按名字字母序排序(PD 是非专家先行)"是否触发
共享专家校验失败。
"""


class DebugReloadExtension:
    """仅供 tests/repro_weight_reload_offline.py 场景 D 使用。"""

    def debug_feed_chunk(self, update_info: dict) -> None:
        import torch
        from safetensors.torch import load_file
        from vllm.config import set_current_vllm_config

        model = self.model_runner.model
        tensors = []
        for name, shard in zip(update_info["names"], update_info["shards"], strict=True):
            tensor = load_file(shard).get(name)
            if tensor is None:
                raise KeyError(f"{name} not in {shard}")
            tensors.append((name, tensor))
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            model.load_weights(weights=iter(tensors))
        torch.accelerator.synchronize()
