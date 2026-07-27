"""C5:MTP chunked-CE 数值等价测试(见 docs/design/chunked_mtp_lmhead.md)。

验证 chunked_ce_over_seq(按 seq 分块 matmul + 逐块 CE + 拼回)在**前向 loss** 与**反向梯度**上
都等于不分块的 full CE。用 TP=1 的纯 cross_entropy 作参照(此时 vocab_parallel_cross_entropy 退化为
普通 CE),从而无需 megatron 分布式初始化即可跑。

注:TP/CP/SP 组合下的等价 + 端到端不 OOM 由 C6(NPU run)验证;这里锁定分块机制本身
(块边界 / 拼接 / label 对齐 / 梯度累加)的正确性。
"""
import torch
import torch.nn.functional as F

from vime.backends.megatron_utils.chunked_mtp_ce_patch import chunked_ce_over_seq


def _ref_compute_lm_loss(labels_bc: torch.Tensor, logits_cbv: torch.Tensor) -> torch.Tensor:
    """参照 LanguageModule.compute_language_model_loss(TP=1):labels[b,c], logits[c,b,V] -> [b,c]。"""
    lbl = labels_bc.transpose(0, 1).contiguous()            # [c, b]
    c, b, V = logits_cbv.shape
    loss = F.cross_entropy(
        logits_cbv.reshape(c * b, V), lbl.reshape(c * b), reduction="none"
    ).view(c, b)                                            # [c, b]
    return loss.transpose(0, 1).contiguous()                # [b, c]


def _make(s=37, b=2, h=16, V=128, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(s, b, h, generator=g, requires_grad=requires_grad)
    weight = torch.randn(V, h, generator=g, requires_grad=requires_grad)
    labels = torch.randint(0, V, (b, s), generator=g)
    return hidden, weight, labels


def test_chunked_ce_forward_equals_full():
    hidden, weight, labels = _make()
    s = hidden.size(0)
    full = chunked_ce_over_seq(hidden, weight, labels, chunk=s, compute_lm_loss=_ref_compute_lm_loss)
    for chunk in (1, 3, 8, 16, s - 1, s, s + 5):  # 含非整除 / 单块 / 超长
        got = chunked_ce_over_seq(hidden, weight, labels, chunk=chunk, compute_lm_loss=_ref_compute_lm_loss)
        assert got.shape == (hidden.size(1), s), f"shape {got.shape}"
        assert torch.allclose(got, full, atol=1e-5, rtol=1e-5), f"chunk={chunk} 前向不等价"


def test_chunked_ce_backward_equals_full():
    # 梯度等价:分块只应改变峰值内存,不改变数学
    h1, w1, labels = _make(requires_grad=True)
    h2, w2 = h1.detach().clone().requires_grad_(True), w1.detach().clone().requires_grad_(True)
    s = h1.size(0)

    chunked_ce_over_seq(h1, w1, labels, chunk=s, compute_lm_loss=_ref_compute_lm_loss).sum().backward()
    chunked_ce_over_seq(h2, w2, labels, chunk=7, compute_lm_loss=_ref_compute_lm_loss).sum().backward()

    assert torch.allclose(w1.grad, w2.grad, atol=1e-5, rtol=1e-5), "weight 梯度不等价"
    assert torch.allclose(h1.grad, h2.grad, atol=1e-5, rtol=1e-5), "hidden 梯度不等价"


if __name__ == "__main__":
    test_chunked_ce_forward_equals_full()
    test_chunked_ce_backward_equals_full()
    print("chunked MTP-CE 数值等价(前向+反向)通过 ✅")
