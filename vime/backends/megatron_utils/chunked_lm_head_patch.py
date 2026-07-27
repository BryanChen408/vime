"""Chunked LM-head: make GPTModel.forward return decoder hidden states.

On the last pipeline stage GPTModel.forward materialises the full [seq, vocab/tp] logits,
which the log-prob stage then has to keep alive. At long sequence lengths that dominates
the activation memory.

When enabled, the patched forward skips the output layer and returns the decoder hidden
states instead, stashing the output layer's weight for the loss to pick up. The loss then
computes log-probs one chunk at a time, so the peak is [chunk, vocab/tp] and no longer
scales with the sequence length.

The patch is only active for a `labels=None` forward on the last pipeline stage; every
other call falls through to the original implementation.
"""

# The weight is stashed at module scope rather than on the returned tensor: the pipeline
# engine may make the output contiguous or rebuild it, which drops tensor attributes.
_LM_HEAD_WEIGHT = None


def get_captured_lm_head_weight():
    """The output layer weight stashed by the patched forward, or None if it did not run."""
    return _LM_HEAD_WEIGHT


def apply_chunked_lm_head_patch():
    """Patch GPTModel.forward in place. Idempotent; call once after the model is built."""
    from megatron.core.models.gpt.gpt_model import GPTModel

    if getattr(GPTModel, "_chunked_lm_head_patched", False):
        return
    original_forward = GPTModel.forward

    def _bypasses_output_layer(model, labels):
        if labels is not None or not getattr(model, "post_process", False):
            return False
        if getattr(model, "mtp_process", False):
            # The MTP path computes its loss inside postprocess, so it needs real logits.
            return False
        weight = getattr(getattr(model, "output_layer", None), "weight", None)
        if weight is None:
            return False
        # A critic's value head is also an output layer, but it emits [T, 1], which is
        # trivially cheap to materialise. Bypassing it would hand get_responses hidden
        # states and trip its size(-1) == 1 assertion. Only the LM head is worth skipping.
        return weight.shape[0] != 1

    def _patched_forward(self, *args, **kwargs):
        global _LM_HEAD_WEIGHT
        if not _bypasses_output_layer(self, kwargs.get("labels")):
            # Clear it, so a caller that also runs an unpatched forward (a critic's value
            # head, say) is never handed the weight left over from an earlier LM forward.
            _LM_HEAD_WEIGHT = None
            return original_forward(self, *args, **kwargs)

        output_layer = self.output_layer
        _LM_HEAD_WEIGHT = output_layer.weight
        sequence_parallel = getattr(output_layer, "sequence_parallel", False)
        tp_group = getattr(output_layer, "tp_group", None)

        def _bypass(hidden, weight=None, runtime_gather_output=None):
            # The real forward gathers the sequence before the matmul when sequence
            # parallelism is on; without that the caller would get a sequence shard.
            if sequence_parallel:
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

                hidden = gather_from_sequence_parallel_region(hidden, group=tp_group)
            return hidden, None

        original_output_layer_forward = output_layer.forward
        try:
            output_layer.forward = _bypass  # type: ignore[method-assign]
            return original_forward(self, *args, **kwargs)
        finally:
            output_layer.forward = original_output_layer_forward  # type: ignore[method-assign]

    GPTModel.forward = _patched_forward
    GPTModel._chunked_lm_head_patched = True
