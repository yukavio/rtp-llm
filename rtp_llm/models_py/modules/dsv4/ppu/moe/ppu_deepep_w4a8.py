"""DeepEP per-token FP8 dispatch with BF16-boundary W4A8 expert GEMMs."""

import torch
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_fp4 import (
    PpuDeepEPFP4Strategy,
    pad_topk,
    take_raw_routed_mxfp4_weights,
)


class PpuDeepEPW4A8Strategy(PpuDeepEPFP4Strategy):
    name = "ppu_deepep_w4a8"

    @classmethod
    def can_handle(cls, cfg):
        return super().can_handle(cfg) and cfg.tp_size == 1 and cfg.ep_size == 8

    def __init__(self, cfg, **kwargs):
        kwargs.setdefault("output_dtype", torch.bfloat16)
        super().__init__(cfg, **kwargs)
        if self.output_dtype != torch.bfloat16 or self._gemm_tile != "auto":
            raise ValueError("W4A8 requires BF16 output and automatic GEMM tiles")
        import deep_gemm_w4a8 as dg

        if not all(
            callable(getattr(dg, key, None))
            for key in (
                "preprocess_weights_and_scales",
                "activation_and_mul_masked",
                "m_grouped_gemm_fp8_fp4_bf16_nt_masked",
            )
        ):
            raise RuntimeError("deep_gemm_w4a8 lacks the PPU W4A8 masked API")
        self._dg = dg

    def setup_weights(self, layer_weights):
        raw = take_raw_routed_mxfp4_weights(self.cfg, layer_weights)
        for prefix, index in (("13", 0), ("2", 2)):
            weight, scale, inverse = self._dg.preprocess_weights_and_scales(
                self._dg.PreprocessMode.W4A8_MXFP4_MMA,
                False,
                raw[index],
                raw[index + 1].view(torch.uint8),
            )
            self.register_buffer("_w" + prefix, weight, persistent=False)
            self.register_buffer("_s" + prefix, scale, persistent=False)
            setattr(self, "_inverse" + prefix, float(inverse))

    def forward(self, x, weights, indices, *, active_token_mask=None):
        wrapper = self._wrapper if self._wrapper is not None else self._bind_wrapper()
        capacity = wrapper.ll_num_max_token_per_rank
        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.shape[1] != self.cfg.dim
            or x.shape[0] > capacity
            or weights.ndim != 2
            or indices.shape != weights.shape
            or indices.shape[0] != x.shape[0]
            or indices.dtype != torch.int64
            or weights.dtype != torch.float32
            or indices.device != x.device
            or weights.device != x.device
        ):
            raise ValueError("Invalid W4A8 DeepEP tokens / routes")
        indices, weights = pad_topk(indices, weights, active_token_mask)
        indices, weights = indices.contiguous(), weights.contiguous()
        buffer, dg = wrapper.buffer, self._dg
        expert_x, counts, handle, _, _ = buffer.low_latency_dispatch(
            x=x.contiguous(),
            topk_idx=indices,
            num_max_dispatch_tokens_per_rank=capacity,
            num_experts=self.cfg.n_routed_experts,
            use_fp8=True,
            use_mxfp4=False,
            quant_size=self.cfg.dim,
            round_scale=False,
            use_ue8m0=False,
            async_finish=False,
            return_recv_hook=False,
        )
        if (
            not isinstance(expert_x, tuple)
            or len(expert_x) != 2
            or expert_x[0].dtype != torch.float8_e4m3fn
            or expert_x[1].dtype != torch.float32
            or expert_x[1].shape != (*expert_x[0].shape[:-1], 1)
        ):
            raise RuntimeError("W4A8 DeepEP requires per-token FP8 dispatch scales")
        e, rows, dim = expert_x[0].shape
        expected = self.expected_rows(x.shape[0])
        up = torch.empty(
            (e, rows, self.cfg.moe_inter_dim * 2), device=x.device, dtype=x.dtype
        )
        dg.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
            (expert_x[0], expert_x[1].contiguous()),
            (self._w13, self._s13),
            up,
            counts,
            expected,
            inv_scale=self._inverse13,
        )
        hidden = dg.activation_and_mul_masked(
            dg.ActivationMode.SILU,
            dg.QuantMode.FP8_CHANNEL,
            up,
            counts,
            expected_m=expected,
            swiglu_limit=self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None,
        )
        # Keep an independent output: DeepEP dispatch and combine may alias.
        down = torch.empty((e, rows, dim), device=x.device, dtype=x.dtype)
        dg.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
            hidden,
            (self._w2, self._s2),
            down,
            counts,
            expected,
            inv_scale=self._inverse2,
        )
        combined, _, _ = buffer.low_latency_combine(
            x=down,
            topk_idx=indices,
            topk_weights=weights,
            handle=handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
        )
        if combined.dtype != torch.bfloat16:
            raise RuntimeError("W4A8 DeepEP combine must return BF16")
        return combined
