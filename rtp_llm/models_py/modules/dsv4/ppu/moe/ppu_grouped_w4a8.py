"""PPU local MXFP4 experts with per-token E4M3 activation quantization.

The two GEMMs produce BF16, matching the reference activation boundaries.
The optional dependency is isolated from the legacy DeepGEMM JIT package.
"""

import logging
import math
from dataclasses import replace

import torch
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_fp4 import (
    take_raw_routed_mxfp4_weights,
)
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_grouped_fp4 import (
    _derive_inter_local_and_tp,
)


class PpuGroupedW4A8Strategy(torch.nn.Module):
    name = "ppu_grouped_w4a8"
    route_weight_contract = "post_w2_normalized_then_scale_v1"

    def __init__(self, cfg):
        super().__init__()
        if cfg.ep_size != 1 or cfg.tp_size < 1:
            raise ValueError("Grouped W4A8 requires EP1 with positive TP size")
        self.cfg = cfg
        self.inter_local = 0
        self.routed_tp_size = 1
        import deep_gemm_w4a8 as dg

        required = (
            "preprocess_weights_and_scales",
            "moe_align_block_size",
            "m_grouped_gemm_fp8_fp4_bf16_nt_fused",
            "m_grouped_gemm_fp8_fp4_bf16_nt_nopad",
            "activation_and_mul",
        )
        if any(not callable(getattr(dg, name, None)) for name in required):
            raise RuntimeError("deep_gemm_w4a8 lacks the PPU W4A8 grouped API")
        self._dg = dg

    def setup_weights(self, layer_weights):
        from rtp_llm.utils.model_weight import W

        shapes = [
            tuple(layer_weights[key].shape)
            for key in (
                W.v4_routed_w1_w,
                W.v4_routed_w2_w,
                W.v4_routed_w3_w,
                W.v4_routed_w1_s,
                W.v4_routed_w2_s,
                W.v4_routed_w3_s,
            )
        ]
        inter, tp = _derive_inter_local_and_tp(self.cfg, *shapes)
        raw = take_raw_routed_mxfp4_weights(
            replace(self.cfg, moe_inter_dim=inter), layer_weights
        )
        dg = self._dg
        pairs = [
            dg.preprocess_weights_and_scales(
                dg.PreprocessMode.W4A8_MXFP4_MMA,
                False,
                raw[i],
                raw[i + 1].view(torch.uint8),
            )
            for i in (0, 2)
        ]
        for prefix, (weight, scale, inverse) in zip(("13", "2"), pairs):
            self.register_buffer("_w" + prefix, weight, persistent=False)
            self.register_buffer("_s" + prefix, scale, persistent=False)
            if not math.isfinite(float(inverse)) or inverse <= 0:
                raise ValueError(
                    "W4A8 weight inverse scale must be finite and positive"
                )
            setattr(self, "_inverse" + prefix, float(inverse))
        self.inter_local, self.routed_tp_size = inter, tp
        logging.info(
            "DSV4_PPU_GROUPED_OPERATOR_PATH strategy=%s package=%s",
            self.name,
            dg.__file__,
        )

    def forward(
        self,
        x,
        weights,
        indices,
        *,
        shared=None,
        route_scale=None,
        shared_schedule=None
    ):
        if self.inter_local <= 0:
            raise RuntimeError("Grouped W4A8 weights were not bound")
        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.shape[1] != self.cfg.dim
            or weights.ndim != 2
            or weights.dtype != torch.float32
            or indices.dtype != torch.int64
            or indices.shape != weights.shape
            or weights.shape[0] != x.shape[0]
            or weights.shape[1] != self.cfg.n_activated_experts
            or weights.device != x.device
            or indices.device != x.device
            or x.device != self._w13.device
        ):
            raise ValueError(
                "Grouped W4A8 requires matching BF16 tokens, FP32 weights and int64 routes"
            )
        if (shared is None) != (route_scale is None):
            raise ValueError("shared and route_scale must be supplied together")
        if shared is not None and (
            shared.shape != x.shape
            or shared.dtype != x.dtype
            or shared.device != x.device
            or not shared.is_contiguous()
            or not math.isfinite(float(route_scale))
        ):
            raise ValueError("Invalid shared expert output or route scale")
        n, dim = x.shape
        if n == 0:
            return torch.empty_like(
                x, dtype=x.dtype if shared is not None else torch.float32
            )
        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather import (
            gather_and_combine_local_loop_compatible,
            gather_local_loop_compatible,
        )
        from rtp_llm.ops.compute_ops import per_token_quant_fp8

        dg = self._dg
        a = torch.empty((n, dim), device=x.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty((n, 1), device=x.device, dtype=torch.float32)
        per_token_quant_fp8(x.contiguous(), a, scale)
        indices, weights = indices.contiguous(), weights.contiguous()
        config, counts, experts, tokens, aligned, inverse, ids = (
            dg.moe_align_block_size(a, self._w13, indices.to(torch.int32))
        )
        total = indices.numel()
        up = torch.empty((total, self.inter_local * 2), device=x.device, dtype=x.dtype)
        if shared_schedule is not None:
            shared_schedule.before_up()
        dg.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            (a, scale),
            (self._w13, self._s13),
            up,
            counts,
            experts,
            tokens,
            aligned,
            config,
            inv_scale=self._inverse13,
        )
        if shared_schedule is not None:
            shared_schedule.after_up()
        hidden = dg.activation_and_mul(
            dg.ActivationMode.SILU,
            dg.QuantMode.FP8_CHANNEL,
            up,
            swiglu_limit=self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None,
        )
        down = torch.empty((total, dim), device=x.device, dtype=x.dtype)
        if shared_schedule is not None:
            shared_schedule.before_down()
        dg.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
            hidden,
            (self._w2, self._s2),
            down,
            ids,
            counts,
            inv_scale=self._inverse2,
        )
        inverse = inverse.view_as(indices)
        if shared is not None:
            out = torch.empty_like(shared)
            gather_and_combine_local_loop_compatible(
                down,
                indices,
                weights,
                inverse,
                shared,
                out,
                float(route_scale),
                original_slot_order=True,
            )
        else:
            out = torch.empty_like(x, dtype=torch.float32)
            gather_local_loop_compatible(
                down,
                indices,
                weights,
                inverse,
                out,
                fused=True,
                original_slot_order=True,
            )
        return out
