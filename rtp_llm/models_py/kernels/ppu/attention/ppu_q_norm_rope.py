"""PPU V4 Q normalization with the reference BF16 boundary before RoPE."""

from functools import lru_cache

import torch

from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)


@lru_cache(maxsize=None)
def _module(arch):
    return load_sglang_kernel(
        "q_norm_rope",
        "csrc/deepseek_v4/main_norm_rope.cuh",
        "FusedQNormRopeKernel<bf16_t,512,64,false>::forward",
        arch,
    )


def q_norm_rope(q, freqs_cis, rope_head_dim, *, eps=1e-6, out=None):
    """Normalize Q, round to BF16, then rotate using already selected phases.

    Each frequency row covers consecutive flattened Q heads, as in the shared
    operator. Position indices are generated on the submitting stream and are
    captured with the operation; no persistent index cache can outlive a Graph.
    The frozen C++ entry must be warmed before its first Graph capture.
    """
    if not (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[-1] == 512
        and rope_head_dim == 64
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.ndim >= 2
        and freqs_cis.shape[-1] == 32
    ):
        return fused_rmsnorm_rope(
            q, None, freqs_cis, rope_head_dim, eps=eps, out=out
        )
    if not q.is_contiguous() or not freqs_cis.is_contiguous():
        raise ValueError("PPU Q/RoPE requires contiguous inputs")
    if freqs_cis.device != q.device:
        raise ValueError("PPU Q and frequencies must share a device")
    if out is None:
        out = torch.empty_like(q)
    elif (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("PPU Q output must match the input layout, dtype and device")
    if q.numel() == 0:
        return out
    rows = q.numel() // 512
    phases = freqs_cis.numel() // 32
    if phases == 0 or rows % phases:
        raise ValueError("Q head rows must be divisible by the frequency count")
    positions = torch.arange(phases, dtype=torch.int64, device=q.device)
    _module(torch.cuda.get_device_capability(q.device)).forward(
        q.view(phases, rows // phases, 512),
        out.view(phases, rows // phases, 512),
        torch.view_as_real(freqs_cis).view(phases, 64),
        positions,
        eps,
    )
    return out
