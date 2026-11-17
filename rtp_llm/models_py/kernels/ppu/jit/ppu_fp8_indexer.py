"""FP8 Indexer adapters for the frozen SGLang query and cache kernels."""

from functools import lru_cache

import torch
import triton

from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel


@lru_cache(maxsize=None)
def _module(kind, arch, page_size=64):
    pdl = "true" if arch[0] >= 9 else "false"
    if kind == "q":
        header = "csrc/deepseek_v4/main_norm_rope.cuh"
        symbol = f"FusedQIndexerRopeHadamardQuantKernel<bf16_t,{pdl}>::forward"
    elif kind == "store":
        header = "compat/fused_norm_rope_v2.cuh"
        symbol = f"FusedNormRopeKernel<fp32_t,128,64,{page_size},{pdl}>::forward"
    else:
        raise ValueError(f"Unknown FP8 Indexer kernel: {kind}")
    return load_sglang_kernel(f"fp8_indexer_{kind}_{page_size}", header, symbol, arch)


def quantize_q(q, weights, weight_scale, freqs, positions):
    if q.ndim != 3 or q.shape[-1] != 128 or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("FP8 Indexer Q requires contiguous BF16 [M,H,128]")
    packed = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    scaled_weights = torch.empty((*q.shape[:-1], 1), dtype=torch.float32, device=q.device)
    _module("q", torch.cuda.get_device_capability(q.device)).forward(
        q, packed, weights, scaled_weights, float(weight_scale), freqs, positions
    )
    # The per-query FP8 scale is already folded into the FP32 weights.
    return packed, None, scaled_weights.squeeze(-1)


def norm_rope_store(compressed, plan, weight, eps, freqs, slots, cache, page_size, *, is_decode=False):
    if cache.dtype != torch.uint8 or cache.shape[-1] != 132 or not cache.is_contiguous():
        raise ValueError("FP8 Indexer store requires native contiguous 132B cache")
    if compressed.shape[0]:
        _module("store", torch.cuda.get_device_capability(cache.device), page_size).forward(
            compressed, plan, weight, eps, freqs, slots,
            cache.view(-1, page_size * 132), is_decode, 4,
        )


def gather_k(cache, block_table, cu_kv_seqlens, total, entries):
    from rtp_llm.models_py.kernels.ppu.attention.ppu_fp4_indexer_cache import _gather

    if cache.dtype != torch.uint8 or cache.shape[-1] != 132 or not cache.is_contiguous():
        raise ValueError("FP8 gather requires native contiguous 132B cache")
    payload = torch.empty((total, 128), device=cache.device, dtype=torch.uint8)
    scale_bits = torch.empty((total,), device=cache.device, dtype=torch.int32)
    if total:
        _gather[(triton.cdiv(total, 16),)](
            cache, block_table, cu_kv_seqlens, payload, scale_bits,
            total, cu_kv_seqlens.numel() - 1, block_table.stride(0), entries, 16, 128,
        )
    return payload.view(torch.float8_e4m3fn), scale_bits.view(torch.float32)


def topk_prefill(scores, starts, ends, out):
    from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_prefill_topk
    from rtp_llm.models_py.kernels.ppu.attention.ppu_bf16_radix_topk import canonicalize_topk_indices

    if scores.dtype != torch.float32:
        raise ValueError("FP8 Indexer prefill TopK requires FP32 scores")
    _run_prefill_topk(scores, starts, ends, out, out.shape[-1], 4)
    return canonicalize_topk_indices(out)
