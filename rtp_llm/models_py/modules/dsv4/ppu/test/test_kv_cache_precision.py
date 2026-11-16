"""KV quantization must not introduce a BF16 rounding before FP8."""

from types import SimpleNamespace
import unittest

import torch

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.kernels.ppu.attention.ppu_qkv_norm import normalize_decode_qkv
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.attention import AttentionFP8


class KVPrecisionDispatchTest(unittest.TestCase):
    def test_target_and_commit_only_opt_in_without_changing_shared_models(self):
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.attention import CommitOnlyAttentionFP8
        from rtp_llm.models_py.modules.dsv4.fp8.attention import (
            AttentionFP8 as Shared, CommitOnlyAttentionFP8 as SharedCommit,
        )

        self.assertEqual(AttentionFP8.kv_cache_source_dtype, torch.float32)
        self.assertEqual(CommitOnlyAttentionFP8.kv_cache_source_dtype, torch.float32)
        self.assertIsNone(Shared.kv_cache_source_dtype)
        self.assertIsNone(SharedCommit.kv_cache_source_dtype)

    def test_attention_consumes_bf16_and_releases_independent_fp32_source(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillQKV

        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(source_dtype=dtype):
                layer = AttentionFP8.__new__(AttentionFP8)
                torch.nn.Module.__init__(layer)
                source = torch.full((2, 512), 1.1875, dtype=dtype)
                qkv = PrefillQKV(qr=torch.empty(2, 1024),
                                q=torch.empty(2, 1, 512, dtype=torch.bfloat16),
                                kv_full=source)
                common = SimpleNamespace(
                    cp_on=False, topk_idxs=torch.zeros(2, 1, dtype=torch.int32),
                    swa_meta=SimpleNamespace(topk_length_kv_full=torch.ones(2, dtype=torch.int32)),
                    freqs_cis=torch.ones(2, 32, dtype=torch.complex64), workspace=None,
                )
                expected = torch.ones(2, 4)

                def attend(**kwargs):
                    kv = kwargs["kv"]
                    self.assertEqual(kv.dtype, torch.bfloat16)
                    self.assertEqual(kv.shape, (2, 1, 512))
                    self.assertTrue(torch.equal(kv, torch.full_like(kv, 1.1875)))
                    # Check the actual consumer boundary, before attention ends.
                    self.assertEqual(source.numel(), 0 if dtype == torch.float32 else 1024)
                    return expected

                layer._flash_mla_sparse_fwd_chunked_projected = attend
                self.assertIs(layer._attn_fp8_swa_via_kv_full(qkv, common), expected)
                self.assertEqual(source.numel(), 0)


@unittest.skipUnless(is_ppu(), "requires a PPU")
class KVCachePrecisionTest(unittest.TestCase):
    def setup_case(self):
        x = torch.ones((1, 1, 512), device="cuda", dtype=torch.bfloat16)
        weight = torch.full((512,), 1.1875, device="cuda", dtype=torch.bfloat16)
        phases = torch.ones((1, 32), device="cuda", dtype=torch.complex64)
        cache = torch.full((1, 256, 584), 0x55, device="cuda", dtype=torch.uint8)
        slots = torch.tensor([3], device="cuda", dtype=torch.long)
        return x, weight, phases, cache, slots

    def check_cache(self, cache, sign=1):
        flat = cache.flatten()
        # Norm of all ones is just below one. 1.1875 * norm is below the
        # FP8 midpoint: quantize to 1.125 (byte 0x79, scale 2^-8). Rounding
        # to BF16 first reaches the midpoint and ties-to-even selects 1.25.
        payload = flat[3 * 576:3 * 576 + 448]
        self.assertTrue(torch.equal(payload, torch.full_like(payload, 0x79 if sign > 0 else 0xF9)))
        scales = flat[256 * 576 + 3 * 8:256 * 576 + 3 * 8 + 7]
        self.assertTrue(torch.equal(scales, torch.full_like(scales, 119)))
        rope = flat[3 * 576 + 448:4 * 576].view(torch.bfloat16)
        self.assertTrue(torch.equal(rope, torch.full_like(rope, sign * 1.1875)))
        self.assertEqual(int(flat[0]), 0x55)

    def layer(self, cache):
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.head_dim, layer.eps = 512, 1e-6
        layer._pool_view_3d_fp8 = lambda _: cache
        layer._swa_cp_byte_sliced = lambda: False
        return layer

    def test_prefill_retains_fp32_until_cache_write(self):
        x, weight, phases, cache, slots = self.setup_case()
        layer = self.layer(cache)
        layer.wkv, layer.wq_a, layer.q_norm = object(), object(), object()
        layer.kv_norm = weight
        layer._can_reuse_qkv_input_quant = lambda: False
        layer._lin = lambda op, value: x if op is layer.wkv else x.expand(1, 1, 512)
        layer._rmsnorm_weighted = lambda value, _: value
        common = SimpleNamespace(rd=64, freqs_cis=phases, cp_on=False,
                                 swa_meta=SimpleNamespace(slot_mapping=slots))
        qkv = layer._prefill_compute_qkv(x.squeeze(0), common)
        self.assertEqual(qkv.kv_full.dtype, torch.float32)
        layer._prefill_write_swa_fp8_paged(common, qkv.kv_full)
        self.check_cache(cache)

    def test_merged_and_separate_decode_share_cache_precision(self):
        x, weight, phases, cache, slots = self.setup_case()
        layer = self.layer(cache)
        metadata = SimpleNamespace(pool_write_slot_mappings={SWA_KV: slots})
        raw = torch.zeros((1, 1, 1536), device="cuda", dtype=torch.bfloat16)
        raw[..., 1024:] = x
        qw = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
        for fused in (False, True):
            cache.fill_(0x55)
            qr, kv = normalize_decode_qkv(
                raw, qw, weight, phases, 1e-6,
                swa_cache=cache if fused else None,
                slot_mapping=slots if fused else None,
            )
            self.assertEqual(qr.dtype, torch.bfloat16)
            self.assertEqual(kv.dtype, torch.float32)
            if not fused:
                layer._decode_write_swa_fp8(kv, 1, 1, metadata)
            self.check_cache(cache)
        cache.fill_(0x55)
        kv = fused_rmsnorm_rope(x, weight, phases, 64, output_dtype=torch.float32)
        layer._decode_write_swa_fp8(kv, 1, 1, metadata)
        self.check_cache(cache)

    def test_graph_reads_current_projection(self):
        x, weight, phases, cache, slots = self.setup_case()
        raw = torch.zeros((1, 1, 1536), device="cuda", dtype=torch.bfloat16)
        raw[..., 1024:] = x
        qw = torch.ones(1024, device="cuda", dtype=torch.bfloat16)

        def run():
            return normalize_decode_qkv(raw, qw, weight, phases, 1e-6,
                                        swa_cache=cache, slot_mapping=slots)

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for sign in (-1, 1, -1):
            raw[..., 1024:].fill_(sign)
            cache.fill_(0x55)
            graph.replay()
            self.check_cache(cache, sign)

    def test_promotion_is_explicit_and_cannot_be_inplace(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as Shared

        x, weight, phases, _, _ = self.setup_case()
        self.assertIsNone(Shared.kv_cache_source_dtype)
        self.assertEqual(fused_rmsnorm_rope(x, weight, phases, 64).dtype, torch.bfloat16)
        empty = x[:0]
        out = fused_rmsnorm_rope(empty, weight, phases, 64, output_dtype=torch.float32)
        self.assertEqual(out.shape, empty.shape)
        self.assertEqual(out.dtype, torch.float32)
        with self.assertRaisesRegex(AssertionError, "inplace cannot promote"):
            fused_rmsnorm_rope(x, weight, phases, 64, output_dtype=torch.float32, inplace=True)


if __name__ == "__main__":
    unittest.main()
