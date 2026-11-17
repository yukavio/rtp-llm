"""FP8 C4 state/cache equivalence across Graph replay and ring boundaries."""

import dataclasses
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.kernels.ppu.attention.ppu_query_slice import (
    prepare_indexer_query,
)
from rtp_llm.models_py.modules.dsv4.ppu.attention.decode_query import (
    slice_compressor_query,
)
from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_fp8_indexer import (
    PpuFP8Compressor,
)


@dataclasses.dataclass(frozen=True)
class QueryMetadata:
    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    compressed_lens_per_token: torch.Tensor = None
    is_batched: bool = True
    seq_start_per_req: torch.Tensor = None
    cu_seq_per_req: torch.Tensor = None
    has_prefix: bool = True


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class FP8IndexerFusionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(890424)

    def same(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            # Include FP8 payload, signed zero and NaN bits in this comparison.
            # Singleton strided views may be considered contiguous by PyTorch;
            # materialize a flat unit-stride buffer before reinterpreting bytes.
            a = torch.empty(actual.numel(), device=actual.device, dtype=actual.dtype)
            e = torch.empty_like(a)
            a.copy_(actual.reshape(-1))
            e.copy_(expected.reshape(-1))
            self.assertTrue(
                torch.equal(
                    a.view(torch.uint8),
                    e.view(torch.uint8),
                )
            )
        elif dataclasses.is_dataclass(expected):
            for field in dataclasses.fields(expected):
                a, e = getattr(actual, field.name), getattr(expected, field.name)
                if torch.is_tensor(e):
                    self.assertEqual(a.stride(), e.stride())
                self.same(a, e)
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.same(a, e)
        else:
            self.assertEqual(actual, expected)

    @torch.inference_mode()
    def test_c4_state_ring_and_cache(self):
        from rtp_llm.models_py.kernels.ppu.jit.ppu_fp4_indexer import (
            compress4_decode,
        )
        from rtp_llm.models_py.kernels.ppu.jit.ppu_fp8_indexer import (
            compress4_decode_store,
            norm_rope_store,
        )

        for m in (1, 8, 32):
            state = torch.randn((m * 3, 4, 512), device="cuda")
            reference_state = state.clone()
            cache = torch.zeros((m, 64, 132), device="cuda", dtype=torch.uint8)
            reference_cache = cache.clone()
            x = torch.randn((m, 512), device="cuda")
            ape = torch.randn((8, 128), device="cuda")
            weight = torch.rand(128, device="cuda")
            angle = torch.randn((64, 32), device="cuda")
            freqs = torch.view_as_real(
                torch.polar(torch.ones_like(angle), angle)
            ).flatten(-2)
            plan = torch.zeros((m, 4), device="cuda", dtype=torch.int32)
            plan[:, 1] = -1
            slots = torch.full((m,), -1, device="cuda", dtype=torch.int32)

            def old():
                y = compress4_decode(reference_state, x, ape, plan.view(torch.uint8))
                norm_rope_store(
                    y,
                    plan.view(torch.uint8),
                    weight,
                    1e-6,
                    freqs,
                    slots,
                    reference_cache,
                    64,
                    is_decode=True,
                )

            def new():
                self.assertTrue(
                    compress4_decode_store(
                        state,
                        x,
                        ape,
                        plan.view(torch.uint8),
                        weight,
                        1e-6,
                        freqs,
                        slots,
                        cache,
                        64,
                    )
                )

            for _ in range(3):
                old()
                new()
            old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(old_graph):
                old()
            with torch.cuda.graph(new_graph):
                new()
            for seq in range(1, 18):
                current = ((seq - 1) // 4) % 3
                rows = [
                    [
                        seq,
                        r * 12 + (seq - 1) % 12,
                        r * 3 + (current + 2) % 3,
                        r * 3 + current,
                    ]
                    for r in range(m)
                ]
                if m > 1:
                    rows[-1][1] = -1
                plan.copy_(torch.tensor(rows, device="cuda", dtype=torch.int32))
                slots.copy_(
                    torch.arange(m, device="cuda", dtype=torch.int32) * 64 + seq // 4
                )
                if m > 1:
                    slots[-1] = -1
                x.normal_()
                old_graph.replay()
                new_graph.replay()
                self.same(state, reference_state)
                self.same(cache, reference_cache)
            self.assertFalse(
                compress4_decode_store(
                    state,
                    x,
                    ape,
                    plan.view(torch.uint8),
                    weight,
                    1e-6,
                    freqs,
                    slots,
                    cache,
                    128,
                )
            )

    @torch.inference_mode()
    def test_c4_decode_queries_ring_and_cache(self):
        from rtp_llm.models_py.kernels.ppu.jit.ppu_fp8_indexer import (
            compress4_decode_store,
            project_compress4_decode_queries_store,
        )
        from rtp_llm.models_py.kernels.ppu.attention.ppu_fp4_indexer_cache import (
            build_decode_plan,
        )

        for batch, width, entries in (
            (1, 2, 8),
            (3, 3, 12),
            (8, 4, 12),
            (17, 2, 12),
            (32, 4, 8),
            (1, 5, 16),
            (1, 6, 16),
            (16, 6, 16),
            (3, 6, 16),
            (32, 6, 16),
        ):
            for dtype_mask in (0, 31, 26):
                dtypes = [
                    torch.int64 if dtype_mask & (1 << i) else torch.int32
                    for i in range(5)
                ]
                # Permuted physical pages and a padded block-table stride.
                pages = torch.randperm(batch * 3, device="cuda").reshape(batch, 3)
                table = torch.zeros((batch, 5), device="cuda", dtype=dtypes[4])
                table[:, :3] = pages
                table = table[:, :3]
                state = torch.randn((batch * 3 * entries, 512), device="cuda")
                ref_state = state.clone()
                cache = torch.zeros((batch * 3, 8448), device="cuda", dtype=torch.uint8)
                ref_cache = cache.clone()
                x = torch.randn(
                    (batch, width, 4096), device="cuda", dtype=torch.bfloat16
                )
                projection = (
                    torch.randn((512, 4096), device="cuda", dtype=torch.bfloat16) / 64
                )
                ape = torch.randn((8, 128), device="cuda")
                weight = torch.rand(128, device="cuda")
                angle = torch.randn((1024, 32), device="cuda")
                freqs = torch.view_as_real(
                    torch.polar(torch.ones_like(angle), angle)
                ).flatten(-2)
                values = [
                    torch.zeros(batch * width, device="cuda", dtype=d)
                    for d in dtypes[:4]
                ]
                values[2].fill_(-1)
                values[3].fill_(-1)
                meta = QueryMetadata(*values, token_to_req=values[1])
                starts = torch.zeros(batch, device="cuda", dtype=torch.int32)

                def old():
                    for index in range(width):
                        prepared = prepare_indexer_query(x, None, starts, meta, index)
                        if prepared is None:
                            step_x = x[:, index : index + 1].contiguous()
                            step = slice_compressor_query(meta, batch, width, index)
                        else:
                            step_x, _, _, step = prepared
                        projected = torch.mm(
                            step_x.reshape(batch, 4096),
                            projection.t(),
                            out_dtype=torch.float32,
                        )
                        plan, slots = build_decode_plan(step, table, entries, 256)
                        self.assertTrue(
                            compress4_decode_store(
                                ref_state,
                                projected,
                                ape,
                                plan,
                                weight,
                                1e-6,
                                freqs,
                                slots,
                                ref_cache,
                                64,
                            )
                        )

                # Exercise the real compressor consumer and its format dispatch.
                compressor = SimpleNamespace(
                    _fuse_decode_queries=True,
                    _state_pool_3d=state,
                    _kv_pool_view=cache.view(-1, 64, 132),
                    _wkv_wgate_fused=projection,
                    ape=ape,
                    norm=SimpleNamespace(weight=weight),
                    norm_eps=1e-6,
                    freqs_cis=torch.view_as_complex(freqs.view(-1, 32, 2)),
                    _state_block_table=table,
                    _state_eb=entries,
                    _state_tokens_per_block=256,
                    _kv_eb=64,
                    _project_decode_queries_store=PpuFP8Compressor._project_decode_queries_store,
                )

                def new():
                    self.assertTrue(
                        PpuFP8Compressor.try_forward_decode_queries(compressor, x, meta)
                    )

                old()
                new()
                old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(old_graph):
                    old()
                with torch.cuda.graph(new_graph):
                    new()
                requests = (
                    torch.arange(batch, device="cuda")
                    .flip(0)[:, None]
                    .expand(-1, width)
                )
                for start in (
                    0,
                    1,
                    3,
                    7,
                    11,
                    15,
                    23,
                    31,
                    127,
                    251,
                    255,
                    259,
                    507,
                    511,
                    7,
                ):
                    pos = (
                        start
                        + torch.arange(width, device="cuda")[None, :]
                        + torch.arange(batch, device="cuda")[:, None] % 4
                    )
                    physical = table[requests, (pos // 256) % 3].long()
                    writes = physical * entries + pos % entries
                    slots = requests * 192 + pos // 4
                    if batch > 1:
                        # Independent inactive state, position, and cache slots.
                        writes[-1, 0] = -1
                        pos[0, -1] = -1
                        slots[0, 0] = -1
                    for target, source in zip(values, (pos, requests, writes, slots)):
                        target.copy_(source.flatten())
                    x.normal_()
                    old_graph.replay()
                    new_graph.replay()
                    self.same(state, ref_state)
                    self.same(cache, ref_cache)
                before = state.clone(), cache.clone()
                self.assertFalse(
                    project_compress4_decode_queries_store(
                        state,
                        x,
                        projection,
                        ape,
                        weight,
                        1e-6,
                        freqs,
                        meta,
                        table,
                        entries,
                        256,
                        cache,
                        128,
                    )
                )
                self.same(state, before[0])
                self.same(cache, before[1])
                if width > 4:
                    # Longer speculative blocks require the qualified ring;
                    # fallback must happen before either pool is mutated.
                    self.assertFalse(
                        project_compress4_decode_queries_store(
                            state,
                            x,
                            projection,
                            ape,
                            weight,
                            1e-6,
                            freqs,
                            meta,
                            table,
                            12,
                            256,
                            cache,
                            64,
                        )
                    )
                    self.same(state, before[0])
                    self.same(cache, before[1])


if __name__ == "__main__":
    unittest.main()
