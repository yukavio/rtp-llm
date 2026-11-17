"""FP8 Indexer cache ABI, graph replay and native score contracts on PPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import is_ppu


@unittest.skipUnless(is_ppu(), "requires M890P")
class FP8IndexerTest(unittest.TestCase):
    @torch.inference_mode()
    def test_transferred_pages_preserve_fp8_payload_and_scale_bits(self):
        from rtp_llm.models_py.kernels.ppu.jit.ppu_fp8_indexer import gather_k

        cache = torch.randint(0, 256, (10, 64, 132), device="cuda", dtype=torch.uint8)
        table = torch.tensor(
            [[4, 2, 6], [4, 3, 7], [8, 5, 1]], device="cuda", dtype=torch.int32
        )
        lengths = (63, 65, 129)
        cu = torch.tensor([0, 63, 128, 257], device="cuda", dtype=torch.int32)
        pages = cache.view(10, 64 * 132)
        payload = pages[:, : 64 * 128].reshape(10, 64, 128)
        scales = pages[:, 64 * 128 :].contiguous().view(torch.int32)
        expected_k, expected_s = [], []
        for row, length in enumerate(lengths):
            pos = torch.arange(length, device="cuda")
            block = table[row, pos // 64].long()
            expected_k.append(payload[block, pos % 64])
            expected_s.append(scales[block, pos % 64])
        cache[9].copy_(cache[4])
        table[1, 0] = 9
        k, s = gather_k(cache, table, cu, sum(lengths), 64)
        self.assertTrue(torch.equal(k.view(torch.uint8), torch.cat(expected_k)))
        self.assertTrue(torch.equal(s.view(torch.int32), torch.cat(expected_s)))

    @torch.inference_mode()
    def test_query_replay_uses_updated_inputs_and_positions(self):
        from rtp_llm.models_py.kernels.ppu.jit.ppu_fp8_indexer import quantize_q

        torch.manual_seed(1708)
        for dtype in (torch.int32, torch.int64):
            q = torch.randn(6, 64, 128, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
            angle = torch.randn(64, 32, device="cuda")
            freqs = torch.view_as_real(
                torch.polar(torch.ones_like(angle), angle)
            ).flatten(-2)
            pos = torch.arange(6, device="cuda", dtype=dtype)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    quantize_q(q, w, 128**-0.5 * 64**-0.5, freqs, pos)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = quantize_q(q, w, 128**-0.5 * 64**-0.5, freqs, pos)
            for _ in range(3):
                q.normal_()
                w.normal_()
                pos.add_(7)
                expected = quantize_q(q, w, 128**-0.5 * 64**-0.5, freqs, pos)
                graph.replay()
                self.assertIsNone(actual[1])
                self.assertTrue(
                    torch.equal(
                        actual[0].view(torch.uint8), expected[0].view(torch.uint8)
                    )
                )
                self.assertTrue(torch.equal(actual[2], expected[2]))

    @torch.inference_mode()
    def test_paged_and_prefill_scores_match_integer_oracle(self):
        import deep_gemm
        from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_fp8_indexer import (
            PpuFP8Indexer,
        )

        torch.manual_seed(1708)
        n, heads, batch = 1024, 64, 6
        q = torch.randint(-4, 5, (batch, heads, 128), device="cuda").to(
            torch.float8_e4m3fn
        )
        k = torch.randint(-4, 5, (n, 128), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((n,), 1 / 32, device="cuda")
        weights = torch.full((batch, heads), 1 / 128, device="cuda")
        starts = torch.zeros(batch, device="cuda", dtype=torch.int32)
        lengths = torch.tensor(
            [64, 129, 512, 700, 999, 1024], device="cuda", dtype=torch.int32
        )
        expected = ((q.float() @ k.float().t()).clamp_min(0) * weights[:, :, None]).sum(
            1
        ) * scale
        scores = deep_gemm.fp8_mqa_logits(
            q, (k, scale), weights, starts, lengths, clean_logits=False
        )
        pages = torch.zeros((17, 64 * 132), device="cuda", dtype=torch.uint8)
        order = torch.randperm(16, device="cuda") + 1
        pages[order, : 64 * 128] = k.view(torch.uint8).reshape(16, 64 * 128)
        pages[order, 64 * 128 :] = scale.view(torch.uint8).reshape(16, 64 * 4)
        table = order.to(torch.int32).expand(batch, -1).contiguous()
        indexer = SimpleNamespace(n_heads=heads, _kv_pool_view=pages.view(-1, 64, 132))
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 9)
        ), patch.object(
            deep_gemm, "fp8_paged_mqa_logits", wraps=deep_gemm.fp8_paged_mqa_logits
        ) as native:
            paged = PpuFP8Indexer._paged_score(
                indexer, q, None, weights, table, lengths[:, None], n
            )
        native.assert_called_once()
        self.assertFalse(native.call_args.kwargs["clean_logits"])
        valid = torch.arange(n, device="cuda")[None] < lengths[:, None]
        torch.testing.assert_close(
            scores[:, :n][valid], expected[valid], rtol=0, atol=0
        )
        torch.testing.assert_close(paged[:, :n][valid], expected[valid], rtol=0, atol=0)

        def run():
            return PpuFP8Indexer._paged_score(
                indexer, q, None, weights, table, lengths[:, None], n
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        for _ in range(3):
            q.copy_(torch.randint(-4, 5, q.shape, device="cuda").to(q.dtype))
            lengths.sub_(7).clamp_min_(1)
            graph.replay()
            expected = (
                (q.float() @ k.float().t()).clamp_min(0) * weights[:, :, None]
            ).sum(1) * scale
            valid = torch.arange(n, device="cuda")[None] < lengths[:, None]
            torch.testing.assert_close(
                actual[:, :n][valid], expected[valid], rtol=0, atol=0
            )

    @torch.inference_mode()
    def test_prefill_topk_windows_stride_and_ties(self):
        from rtp_llm.models_py.kernels.ppu.jit.ppu_fp8_indexer import topk_prefill

        torch.manual_seed(1708)
        # 3072/3073 crosses the fast/radix prefill branch threshold.
        for n in (1024, 3072, 3073, 4096):
            for tied in (False, True):
                with self.subTest(columns=n, tied=tied):
                    scores = torch.empty((6, n + 256), device="cuda")[:, :n]
                    for row in range(6):
                        values = (
                            torch.zeros(n, device="cuda")
                            if tied
                            else torch.randperm(n, device="cuda").float()
                        )
                        scores[row].copy_(values)
                    starts = torch.tensor(
                        [0, 13, 31, 7, 29, 3], device="cuda", dtype=torch.int32
                    )
                    lengths = (0, 1, 511, 512, 513, n - 64)
                    ends = starts + torch.tensor(
                        lengths, device="cuda", dtype=torch.int32
                    )
                    out = torch.empty((6, 512), device="cuda", dtype=torch.int32)
                    topk_prefill(scores, starts, ends, out)
                    for row, length in enumerate(lengths):
                        selected = out[row][out[row] >= 0]
                        count = min(length, 512)
                        self.assertEqual(selected.numel(), count)
                        self.assertEqual(selected.unique().numel(), count)
                        self.assertTrue(torch.equal(selected, selected.sort().values))
                        self.assertTrue(torch.all(out[row, count:] == -1))
                        if count:
                            self.assertGreaterEqual(int(selected.min()), 0)
                            self.assertLess(int(selected.max()), length)
                            if not tied:
                                start = int(starts[row])
                                expected = (
                                    scores[row, start : start + length]
                                    .topk(count)
                                    .indices.sort()
                                    .values
                                )
                                self.assertTrue(torch.equal(selected.long(), expected))


if __name__ == "__main__":
    unittest.main()
