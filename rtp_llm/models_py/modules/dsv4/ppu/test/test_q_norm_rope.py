"""Reference rounding boundary and captured Q/RoPE input lifetimes."""

import unittest

import torch

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.kernels.ppu.attention.ppu_q_norm_rope import q_norm_rope


@unittest.skipUnless(is_ppu(), "requires a PPU")
class QNormRopeTest(unittest.TestCase):
    def inputs(self, shape=(2, 3, 4, 512), phases=6):
        torch.manual_seed(170890)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(phases, 32, device="cuda")
        return q, torch.polar(torch.ones_like(angles), angles)

    def reference(self, q, phases):
        # The model contract materializes BF16 normalization before rotation.
        norm = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        result = norm.clone()
        pairs = torch.view_as_complex(norm[..., -64:].float().reshape(-1, 32, 2))
        rotations = phases.reshape(-1, 32).repeat_interleave(pairs.shape[0] // (phases.numel() // 32), dim=0)
        result[..., -64:] = torch.view_as_real(pairs * rotations).reshape(*q.shape[:-1], 64)
        return result

    def test_rounds_normalization_before_rotation(self):
        q = torch.ones((1, 1, 64, 512), device="cuda", dtype=torch.bfloat16)
        midpoint = torch.tensor(0.501953125, device="cuda")
        c = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
        phases = torch.complex(c.expand(1, 32), c.expand(1, 32)).contiguous()
        result = q_norm_rope(q, phases, 64)
        self.assertTrue(torch.equal(result[..., :448], torch.ones_like(result[..., :448])))
        self.assertTrue(torch.equal(result[..., 448::2], torch.zeros_like(result[..., 448::2])))
        # Keeping normalization in FP32 until after RoPE yields 1.0 here.
        self.assertTrue(torch.equal(result[..., 449::2], torch.full_like(result[..., 449::2], 1.0078125)))

    def test_phase_grouping_and_alias(self):
        for count in (1, 2, 6, 24):
            with self.subTest(phases=count):
                q, phases = self.inputs(phases=count)
                expected = self.reference(q, phases)
                actual = q_norm_rope(q, phases, 64)
                # Independent reductions may straddle one BF16 rounding step.
                torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1 / 128)
                alias = q.clone()
                result = q_norm_rope(alias, phases, 64, out=alias)
                self.assertEqual(result.data_ptr(), alias.data_ptr())
                self.assertTrue(torch.equal(result, actual))

    def test_graph_replay_reads_changed_inputs_and_phases(self):
        q, phases = self.inputs()
        output = torch.empty_like(q)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                q_norm_rope(q, phases, 64, out=output)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q_norm_rope(q, phases, 64, out=output)
        torch.cuda.current_stream().wait_stream(stream)
        pointer = output.data_ptr()
        for _ in range(3):
            q.normal_()
            phases.copy_(torch.polar(torch.ones_like(phases.real), torch.randn_like(phases.real)))
            expected = q_norm_rope(q, phases, 64)
            output.fill_(float("nan"))
            graph.replay()
            self.assertEqual(output.data_ptr(), pointer)
            self.assertTrue(torch.equal(output, expected))

    def test_empty_and_invalid_layouts(self):
        q, phases = self.inputs()
        empty = q[:0]
        self.assertEqual(q_norm_rope(empty, phases, 64).shape, empty.shape)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            q_norm_rope(q.transpose(0, 1), phases, 64)
        with self.assertRaisesRegex(ValueError, "divisible"):
            q_norm_rope(q, phases[:5], 64)
        with self.assertRaisesRegex(ValueError, "divisible"):
            q_norm_rope(q, phases[:0], 64)
        with self.assertRaisesRegex(ValueError, "output must match"):
            q_norm_rope(q, phases, 64, out=q.float())
        with self.assertRaisesRegex(ValueError, "share a device"):
            q_norm_rope(q, phases.cpu(), 64)


if __name__ == "__main__":
    unittest.main()
