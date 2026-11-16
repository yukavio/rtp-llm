"""W4A8 configuration gates and a constant-expert arithmetic oracle."""

import math
import unittest

import torch
from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.modules.dsv4.ppu.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.models_py.modules.dsv4.ppu.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import PpuModuleProvider


class W4A8ConfigurationTest(unittest.TestCase):
    def test_precision_choice_is_validated_and_snapshotted(self):
        options = {"DSV4_PPU_MOE_ACTIVATION": "fp8"}
        provider = PpuModuleProvider(options)
        options["DSV4_PPU_MOE_ACTIVATION"] = "fp4"
        self.assertEqual(provider._moe_activation, "fp8")
        with self.assertRaises(ValueError):
            PpuModuleProvider({"DSV4_PPU_MOE_ACTIVATION": "fp16"})
        with self.assertRaisesRegex(ValueError, "EP1"):
            provider.build_moe(None, ep_size=8)

    def test_decode_rejects_unsupported_backend_or_topology(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_MOE_ACTIVATION": "fp8"}
        provider = PpuDecodeProvider(options, ep_size=8, tp_size=1)
        self.assertEqual(provider._moe_backend, "deepep")
        for ep, tp, extra in (
            (16, 1, {}),
            (8, 2, {}),
            (8, 1, {"DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}),
            (8, 1, {"DSV4_PPU_MTP_MOE_TILE": "n128"}),
        ):
            with self.subTest(ep=ep, tp=tp, extra=extra), self.assertRaises(ValueError):
                PpuDecodeProvider({**options, **extra}, ep_size=ep, tp_size=tp)


@unittest.skipUnless(
    is_ppu() and "M890P" in torch.cuda.get_device_name(),
    "requires M890P and deep_gemm_w4a8",
)
class W4A8GroupedTest(unittest.TestCase):
    def test_gather_preserves_original_slot_rounding(self):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather import (
            gather_and_combine_local_loop_compatible,
            gather_local_loop_compatible,
        )

        down = (
            torch.tensor(
                [2.0**24, 1.0, -(2.0**24)], device="cuda", dtype=torch.bfloat16
            )[:, None]
            .expand(3, 512)
            .contiguous()
        )
        ids = torch.tensor([[2, 0, 1]], device="cuda")
        inverse = torch.tensor([[0, 1, 2]], device="cuda")
        weights = torch.ones(1, 3, device="cuda")
        out = torch.empty(1, 512, device="cuda")
        gather_local_loop_compatible(down, ids, weights, inverse, out, fused=True)
        self.assertTrue(bool((out == 1).all()))
        gather_local_loop_compatible(
            down, ids, weights, inverse, out, fused=True, original_slot_order=True
        )
        self.assertTrue(bool((out == 0).all()))
        shared = torch.ones(1, 512, device="cuda", dtype=torch.bfloat16)
        combined = torch.empty_like(shared)
        gather_and_combine_local_loop_compatible(
            down, ids, weights, inverse, shared, combined, 1.0, original_slot_order=True
        )
        torch.testing.assert_close(combined, shared, rtol=0, atol=0)

    def test_constant_experts_and_weight_exponent_rebasing(self):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_grouped_w4a8 import (
            PpuGroupedW4A8Strategy,
        )
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_moe_config import PpuMoeConfig
        from rtp_llm.utils.model_weight import W

        e, dim, inter, n = 8, 512, 128, 129
        cfg = PpuMoeConfig(0, dim, inter, e, 2, 2.0, 1, 0, e, 0, e, 256)
        x = torch.ones(n, dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.stack(
            (
                torch.arange(n, device="cuda") % e,
                (torch.arange(n, device="cuda") + 3) % e,
            ),
            dim=1,
        )
        weights = torch.tensor([0.25, 0.75], device="cuda").expand(n, 2).contiguous()
        for exponent in (121, 109):
            with self.subTest(exponent=exponent):

                def packed(rows, cols):
                    return torch.full(
                        (e, rows, cols // 2), 0x22, dtype=torch.uint8, device="cuda"
                    )

                def scales(rows, cols):
                    return torch.full(
                        (e, rows, cols // 32),
                        exponent,
                        dtype=torch.uint8,
                        device="cuda",
                    ).view(torch.float8_e8m0fnu)

                layer = {
                    W.v4_routed_w1_w: packed(inter, dim),
                    W.v4_routed_w3_w: packed(inter, dim),
                    W.v4_routed_w2_w: packed(dim, inter),
                    W.v4_routed_w1_s: scales(inter, dim),
                    W.v4_routed_w3_s: scales(inter, dim),
                    W.v4_routed_w2_s: scales(dim, inter),
                }
                strategy = PpuGroupedW4A8Strategy(cfg)
                strategy.setup_weights(layer)
                self.assertFalse(layer)
                # A constant row quantizes exactly to E4M3 448 with its row scale.
                # Each packed nibble is one, so both matrix sums are analytic.
                weight = 2.0 ** (exponent - 127)
                up = min(2.0, dim * weight)
                activated = up * up / (1.0 + math.exp(-up))
                down = (
                    torch.tensor(inter * weight * activated, dtype=torch.bfloat16)
                    .float()
                    .item()
                )
                actual = strategy(x, weights, ids)
                torch.testing.assert_close(
                    actual, torch.full_like(actual, down), rtol=0, atol=0
                )
                combined = strategy(x, weights, ids, shared=x, route_scale=2.0)
                expected = torch.full_like(x, 2.0 * down + 1.0)
                torch.testing.assert_close(combined, expected, rtol=0, atol=0)
                empty = strategy(x[:0], weights[:0], ids[:0])
                self.assertEqual(empty.shape, (0, dim))
                self.assertEqual(empty.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
