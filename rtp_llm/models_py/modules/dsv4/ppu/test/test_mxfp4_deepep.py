"""Real PPU DeepEP dispatch/MLP/combine with an analytic MXFP4 oracle.

Run with four or eight exclusively leased visible GPUs. Four-rank diagnostics
do not qualify the eight-rank model or its SGLang throughput target.
"""

import math
import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rtp_llm.device.device_type import is_ppu


@torch.inference_mode()
def run_rank(rank, world, rendezvous):
    from rtp_llm.models_py.distributed.deepep_wrapper import (
        DeepEPWrapper,
        DeepepWrapperConfig,
    )
    from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_fp4 import (
        PpuDeepEPFP4Strategy,
    )
    from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_moe_config import (
        PpuMoeConfig as MoeCfg,
    )
    from rtp_llm.utils.model_weight import W

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    batch, capacity, dim, inter, experts, topk = 80, 128, 4096, 2048, 256, 6
    local = experts // world
    device = torch.device("cuda", rank)
    config = DeepepWrapperConfig(
        ep_rank=rank,
        ep_size=world,
        tp_size=1,
        local_rank=rank,
        world_size=world,
        hidden_size=dim,
        expert_num=experts,
        moe_k=topk,
        deep_ep_num_sm=24,
        use_deepep_low_latency=True,
        use_deepep_internode=False,
        ll_num_max_token=capacity,
        ll_num_max_token_per_rank=capacity,
    )
    DeepEPWrapper.create(config, group=dist.group.WORLD)
    activation = os.environ.get("RTP_PPU_MOE_TEST_ACTIVATION", "fp4")
    strategy_type = PpuDeepEPFP4Strategy
    if activation == "fp8":
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_w4a8 import (
            PpuDeepEPW4A8Strategy,
        )

        strategy_type = PpuDeepEPW4A8Strategy
    elif activation != "fp4":
        raise ValueError("RTP_PPU_MOE_TEST_ACTIVATION must be fp4 or fp8")
    output_dtype = torch.bfloat16 if activation == "fp8" else torch.float32
    strategy = strategy_type(
        MoeCfg(
            layer_id=0,
            dim=dim,
            moe_inter_dim=inter,
            n_routed_experts=experts,
            n_activated_experts=topk,
            swiglu_limit=0.0,
            ep_size=world,
            ep_rank=rank,
            n_local_experts=local,
            local_expert_start=rank * local,
            local_expert_end=(rank + 1) * local,
            max_tokens_per_rank=capacity,
            tp_size=1,
        ),
        expected_m_policy=os.environ.get("RTP_PPU_MOE_TEST_HINT", "capacity"),
        gemm_tile=os.environ.get("RTP_PPU_MOE_TEST_TILE", "auto"),
        output_dtype=output_dtype,
    )
    # Each nibble denotes exactly 1. W1/W3 use 2**-8; W2 uses an
    # expert-specific power of two, making route/EP-shard mistakes observable.
    w1 = torch.full((local, inter, dim // 2), 0x22, dtype=torch.uint8, device=device)
    s1 = torch.full((local, inter, dim // 32), 119, dtype=torch.uint8, device=device)
    if activation == "fp8":
        # Keep the exact same analytic dot product with only 2048 nonzero
        # terms. Continuous mixed-FP8 MMA accumulation of 4096 positive
        # maximum-payload terms loses precision on M890P (also in the
        # reference backend). This test isolates dispatch/combine and Graph
        # state; long-K arithmetic is checked separately against reference.
        w1[..., dim // 4 :].zero_()
        s1.fill_(120)
    w2 = torch.full((local, dim, inter // 2), 0x22, dtype=torch.uint8, device=device)
    s2 = torch.empty((local, dim, inter // 32), dtype=torch.uint8, device=device)
    for expert in range(local):
        s2[expert].fill_(121 + (rank * local + expert) % 4)
    strategy.setup_weights(
        {
            W.v4_routed_w1_w: w1,
            W.v4_routed_w3_w: w1,
            W.v4_routed_w2_w: w2,
            W.v4_routed_w1_s: s1.view(torch.float8_e8m0fnu),
            W.v4_routed_w3_s: s1.view(torch.float8_e8m0fnu),
            W.v4_routed_w2_s: s2.view(torch.float8_e8m0fnu),
        }
    )
    del w1, w2, s1, s2
    x = torch.empty((batch, dim), dtype=torch.bfloat16, device=device)
    indices = torch.empty((batch, topk), dtype=torch.int64, device=device)
    weights = torch.empty((batch, topk), dtype=torch.float32, device=device)

    masked = os.environ.get("RTP_PPU_MOE_TEST_MASK", "0") == "1"
    active_token_mask = torch.ones(batch, dtype=torch.bool, device=device)

    def compute():
        if masked:
            return strategy(x, weights, indices, active_token_mask=active_token_mask)
        return strategy(x, weights, indices)

    def inputs(step, hot, active):
        # W13 outputs 0.25, 0.75, 1 or 3. Masked SwiGLU/MXFP4 outputs
        # 1/32, 3/8, 3/4 or 8; unscaled W2 sums give 1, 12, 24 or 256.
        case = (rank + step) % 4
        x.fill_((1, 3, 4, 12)[case] / 64)
        ids = torch.arange(topk).expand(batch, -1).clone()
        ids += step * 64
        if not hot:
            ids += (torch.arange(batch) + rank * batch)[:, None] * topk
        ids %= experts
        if not masked:
            ids[active:] = -1
        active_token_mask.copy_(torch.arange(batch, device=device) < active)
        route_weights = torch.full((batch, topk), 0.125)
        indices.copy_(ids)
        weights.copy_(route_weights)
        route_weights[active:] = 0
        if not masked:
            weights.copy_(route_weights)
        multipliers = (2.0 ** (ids.clamp_min(0) % 4)) * route_weights
        if activation == "fp8":
            # Constant rows have exact E4M3 payload 448 and a row scale.
            # Quantization therefore preserves the SwiGLU result up to FP32
            # arithmetic, followed by the reference's BF16 W2 boundary.
            gate = (0.25, 0.75, 1.0, 3.0)[case]
            down = 32.0 * gate * gate / (1.0 + math.exp(-gate))
            base = torch.tensor(down, dtype=torch.bfloat16).float().item()
        else:
            base = (1, 12, 24, 256)[case]
        reference = base * multipliers.sum(-1)
        return reference.to(torch.bfloat16).float().to(device)

    def check(output, expected):
        assert output.shape == (batch, dim)
        assert output.dtype == output_dtype
        torch.testing.assert_close(
            output.float(), expected[:, None].expand_as(output), rtol=0, atol=0
        )

    expected = inputs(0, True, batch)
    check(compute(), expected)
    # All four/eight ranks route their 80 tokens to the same six experts:
    # 320/640 valid rows per expert must survive dispatch and combine.
    assert batch * world > 128
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compute()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = compute()
    torch.cuda.current_stream().wait_stream(stream)
    for step, hot, active in (
        (0, True, batch),
        (1, False, max(0, batch - 27 * rank)),
        (2, True, batch),
        (3, True, 0),
        (4, False, batch),
    ):
        expected = inputs(step, hot, active)
        graph.replay()
        check(output, expected)
        dist.barrier()
    torch.cuda.synchronize()
    print(
        f"PASS rank={rank} EP{world}: eager + 5 Graph replays, hot rows={batch*world}",
        flush=True,
    )
    del graph, output, strategy
    DeepEPWrapper.reset()
    dist.destroy_process_group()


class Mxfp4DeepEPTest(unittest.TestCase):
    def test_actual_dispatch_mlp_combine_and_graph(self):
        activation = os.environ.get("RTP_PPU_MOE_TEST_ACTIVATION", "fp4")
        world = int(
            os.environ.get(
                "RTP_PPU_EP_TEST_WORLD_SIZE", "8" if activation == "fp8" else "4"
            )
        )
        if activation == "fp8" and world != 8:
            raise ValueError("W4A8 requires the actual EP8 topology")
        if world not in (4, 8):
            raise ValueError("The model-shape diagnostic requires EP4 or EP8")
        if not torch.cuda.is_available() or torch.cuda.device_count() < world:
            self.skipTest(f"requires {world} exclusively leased PPU GPUs")
        if not is_ppu():
            self.skipTest("requires PPU")
        with tempfile.TemporaryDirectory(prefix="rtp-ppu-deepep-") as directory:
            mp.spawn(
                run_rank,
                args=(world, directory + "/rendezvous"),
                nprocs=world,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
