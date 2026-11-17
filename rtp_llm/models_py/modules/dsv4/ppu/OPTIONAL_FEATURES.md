# DeepSeek V4 Flash PPU optional features

These options belong to the Flash PPU model/provider. They do not change
the shared GPU model, loader, scheduler or CUDA Graph runner. Presence in
this table describes code availability, not hardware qualification.

| Function | Selection | Status |
| --- | --- | --- |
| Decode operator fusion, MTP input fusion, ordered C4 queries | `DSV4_PPU_DECODE_FUSIONS=1` | Optional runtime path, default off |
| MegaMoE | `DSV4_PPU_DECODE_MOE_BACKEND=megamoe` | Optional runtime path; compatible native package required |
| MegaMoE overlap | `DSV4_PPU_MEGAMOE_MODE=plain|streamed|staged` | Default `plain`; `streamed` needs the packaged kernel and falls back on unsupported tiles; `staged` is experimental |
| Decode metadata graph and shared RoPE tables | Always on for q_len=1 decode | Captured on first preparation; replay keeps the runner's host mirrors |
| MTP metadata graph (q_len 2 through 4) and fused compressor state slots | `DSV4_PPU_MTP_METADATA_GRAPH=1` | Optional runtime path, default off |
| Inactive-route masking | `DSV4_PPU_MTP_MASK_INACTIVE=1` with the MTP metadata graph | Optional runtime path, default off |
| Prefix reuse | `DSV4_PPU_PREFIX_REUSE=1` with reuse-cache configuration | Experimental runtime path; correctness still needs PPU qualification |

For q_len=1 decode the PPU provider always builds the metadata graph with
shared RoPE tables; there is no separate reference metadata mode to select
at startup. The graph is captured on its first preparation, before replay
reuses it, and the shared runner keeps its normal host mirrors. q_len 2
through 4 needs `DSV4_PPU_MTP_METADATA_GRAPH=1`; without it the framework
metadata factory is used for verify and draft catch-up, and turning the
option off does not change q_len=1 decode.
MegaMoE requires a compatible PPU DeepGEMM build exposing
`get_symm_buffer_for_mega_moe`, `transform_weights_for_mega_moe`,
`mega_moe_pre_dispatch` and `fp4_fp4_mega_moe`. The strategy uses
`preserve_bf16=True` and separate reduction (`fuse_reduce=False`). Selecting
`DSV4_PPU_MEGAMOE_PACKAGE=deep_gemm_mega` only changes the imported module;
it does not install a package or establish runtime compatibility. Exactly
one overlap mode is active per process: streamed dispatch and staged shared
expert are alternatives, not composable.

### FP8 Indexer

`DSV4_PPU_INDEXER_DTYPE=fp8` selects E4M3 Indexer queries and keys, with
FP32 query weights and scores. The default remains `fp4`. This option is
independent of attention KV precision and routed-expert activation precision.
Set the same value on Prefill and Decode and restart both roles: native
Indexer pages change from 68 to 132 bytes per compressed token, and existing
pages cannot be reused across modes.

The FP8 path retains PPU FP32 compression, chronological speculative-state
updates, stream joins and canonical TopK ordering. With
`DSV4_PPU_DECODE_FUSIONS=1`, compression and cache writes share a kernel.
Ordered multi-query fusion supports up to six speculative queries; five or
six queries require a 16-entry state ring. Unsupported layouts use separate
compression and store kernels. Each query keeps its original FP32 projection
GEMM, preserving its reduction order.

The manual M890P `fp8_indexer_test` target covers cache transfers,
changing-input Graph replay and paged/non-paged score agreement with an exact
integer oracle. `fp8_indexer_fusions_test` compares state and cache bytes across
Graph replay, ring boundaries, rollback, inactive rows and metadata layouts.
These operator checks do not establish model-level accuracy or performance.

### Routed W4A8 experts

`DSV4_PPU_MOE_ACTIVATION=fp8` selects per-token E4M3 activations with the
checkpoint's MXFP4 expert weights. The default remains `fp4`. Prefill requires
EP1; Decode requires EP8/TP1, `DSV4_PPU_DECODE_MOE_BACKEND=deepep` and automatic
GEMM tiles. Both GEMMs produce BF16, including the boundary before SwiGLU.
Unsupported combinations fail during model construction. This option does not
change shared-expert quantization or establish model-level accuracy acceptance.

The optional `deep_gemm_w4a8` dependency coexists with `deep_gemm` and
`deep_gemm_mega`. Build it with the runtime's Python/Torch and selected PPU SDK:

```bash
git clone --recursive https://github.com/t-head/DeepGEMM-for-sail.git deepgemm-source
git -C deepgemm-source checkout 209dd9fa0c85982971f6657a001f97d26602d0c6
git -C deepgemm-source submodule update --init --recursive
python -m pip install setuptools-scm wheel
python tools/ppu_w4a8/build.py --source deepgemm-source \
  --work deepgemm-build --output deepgemm-runtime --cuda-abi-torch
export PYTHONPATH="$PWD/deepgemm-runtime:${PYTHONPATH:-}"
```

The CUDA-ABI flag applies to the Torch 2.11 / SDK 2.2.0 compatibility runtime.
It bridges host device/stream queries without changing device kernels or SDK
headers. Omit it for a native PPU Torch build. The helper copies the pinned
source into a fresh build directory and leaves the existing packages in place.
Run the manual M890P numerical tests in `test/test_w4a8_moe.py`, plus real
EP dispatch/Graph and model comparisons, before qualifying this option.

On an exclusively allocated eight-device M890P worker, the corresponding
manual Bazel suite runs both the single-device arithmetic tests and the real
EP8 test. Missing devices/dependencies or skipped tests fail the run:

```bash
bazel test --config=ppu --local_test_jobs=1 \
  --test_env=PYTHONPATH="$PWD/deepgemm-runtime" \
  //rtp_llm/models_py/modules/dsv4/ppu/test:dsv4_ppu_w4a8_tests
```

The native mixed E4M3/E5M2 MMA does not give full FP32 accumulation accuracy
for every input. On M890P, a K4096 constant maximum-payload test produces
0.94140625 instead of 1 in both the source build and the reference package;
K2048 and power-of-two payload controls are exact. This behavior must remain
visible in numerical reports. The EP8 dispatch/Graph analytic test uses 2048
nonzero columns with doubled scale to isolate communication from this
long-accumulation case. Reference agreement alone is not mathematical accuracy.
