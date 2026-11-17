"""FP8 Indexer with PPU compression, stream and speculative-state ownership."""

from rtp_llm.models_py.kernels.ppu.jit import ppu_fp8_indexer as kernels
from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_fp4_indexer import (
    PpuFP4Compressor,
    PpuFP4Indexer,
)


class PpuFP8Compressor(PpuFP4Compressor):
    CACHE_ENTRY_BYTES = 132
    _norm_rope_store = staticmethod(kernels.norm_rope_store)


class PpuFP8Indexer(PpuFP4Indexer):
    CACHE_ENTRY_BYTES = 132
    QUERY_BYTES = 128
    _compressor_factory = PpuFP8Compressor
    _quantize_q = staticmethod(kernels.quantize_q)
    _gather_k = staticmethod(kernels.gather_k)
    _topk_prefill = staticmethod(kernels.topk_prefill)
    _score_symbol = "fp8_mqa_logits"

    def _prefill_score(self, q, qs, k, ks, weights, starts, ends):
        return self._score(q, (k, ks), weights, starts, ends, clean_logits=False)

    def _paged_score(self, q, qs, weights, table, lengths, width):
        return kernels.paged_score(
            q.reshape(-1, 1, self.n_heads, 128),
            weights,
            self._kv_pool_view,
            table,
            lengths,
            width,
        )
