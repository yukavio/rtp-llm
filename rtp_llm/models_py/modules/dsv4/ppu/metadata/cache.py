"""PPU indexer layout using the existing opaque-cache descriptor API."""

from rtp_llm.models.dsv4_kv_cache import INDEXER_KV_TAG, INDEXER_STATE_TAG
from rtp_llm.ops import OpaqueBlockEntryCountMode


def configure_indexer_cache(model_config, options=None):
    from rtp_llm.models_py.modules.dsv4.ppu.manifest import resolve_indexer_mode

    entry_bytes = 132 if resolve_indexer_mode(options) == "FP8" else 68
    # Called at load time: ModelFactory has now assigned the final MTP depth.
    gamma = int(model_config.gen_num_per_cycle)
    if gamma < 0:
        raise ValueError("PPU indexer cache requires non-negative MTP depth")
    descs = model_config.kv_cache_spec_descs
    for layer in descs:
        for desc in layer:
            if desc.tag == INDEXER_KV_TAG:
                # FP4: 64 packed bytes + four UE8M0 scales.
                # FP8: 128 E4M3 bytes + one FP32 scale.
                desc.entry_elems = entry_bytes
            elif desc.tag == INDEXER_STATE_TAG:
                window = (1 + desc.state_ring_overlap) * desc.compression_ratio
                raw = window + (gamma if desc.state_ring_include_gen_num_per_cycle else 0)
                desc.entry_count_mode = OpaqueBlockEntryCountMode.EXPLICIT
                desc.explicit_entry_count = (raw + 3) // 4 * 4
    # Pybind returns a copy of the descriptor list.
    model_config.kv_cache_spec_descs = descs
