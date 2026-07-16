# Temporal model configurations

The `v1` files preserve the original Phase 1 experiments and their checkpoint
metadata. They keep 32 chunk tokens for every K, so their physical history
horizon changes with K and they are not a controlled chunk-size comparison.

The `v2` fixed-chunk files hold the physical history horizon at 64 ticks:

| config | chunk size | memory tokens | history ticks |
| --- | ---: | ---: | ---: |
| `temporal_fixed_k1_v2.json` | 1 | 64 | 64 |
| `temporal_fixed_k4_v2.json` | 4 | 16 | 64 |
| `temporal_fixed_k8_v2.json` | 8 | 8 | 64 |
| `temporal_fixed_k16_v2.json` | 16 | 4 | 64 |

All v2 configurations use Transformer dropout 0.1. Temporal training rejects a
configuration when `sequence_length` is shorter than `chunk_size *
max_chunk_memory`, because that would deploy untrained positional embeddings.

The `*_gru_v2.json` files form the parameter-matched non-Transformer baseline.
They preserve the same U-Net, gated global chunk encoder, token dimension,
chunk size, and 64-tick memory horizon.  Only the cross-chunk memory changes to
a three-layer GRU recomputed over the same bounded token window.

`temporal_ewsc_explicit_k8_v2.json` is the Phase 2 spatial compression model.
It uses bottleneck, visibility, ownership, and army changes to weight frames at
each location, retains weighted mean / final / delta / maximum statistics, and
emits a 4x4 spatial token grid.  Its 4.17M parameter budget is within 4% of the
4.34M global K=8 Transformer model.
