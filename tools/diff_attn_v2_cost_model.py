#!/usr/bin/env python3
"""Rough compute/memory cost model for baseline attention vs DIFF Attention V2.

This is a relative model intended for architecture-level tradeoffs, not exact kernel timing.
Assumptions (aligned with train_gpt.py):
- Model width D is fixed.
- Baseline uses QKV projection (3*D output) + O projection (D output).
- DIFF V2 uses extra Q branch (4*D projection total for Q1/Q2/K/V), same O projection,
  plus lambda projection from X -> per-head scalar.
- Attention kernel complexity scales as O(B * T * W * q_heads * head_dim).
  With fixed D and aligned Q/K/V head dim, DIFF V2 approximately doubles this term.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ModelCfg:
    d_model: int = 768
    num_layers: int = 11
    attn_layer_indices: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)  # layer 6 has no attention
    num_heads: int = 6
    seq_len: int = 2048
    batch: int = 1


@dataclass
class StageCfg:
    name: str
    ws_short: int
    ws_long: int


def layer_windows(ws_short: int, ws_long: int) -> Dict[int, int]:
    # Mirrors train_gpt.py bm_sizes for 11 layers.
    # layer 6 has no attention and is omitted from cost sums.
    bm = {
        0: ws_short,
        1: ws_short,
        2: ws_short,
        3: ws_long,
        4: ws_short,
        5: ws_short,
        7: ws_short,
        8: ws_short,
        9: ws_short,
        10: ws_long,
    }
    return bm


def flops_baseline_per_layer(B: int, T: int, D: int, W: int) -> Dict[str, float]:
    # GEMMs use multiply+add = 2 flops convention.
    proj_qkv = 2.0 * B * T * D * (3 * D)
    proj_o = 2.0 * B * T * D * D
    # rough SDPA local-window core: qk + av
    attn_core = 4.0 * B * T * W * D
    total = proj_qkv + proj_o + attn_core
    return {
        "proj_qkv": proj_qkv,
        "proj_o": proj_o,
        "attn_core": attn_core,
        "total": total,
    }


def flops_diffv2_per_layer(B: int, T: int, D: int, W: int, h: int) -> Dict[str, float]:
    proj_qqkv = 2.0 * B * T * D * (4 * D)  # extra Q branch
    proj_o = 2.0 * B * T * D * D
    attn_core = 8.0 * B * T * W * D  # roughly 2x baseline q_heads*head_dim term
    proj_lambda = 2.0 * B * T * D * h  # x -> lambda per token/per head
    combine = 2.0 * B * T * D  # attn1 - sigmoid(lam) * attn2
    total = proj_qqkv + proj_o + attn_core + proj_lambda + combine
    return {
        "proj_qqkv": proj_qqkv,
        "proj_o": proj_o,
        "attn_core": attn_core,
        "proj_lambda": proj_lambda,
        "combine": combine,
        "total": total,
    }


def bytes_baseline_per_layer(B: int, T: int, D: int, dtype_bytes: int = 2) -> Dict[str, float]:
    # Minimal extra activation traffic estimate around attention output tensors.
    # Baseline flash output shape ~ (B, T, D).
    y = B * T * D * dtype_bytes
    return {
        "flash_out_write": y,
        "post_read": y,
        "post_write": y,
        "total": 3 * y,
    }


def bytes_diffv2_per_layer(B: int, T: int, D: int, dtype_bytes: int = 2) -> Dict[str, float]:
    # DIFF V2 flash output shape ~ (B, T, 2D) before pairwise subtraction.
    y2 = B * T * (2 * D) * dtype_bytes
    y = B * T * D * dtype_bytes
    lam = B * T * D * dtype_bytes  # broadcasted lambda math approx
    total = (
        y2 +      # flash write
        y2 +      # read for split/combine
        lam +     # lambda read
        y         # combined output write
    )
    return {
        "flash_out_write": y2,
        "combine_reads": y2 + lam,
        "combine_write": y,
        "total": total,
    }


def summarize_stage(model: ModelCfg, stage: StageCfg) -> None:
    ws = layer_windows(stage.ws_short, stage.ws_long)
    b_flops = 0.0
    d_flops = 0.0
    b_bytes = 0.0
    d_bytes = 0.0

    for layer in model.attn_layer_indices:
        w = ws[layer]
        b_flops += flops_baseline_per_layer(model.batch, model.seq_len, model.d_model, w)["total"]
        d_flops += flops_diffv2_per_layer(model.batch, model.seq_len, model.d_model, w, model.num_heads)["total"]
        b_bytes += bytes_baseline_per_layer(model.batch, model.seq_len, model.d_model)["total"]
        d_bytes += bytes_diffv2_per_layer(model.batch, model.seq_len, model.d_model)["total"]

    print(f"\nStage: {stage.name}")
    print(f"  windows(short,long)=({stage.ws_short},{stage.ws_long})")
    print(f"  baseline attn FLOPs: {b_flops/1e9:.2f} GF")
    print(f"  diff-v2  attn FLOPs: {d_flops/1e9:.2f} GF")
    print(f"  FLOP ratio diff/base: {d_flops/b_flops:.3f}x")
    print(f"  baseline attn extra bytes: {b_bytes/1e6:.2f} MB")
    print(f"  diff-v2  attn extra bytes: {d_bytes/1e6:.2f} MB")
    print(f"  byte ratio diff/base: {d_bytes/b_bytes:.3f}x")


def heads_note(d_model: int) -> None:
    print("\nHead-count note (fixed d_model, aligned q/k/v head dim):")
    print("  q_heads * head_dim = 2 * d_model in DIFF v2, regardless of number of heads.")
    print("  So reducing heads alone does not reduce the dominant DIFF-v2 attention FLOP term.")
    print("  It mostly changes kernel efficiency/quality behavior, not asymptotic compute.")


def main() -> None:
    model = ModelCfg()

    stages: List[StageCfg] = [
        StageCfg("train_stage_1", ws_short=1 * 128, ws_long=3 * 128),
        StageCfg("train_stage_2", ws_short=3 * 128, ws_long=7 * 128),
        StageCfg("train_stage_3", ws_short=5 * 128, ws_long=11 * 128),
        StageCfg("extension", ws_short=6 * 128, ws_long=13 * 128),
    ]

    print("DIFF v2 vs baseline attention cost model")
    print(f"  d_model={model.d_model}, num_heads={model.num_heads}, seq_len={model.seq_len}, batch={model.batch}")
    print(f"  attn_layers={len(model.attn_layer_indices)} / {model.num_layers}")

    for st in stages:
        summarize_stage(model, st)

    heads_note(model.d_model)


if __name__ == "__main__":
    main()
