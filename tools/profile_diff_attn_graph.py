#!/usr/bin/env python3
"""Profile baseline attention vs differential-attention-v2 style attention.

Usage example (H100 machine):
  python3 tools/profile_diff_attn_graph.py \
    --device cuda --dtype bf16 --batch 1 --seq-len 2048 --d-model 768 --num-heads 6 \
    --window 768 --iters 50 --warmup 20 --trace-dir traces_attn

Notes:
- This is a microbenchmark harness; it does not import train_gpt.py to avoid distributed side effects.
- It uses SDPA, not flash_attn_varlen_func. Use it for relative graph/cost inspection.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_local_causal_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
    # True means masked-out for SDPA bool mask.
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    causal = j > i
    local = (i - j) >= window
    return causal | local


def split_heads(x: torch.Tensor, h: int, d: int) -> torch.Tensor:
    B, T, _ = x.shape
    return x.view(B, T, h, d).transpose(1, 2).contiguous()  # (B, h, T, d)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, h, T, d = x.shape
    return x.transpose(1, 2).reshape(B, T, h * d).contiguous()


class BaselineAttn(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = split_heads(q, self.num_heads, self.head_dim)
        k = split_heads(k, self.num_heads, self.head_dim)
        v = split_heads(v, self.num_heads, self.head_dim)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
        y = merge_heads(y)
        return self.o(y)


class DiffAttnV2Like(nn.Module):
    """DIFF-v2-like shape pattern: q has 2x heads, k/v keep base heads, then pairwise combine."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q1 = nn.Linear(d_model, d_model, bias=False)
        self.q2 = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.lambda_proj = nn.Linear(d_model, num_heads, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q1 = split_heads(self.q1(x), self.num_heads, self.head_dim)
        q2 = split_heads(self.q2(x), self.num_heads, self.head_dim)
        k = split_heads(self.k(x), self.num_heads, self.head_dim)
        v = split_heads(self.v(x), self.num_heads, self.head_dim)

        q = torch.cat([q1, q2], dim=1)  # (B, 2h, T, d)
        # Repeat KV for grouped-query attention behavior in a backend-agnostic way.
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
        y1, y2 = y.chunk(2, dim=1)

        lam = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1)  # (B, h, T, 1)
        y = y1 - lam * y2
        y = merge_heads(y)
        return self.o(y)


@dataclass
class RunResult:
    name: str
    ms_per_iter: float


def bench(module: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor | None, warmup: int, iters: int) -> float:
    module.train()
    for _ in range(warmup):
        y = module(x, attn_mask)
        (y.square().mean()).backward()
        module.zero_grad(set_to_none=True)
    if x.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = module(x, attn_mask)
        (y.square().mean()).backward()
        module.zero_grad(set_to_none=True)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / iters


def profile_once(name: str, module: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor | None, out_json: str) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        y = module(x, attn_mask)
        (y.square().mean()).backward()

    trace_path = out_json.replace('.json', '.trace.json')
    prof.export_chrome_trace(trace_path)

    events = prof.key_averages(group_by_input_shape=True)
    rows = []
    for e in events:
        rows.append(
            {
                "name": e.key,
                "cpu_time_total_us": float(e.cpu_time_total),
                "cuda_time_total_us": float(getattr(e, "cuda_time_total", 0.0)),
                "self_cuda_time_total_us": float(getattr(e, "self_cuda_time_total", 0.0)),
                "cpu_memory_usage": int(getattr(e, "cpu_memory_usage", 0)),
                "cuda_memory_usage": int(getattr(e, "cuda_memory_usage", 0)),
                "count": int(e.count),
            }
        )

    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == 'bf16':
        return torch.bfloat16
    if s == 'fp16':
        return torch.float16
    if s == 'fp32':
        return torch.float32
    raise ValueError(f"unsupported dtype: {s}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--d-model', type=int, default=768)
    p.add_argument('--num-heads', type=int, default=6)
    p.add_argument('--window', type=int, default=768, help='local causal window size; <=0 means full causal')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--trace-dir', default='traces_attn')
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    os.makedirs(args.trace_dir, exist_ok=True)
    torch.manual_seed(0)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(0)

    x = torch.randn(args.batch, args.seq_len, args.d_model, device=device, dtype=dtype, requires_grad=True)

    attn_mask = None
    if args.window > 0:
        attn_mask = make_local_causal_mask(args.seq_len, args.window, device)
        # SDPA bool mask is broadcastable to (B, h, T, T).
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    baseline = BaselineAttn(args.d_model, args.num_heads).to(device=device, dtype=dtype)
    diffv2 = DiffAttnV2Like(args.d_model, args.num_heads).to(device=device, dtype=dtype)

    # Optional compile for closer runtime behavior.
    if hasattr(torch, 'compile'):
        baseline = torch.compile(baseline, dynamic=False, fullgraph=False)
        diffv2 = torch.compile(diffv2, dynamic=False, fullgraph=False)

    base_ms = bench(baseline, x, attn_mask, warmup=args.warmup, iters=args.iters)
    diff_ms = bench(diffv2, x, attn_mask, warmup=args.warmup, iters=args.iters)

    profile_once('baseline', baseline, x, attn_mask, os.path.join(args.trace_dir, 'baseline_ops.json'))
    profile_once('diff_v2', diffv2, x, attn_mask, os.path.join(args.trace_dir, 'diff_v2_ops.json'))

    ratio = diff_ms / base_ms if base_ms > 0 else math.inf
    print(f'baseline_ms_per_iter={base_ms:.3f}')
    print(f'diff_v2_ms_per_iter={diff_ms:.3f}')
    print(f'diff_over_base={ratio:.3f}x')
    print(f'traces written to: {args.trace_dir}')


if __name__ == '__main__':
    main()
