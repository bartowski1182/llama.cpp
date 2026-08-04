#!/usr/bin/env python3
"""Structural probe of an MXFP4 weight source.

Answers the question "how much information is actually in an MXFP4 tensor, and
where is it wasted?" before anyone writes a new ggml type.  Run it on a real
MXFP4 GGUF (gpt-oss, etc.) or on synthetic weights when no file is at hand.

    python3 research/mxfp4/mxfp4_probe.py --gguf gpt-oss-20b-mxfp4.gguf
    python3 research/mxfp4/mxfp4_probe.py --simulate

Block layout (ggml-common.h):
    block_mxfp4 { uint8_t e; uint8_t qs[16]; }          # 32 weights, 17 bytes
    qs[j] low nibble  -> element j
    qs[j] high nibble -> element j + 16
    value = kvalues_fp4[code] * 2^(e - 128)             # kvalues are 2x E2M1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# kvalues_fp4 from ggml-common.h (E2M1 doubled). Index 8 is a second encoding of
# zero that quantize_row_mxfp4_ref never emits -- a free escape symbol.
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float64)

QK = 32
BLOCK_BYTES = 1 + QK // 2


def entropy(counts):
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def encode_mxfp4(x):
    """Reference encoder from ggml-quants.c, vectorised. x: (nblocks, 32)."""
    amax = np.abs(x).max(1)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))) - 2 + 127, 0).astype(np.int32)
    d = np.ldexp(1.0, e - 128)  # ggml_e8m0_to_fp32_half
    codes = np.abs(x[:, :, None] - KVALUES[None, None, :] * d[:, None, None]).argmin(2).astype(np.uint8)
    return codes, e


def unpack_gguf(path, max_blocks):
    """Yield (name, codes (nb,32) uint8, e (nb,) int32, n_per_row) per MXFP4 tensor."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    from gguf.gguf_reader import GGUFReader
    from gguf.constants import GGMLQuantizationType

    reader = GGUFReader(path, "r")
    for t in reader.tensors:
        if t.tensor_type != GGMLQuantizationType.MXFP4:
            continue
        raw = t.data.view(np.uint8).reshape(-1, BLOCK_BYTES)
        n_per_row = int(t.shape[0])
        if max_blocks and raw.shape[0] > max_blocks:
            raw = raw[:max_blocks]
        e = raw[:, 0].astype(np.int32)
        qs = raw[:, 1:]
        codes = np.empty((raw.shape[0], QK), dtype=np.uint8)
        codes[:, :16] = qs & 0x0F
        codes[:, 16:] = qs >> 4
        yield t.name, codes, e, n_per_row


def simulate(nblocks, dist):
    rng = np.random.default_rng(0)
    if dist == "gaussian":
        x = rng.standard_normal((nblocks, QK))
    else:
        x = rng.standard_t(4, (nblocks, QK))
    # give each row a different scale so the E8M0 statistics are not degenerate
    x *= np.exp2(rng.integers(-3, 4, (nblocks, 1)).astype(np.float64))
    codes, e = encode_mxfp4(x)
    return codes, e


def report(name, codes, e, n_per_row):
    nb = codes.shape[0]
    mag = codes & 7
    hist = np.bincount(codes.ravel(), minlength=16)
    mhist = np.bincount(mag.ravel(), minlength=8)
    p_zero = mhist[0] / mhist.sum()

    h_code = entropy(hist)
    h_mag = entropy(mhist)
    # a sign bit is only needed for non-zero magnitudes
    floor_bpw = h_mag + (1.0 - p_zero)

    print(f"\n=== {name}   ({nb} blocks, {nb * QK} weights) ===")
    print(f"  code histogram      {np.round(hist / hist.sum(), 4)}")
    print(f"  code 8 (-0) used    {bool(hist[8])}")
    print(f"  H(code)             {h_code:.3f} bits   (stored: 4.000)")
    print(f"  H(|code|)           {h_mag:.3f} bits    P(zero) = {p_zero:.3f}")
    print(f"  lossless floor      {floor_bpw:.3f} b/w  -> {floor_bpw + 0.25:.3f} bpw with the raw scale")

    # --- E8M0 scale field: how much of the 8 stored bits is real information?
    h_e = entropy(np.bincount(e - e.min()))
    per_row = max(1, n_per_row // QK)
    rows = e[: (nb // per_row) * per_row].reshape(-1, per_row) if nb >= per_row else e[None, :]
    delta = rows - rows.min(1, keepdims=True)
    h_delta = entropy(np.bincount(delta.ravel()))
    span = int(delta.max())
    print(f"  H(e8m0)             {h_e:.3f} bits -> {h_e / QK:.4f} bpw (stored 0.2500 bpw)")
    print(f"  per-row delta       H={h_delta:.3f} bits, max span {span}"
          f"  -> {span <= 15 and '4-bit deltas fit' or 'needs escape'}")
    for dbits in (3, 4):
        cost = (8 + dbits * 8) / 256.0  # base + 8 deltas per 256-weight superblock
        print(f"    superblock(256w) base+{dbits}b deltas: {cost:.4f} bpw"
              f"  saves {0.25 - cost:.4f} bpw ({(0.25 - cost) / 4.25 * 100:.1f}% of file)")

    # --- is a per-block sub-alphabet viable?
    sub = mag[: min(nb, 20000)]
    nd = (np.sort(sub, axis=1)[:, 1:] != np.sort(sub, axis=1)[:, :-1]).sum(1) + 1
    print(f"  distinct |code| per block: mean {nd.mean():.2f}, "
          f"P(<=4) = {(nd <= 4).mean():.3f}, P(<=5) = {(nd <= 5).mean():.3f}")

    # --- VQ feasibility: how concentrated are k-tuples?
    for dim in (4, 8):
        g = mag[:, : (QK // dim) * dim].reshape(-1, dim).astype(np.int64)
        key = (g * (8 ** np.arange(dim))).sum(1)
        cnt = np.sort(np.unique(key, return_counts=True)[1])[::-1]
        cum = np.cumsum(cnt) / cnt.sum()
        pts = [(2 ** b, cum[min(2 ** b - 1, len(cum) - 1)]) for b in (8, 11, 13)]
        cov = "  ".join(f"top-{k}: {v * 100:.1f}%" for k, v in pts)
        print(f"  |code| {dim}-tuples: {len(cnt)} distinct, {cov}")

    # --- exact duplicate blocks (dictionary coding sanity check)
    packed = np.ascontiguousarray(codes).view(np.void, ) if False else None
    b = np.ascontiguousarray(codes).reshape(nb, -1)
    uniq = len(np.unique(b.view([('', b.dtype)] * b.shape[1])))
    print(f"  duplicate code blocks: {100 * (1 - uniq / nb):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=str, help="path to an MXFP4 GGUF")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--dist", default="student-t", choices=["gaussian", "student-t"])
    ap.add_argument("--max-blocks", type=int, default=2_000_000)
    ap.add_argument("--max-tensors", type=int, default=4)
    args = ap.parse_args()

    if args.gguf:
        for i, (name, codes, e, npr) in enumerate(unpack_gguf(args.gguf, args.max_blocks)):
            if i >= args.max_tensors:
                break
            report(name, codes, e, npr)
    else:
        codes, e = simulate(min(args.max_blocks, 400_000), args.dist)
        report(f"simulated {args.dist}", codes, e, 4096)


if __name__ == "__main__":
    main()
