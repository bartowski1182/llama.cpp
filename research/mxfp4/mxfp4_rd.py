#!/usr/bin/env python3
"""Rate-distortion bench for re-compressing an MXFP4 source below 4.25 bpw.

The point of this script is to rank codec *designs* against imatrix-weighted
error in minutes, so that only the winners ever get a ggml type, CUDA kernel and
a perplexity run.

Ground truth is the dequantised MXFP4 value, not some hypothetical bf16: for a
model that shipped in MXFP4 (or was QAT'd into it) those values *are* the model.

    python3 research/mxfp4/mxfp4_rd.py --simulate
    python3 research/mxfp4/mxfp4_rd.py --gguf model.gguf --imatrix imatrix.dat

Distortion is reported as NMSE = E[w (t - t_hat)^2] / E[w t^2], i.e. relative
weighted error power.  -1 dB of NMSE is worth roughly 1/6 of a bit, so the
"equiv bits" column tells you how much rate a design is really buying.
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp4_probe import KVALUES, QK, encode_mxfp4, unpack_gguf  # noqa: E402

# the 8 E2M1 magnitudes (doubled), i.e. |kvalues_fp4|
LADDER = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.float64)


def load_source(args):
    """Return u: (nblocks, 32) MXFP4 values normalised by per-block amax."""
    if args.gguf:
        chunks = []
        for i, (_, codes, e, _) in enumerate(unpack_gguf(args.gguf, args.max_blocks)):
            if i >= args.max_tensors:
                break
            chunks.append(KVALUES[codes] * np.ldexp(1.0, e - 128)[:, None])
        t = np.concatenate(chunks)
    else:
        rng = np.random.default_rng(0)
        x = rng.standard_t(4, (args.max_blocks, QK))
        codes, e = encode_mxfp4(x)
        t = KVALUES[codes] * np.ldexp(1.0, e - 128)[:, None]
    amax = np.abs(t).max(1, keepdims=True)
    amax[amax == 0] = 1.0
    return t / amax


def optimal_scalar(v, m):
    """Exactly optimal m-level scalar quantiser.

    The normalised MXFP4 source only takes a few dozen distinct values, so the
    optimal quantiser is a contiguous partition of that atom list and can be
    solved by DP instead of Lloyd iterations (which get stuck badly here).
    """
    a, w = np.unique(v, return_counts=True)
    w = w.astype(np.float64)
    n = len(a)
    if m >= n:
        return a
    # prefix sums -> weighted SSE of any contiguous atom run
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cs = np.concatenate([[0.0], np.cumsum(w * a)])
    cq = np.concatenate([[0.0], np.cumsum(w * a * a)])

    def sse(i, j):  # atoms [i, j)
        cnt = cw[j] - cw[i]
        s = cs[j] - cs[i]
        return (cq[j] - cq[i]) - (s * s / cnt if cnt > 0 else 0.0)

    cost = np.full((m + 1, n + 1), np.inf)
    back = np.zeros((m + 1, n + 1), np.int32)
    cost[0, 0] = 0.0
    for k in range(1, m + 1):
        for j in range(k, n + 1):
            best, arg = np.inf, k - 1
            for i in range(k - 1, j):
                c = cost[k - 1, i] + sse(i, j)
                if c < best:
                    best, arg = c, i
            cost[k, j], back[k, j] = best, arg
    lev, j = [], n
    for k in range(m, 0, -1):
        i = back[k, j]
        cnt = cw[j] - cw[i]
        lev.append((cs[j] - cs[i]) / cnt if cnt > 0 else a[i])
        j = i
    return np.array(sorted(lev))


def apply_levels(u, lev):
    return lev[np.abs(u[:, :, None] - lev[None, None, :]).argmin(2)]


def kmeans(g, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    c = g[rng.choice(len(g), k, replace=False)].copy()
    for _ in range(iters):
        idx = assign(g, c)
        for j in range(k):
            sel = idx == j
            if sel.any():
                c[j] = g[sel].mean(0)
    return c


def assign(g, c):
    cn = (c * c).sum(1)
    out = np.empty(len(g), np.int32)
    for s in range(0, len(g), 16384):
        b = g[s:s + 16384]
        out[s:s + 16384] = (cn[None, :] - 2.0 * b @ c.T).argmin(1)
    return out


def nmse(u, q, w=None):
    if w is None:
        return float(((u - q) ** 2).mean() / (u ** 2).mean())
    return float((w * (u - q) ** 2).sum() / (w * u ** 2).sum())


def best_ladder_subset(u, k):
    """Exhaustively pick the k magnitudes (of the 8 E2M1 rungs) that minimise
    error.  Sign is always preserved exactly.  Rate = 1 + log2(k) bits."""
    a = np.abs(u).ravel()
    a = a[:: max(1, len(a) // 400_000)]
    scale = LADDER / LADDER[-1]
    best, best_err = None, np.inf
    for sub in combinations(range(8), k):
        lv = scale[list(sub)]
        err = ((a[:, None] - lv[None, :]) ** 2).min(1).mean()
        if err < best_err:
            best, best_err = lv, err
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=120_000)
    ap.add_argument("--max-tensors", type=int, default=2)
    ap.add_argument("--scale-bpw", type=float, default=0.125,
                    help="cost of the block scale field (delta-coded E8M0 default)")
    ap.add_argument("--vq-dims", type=int, nargs="*", default=[2, 4])
    args = ap.parse_args()

    u = load_source(args)
    half = len(u) // 2
    tr, te = u[:half], u[half:]
    print(f"source: {len(u)} blocks, {len(u) * QK} weights\n")

    rows = []

    # 1. plain scalar quantiser, levels learned on the MXFP4 source
    for m in (4, 6, 8, 11, 16):
        lev = optimal_scalar(tr.ravel()[:400_000], m)
        rows.append((f"scalar (DP-optimal), {m} levels", np.log2(m) + args.scale_bpw,
                     nmse(te, apply_levels(te, lev))))

    # 2. sign-exact, magnitude restricted to k rungs of the E2M1 ladder
    for k in (2, 3, 4, 6):
        lv = best_ladder_subset(tr, k)
        lv = np.concatenate([-lv[::-1], lv])
        rows.append((f"sign-exact, {k} of 8 E2M1 rungs", 1 + np.log2(k) + args.scale_bpw,
                     nmse(te, apply_levels(te, np.unique(lv)))))

    # 3. vector quantisation over d-tuples (the "IQ codebook" family)
    for d in args.vq_dims:
        for bits in (2.0, 2.5, 3.0):
            k = int(round(2 ** (bits * d)))
            if k > 8192:
                continue
            c = kmeans(tr.reshape(-1, d), k)
            g = te.reshape(-1, d)
            j = assign(g, c)
            rows.append((f"VQ dim {d}, {k} entries", bits + args.scale_bpw,
                         nmse(g, c[j])))

    rows.sort(key=lambda r: (round(r[1], 2), r[2]))
    print(f"{'design':<34}{'bpw':>7}{'NMSE':>10}{'dB':>8}   equiv bits vs best scalar")
    scal = {round(b, 2): n for name, b, n in rows if name.startswith("scalar")}
    for name, bpw, n in rows:
        ref = min((v for kk, v in scal.items() if abs(kk - bpw) < 0.35), default=None)
        eq = "" if ref is None else f"{(10 * np.log10(ref / n)) / 6.02:+.2f} b"
        print(f"{name:<34}{bpw:>7.3f}{n:>10.5f}{10 * np.log10(n):>8.2f}   {eq}")


if __name__ == "__main__":
    main()
