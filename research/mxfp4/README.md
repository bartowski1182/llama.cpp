# Sub-4-bit MXFP4: what the numbers say

Exploratory harness for the question "can we re-compress an MXFP4 model below
4.25 bpw better than Q3_K / Q2_K / IQ\* do?".  Nothing here is proposed for
upstream — it exists to kill bad ideas cheaply before anyone writes a ggml type
and six backend kernels.

> **All numbers below come from synthetic weights.** The sandbox this was run in
> blocks huggingface.co, so no real gpt-oss tensor was measured. Re-run every
> script against an actual MXFP4 GGUF before trusting a single figure — the
> conclusions about *where* the headroom is should survive, the magnitudes may
> not. gpt-oss was QAT'd into MXFP4, so its code histogram is likely flatter
> (more use of the extreme rungs) than post-hoc quantized noise, which would
> make the entropy findings *more* pessimistic, not less.

## The setup

`block_mxfp4` is 32 weights in 17 bytes: one E8M0 exponent byte plus 16 packed
nibbles, each indexing `kvalues_fp4` = `{0,±1,±2,±3,±4,±6,±8,±12}` (E2M1 doubled).
Reference encoder in `ggml/src/ggml-quants.c:350`.

Two facts fall straight out of the layout:

* The 4-bit code is really **1 sign bit + a 3-bit index into an 8-rung magnitude
  ladder**. Index 8 (negative zero) is never emitted — a free escape symbol.
* The scale is `2^(floor(log2 amax) - 2)`, so the largest element of every block
  is clipped to between 75% and 100% of its true value (mean 93.7%). That is the
  MX spec behaving as specified, not a bug, but it means MXFP4 is already leaving
  ~0.5 dB on the table before you touch it.

## Tools

| file | what it does |
| --- | --- |
| `mxfp4_probe.py` | entropy / structure of an MXFP4 source. Where are the wasted bits? |
| `mxfp4_rd.py` | rate–distortion bound: best achievable NMSE at a given bpw, scalar vs VQ |
| `mxfp4_requant_rd.cpp` | what the **real** ggml encoders achieve on the same source |

```sh
python3 research/mxfp4/mxfp4_probe.py --gguf gpt-oss-20b-mxfp4.gguf
python3 research/mxfp4/mxfp4_rd.py    --gguf gpt-oss-20b-mxfp4.gguf

cmake -B build-rd -DCMAKE_BUILD_TYPE=Release && cmake --build build-rd --target ggml -j
c++ -O2 -std=c++17 research/mxfp4/mxfp4_requant_rd.cpp -Iggml/include \
    -Lbuild-rd/bin -lggml -lggml-base -o mxfp4_requant_rd
LD_LIBRARY_PATH=build-rd/bin ./mxfp4_requant_rd
```

Distortion is NMSE against the **dequantised MXFP4 values**, not against some
hypothetical bf16 — for a model that shipped in MXFP4 those values *are* the
weights. Rule of thumb: 6.02 dB ≈ 1 bit/weight.

## Finding 1 — the nibbles are already near-incompressible

Simulated heavy-tailed source, 6.4M weights:

```
H(code)   3.705 bits   (4.000 stored)
H(|code|) 2.842 bits   P(zero) = 0.136
lossless floor  3.705 b/w  ->  3.955 bpw including the raw scale byte
```

The code distribution is within 0.3 bits of uniform. **There is no "likely MXFP4
layout" to build a codebook around.** Concretely, the 8-tuple magnitude
distribution has 686k distinct patterns in 6.4M weights; the top 8192 cover 6.4%
of occurrences. An IQ-style codebook indexed by "common patterns" has nothing to
latch onto — IQ grids work by *projecting* onto a coarse lattice, not by covering
a concentrated distribution, and that stays true here.

So a pure lossless entropy-coding play buys ~7%, and costs you random access
into the row. Not worth a new type.

## Finding 2 — the scale field is 0.1 bpw of pure waste

```
H(e8m0)              3.006 bits  ->  0.094 bpw   (0.250 bpw stored)
per-row delta span   11          ->  4-bit deltas fit
superblock(256w) base + 3-bit deltas: 0.125 bpw, saves 0.125 bpw (2.9% of file)
```

An 8-bit E8M0 per 32 weights carries about 3 bits of information. A 256-weight
superblock holding one base exponent plus eight 3- or 4-bit deltas is **bit-exact
— identical dequantised values — and 2–3% smaller**. This is the only free lunch
on the table, and it's the least interesting one.

## Finding 3 — "make the codebook E2M1-native" is wrong, and I have the receipts

This was the intuition worth testing, and it loses. Forcing reconstruction points
onto the E2M1 rungs is a *constraint*; the optimal reconstruction level for a
bucket is its conditional mean, which is not a lattice point:

```
design                                bpw      NMSE      dB    equiv bits
scalar (DP-optimal), 8 levels       3.125   0.02590  -15.87    +0.00 b
sign-exact, 4 of 8 E2M1 rungs       3.125   0.03786  -14.22    -0.27 b
scalar (DP-optimal), 6 levels       2.710   0.05555  -12.55    +0.00 b
sign-exact, 3 of 8 E2M1 rungs       2.710   0.07910  -11.02    -0.25 b
```

Snapping to the ladder costs about a quarter of a bit. (The scalar baseline here
is solved exactly by DP over the source's atom histogram — Lloyd–Max iterations
get badly stuck on a discrete source and will flatter any competing design by
~2 dB if you let them. That mistake is why this table exists.)

Vector quantisation over the same source gains only what theory says it should —
the 1.53 dB space-filling gain, no more:

```
VQ dim 4, 256 entries               2.125   0.09742  -10.11    +0.20 b
VQ dim 4, 4096 entries              3.125   0.02214  -16.55    +0.11 b
```

**Ceiling for any amount of MXFP4-specific codebook cleverness over a well-tuned
generic scalar quantiser: ~0.25 bits/weight.** Worth knowing before spending a
month on it.

## Finding 4 — this is where the actual headroom is

The bound vs. what ggml's shipped encoders deliver on the same MXFP4 source:

| bpw | best achievable | ggml | gap |
| --- | --- | --- | --- |
| ~2.1 | −8.93 dB | iq2_xxs 2.06 bpw, −7.61 dB | 1.3 dB |
| ~2.7 | −12.55 dB | q2_K 2.63 bpw, −10.44 dB | 2.1 dB |
| ~3.1 | −15.87 dB | iq3_xxs 3.06 bpw, −13.46 dB | 2.4 dB |
| ~3.6 | −20.80 dB | q3_K 3.44 bpw, −15.28 dB | **5.5 dB** |
| ~4.1 | −29.09 dB | iq4_xs 4.25 bpw, −22.00 dB | **7.1 dB** |

The gap explodes above 3 bpw, and the reason is the discreteness this whole
exercise was supposed to exploit — just not where I expected it. Normalised by
per-block amax, the source has only ~15 distinct values (the ladder divided by
whichever rung the block max landed on). Once your alphabet is large enough to
*cover* those atoms, error collapses; 11 levels → −20.8 dB, 16 levels → −29.1 dB,
a cliff of 8 dB for half a bit. Every ggml type ≥3 bpw has a grid tuned for
continuous Gaussian-ish weights and lands between the atoms instead of on them.

The most concrete consequence: `iq4_nl`'s 16-entry LUT is
`{-127,-104,-83,...,113}`, tuned for Gaussian weights, and it scores −22.15 dB
here. A 16-level grid placed optimally for this source scores −29.09 dB in the
same rate class. (I did not build the LUT-swapped `iq4_nl` to confirm the full
7 dB transfers — that is experiment 1 below.) Same bitrate either way, so this
is not a win on its own, but it says
the ~11–12 level / ~3.6 bpw point is where a matched format should sit — roughly
`iq4_xs` quality at 0.6–0.7 fewer bpw.

## Ranked list of things worth actually trying

1. **Retune an existing 3-bit encoder for this source before inventing a
   format.** 5.5 dB is a lot to leave on the floor. `mxfp4_requant_rd.cpp` is the
   loop: swap grids/level counts, re-run, only build a type once something wins.
2. **~11-level matched-grid type at ~3.6 bpw** (pack 5 weights in 18 bits, or 3
   bits + escape using the free `-0` code). Predicted ≈ iq4_xs quality at
   0.6 bpw less. This is the highest-value new-format bet.
3. **Delta-coded E8M0 superblock.** 2–3% bit-exact. Cheap, boring, safe.
4. **Trellis / TCQ (QTIP, EXL3-style) over the matched alphabet.** The honest way
   to bank the 1.53 dB space-filling gain without a giant codebook. This is the
   real "something else altogether", and it is orthogonal to everything above.
5. **Error-feedback rounding (GPTQ/OBQ-style) across the row.** llama.cpp uses
   the imatrix as a diagonal weight only. Compensating quantisation error into
   not-yet-quantised columns is format-independent and is where most modern
   sub-3-bit results actually come from.
6. **Encoder-side, orthogonal:** searching `e ∈ {e−1, e, e+1}` instead of the
   spec's `floor(log2 amax)` rule cuts MXFP4 encode MSE by 5.8% when quantising
   *to* MXFP4 from bf16. Irrelevant for re-compressing an already-MXFP4 file.

## Dead ends, so nobody re-runs them

* Entropy-coding the nibbles: 7% at best, kills random access.
* Per-block adaptive sub-alphabets: blocks use 7.3 of the 8 rungs on average,
  only 0.8% use ≤4. Nothing to adapt to.
* Block-level dedup/dictionary: 0.00% exact duplicate blocks.
* Codebooks whose entries are E2M1 lattice points: −0.25 bits vs free levels.
