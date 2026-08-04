// Measures how well the *real* ggml quantizers re-compress an MXFP4 source.
//
// The python harness in this directory says what is theoretically achievable at
// a given bit rate.  This says what the shipped encoders actually achieve, so
// the gap between the two tells you whether a new ggml type is warranted or
// whether an existing encoder is simply mistuned for this source.
//
// Ground truth is the dequantised MXFP4 tensor - for a model that shipped in
// MXFP4 those values *are* the weights, so that is what the error is measured
// against.
//
// Build (after `cmake --build <dir> --target ggml`):
//   c++ -O2 -std=c++17 research/mxfp4/mxfp4_requant_rd.cpp -Iggml/include \
//       -L<dir>/bin -lggml -lggml-base -o mxfp4_requant_rd
//
// Usage:
//   ./mxfp4_requant_rd                      # synthetic heavy-tailed weights
//   ./mxfp4_requant_rd rows.f32 4096 [imatrix.f32]

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> synth_rows(int64_t nrows, int64_t n_per_row, uint32_t seed) {
    // heavy-tailed, per-row scale spread - a stand-in for a real expert tensor
    std::mt19937 rng(seed);
    std::student_t_distribution<float> t(4.0f);
    std::uniform_int_distribution<int> oct(-3, 3);
    std::vector<float> x(nrows * n_per_row);
    for (int64_t r = 0; r < nrows; ++r) {
        const float s = std::ldexp(1.0f, oct(rng));
        for (int64_t j = 0; j < n_per_row; ++j) {
            x[r * n_per_row + j] = s * t(rng);
        }
    }
    return x;
}

static std::vector<float> read_f32(const char * path, size_t n) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return v;
}

// round-trip src through `type` and report weighted NMSE
static void eval(ggml_type type, const std::vector<float> & src, int64_t nrows, int64_t n_per_row,
                 const std::vector<float> & imat, double ref_energy) {
    const auto * tt = ggml_get_type_traits(type);
    if (n_per_row % tt->blck_size != 0) {
        printf("  %-10s  skipped (row %lld not a multiple of block %lld)\n",
               ggml_type_name(type), (long long) n_per_row, (long long) tt->blck_size);
        return;
    }

    const size_t row_bytes = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> q(row_bytes * nrows);
    // iq2_xs / iq2_xxs / iq1_* assert on a null imatrix, so every type gets one;
    // a flat imatrix is just uniform weighting and keeps the comparison honest.
    const float * ip = imat.data();

    ggml_quantize_chunk(type, src.data(), q.data(), 0, nrows, n_per_row, ip);

    std::vector<float> deq(n_per_row);
    double err = 0.0;
    for (int64_t r = 0; r < nrows; ++r) {
        tt->to_float(q.data() + r * row_bytes, deq.data(), n_per_row);
        for (int64_t j = 0; j < n_per_row; ++j) {
            const double d = (double) src[r * n_per_row + j] - (double) deq[j];
            err += ip[j] * d * d;
        }
    }

    const double nmse = err / ref_energy;
    const double bpw  = 8.0 * (double) row_bytes / (double) n_per_row;
    printf("  %-10s  %6.3f bpw   NMSE %10.6f   %7.2f dB\n",
           ggml_type_name(type), bpw, nmse, 10.0 * std::log10(nmse));
}

int main(int argc, char ** argv) {
    int64_t n_per_row = 4096;
    int64_t nrows     = 512;

    std::vector<float> raw;
    std::vector<float> imat;

    if (argc >= 3) {
        n_per_row = std::stoll(argv[2]);
        FILE * f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        fseek(f, 0, SEEK_END);
        const long bytes = ftell(f);
        fclose(f);
        nrows = bytes / (long) (n_per_row * sizeof(float));
        raw = read_f32(argv[1], (size_t) nrows * n_per_row);
        if (argc >= 4) {
            imat = read_f32(argv[3], (size_t) n_per_row);
        }
    } else {
        raw = synth_rows(nrows, n_per_row, 1234);
    }

    // the source: whatever we were given, projected onto MXFP4. this is the
    // reference every candidate is scored against.
    {
        const auto * tt = ggml_get_type_traits(GGML_TYPE_MXFP4);
        const size_t rb = ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
        std::vector<uint8_t> q(rb * nrows);
        ggml_quantize_chunk(GGML_TYPE_MXFP4, raw.data(), q.data(), 0, nrows, n_per_row, nullptr);
        for (int64_t r = 0; r < nrows; ++r) {
            tt->to_float(q.data() + r * rb, raw.data() + r * n_per_row, n_per_row);
        }
    }

    const bool weighted = !imat.empty();
    if (imat.empty()) {
        imat.assign((size_t) n_per_row, 1.0f);
    }

    double ref_energy = 0.0;
    for (int64_t r = 0; r < nrows; ++r) {
        for (int64_t j = 0; j < n_per_row; ++j) {
            const double v = raw[r * n_per_row + j];
            ref_energy += imat[j] * v * v;
        }
    }

    printf("source: %lld rows x %lld, MXFP4 (4.250 bpw), imatrix %s\n\n",
           (long long) nrows, (long long) n_per_row, weighted ? "on" : "flat");

    const ggml_type types[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL,
        GGML_TYPE_Q3_K, GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS,
        GGML_TYPE_Q2_K, GGML_TYPE_IQ2_S,  GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS,
    };
    for (ggml_type t : types) {
        eval(t, raw, nrows, n_per_row, imat, ref_energy);
    }

    ggml_quantize_free();
    return 0;
}
