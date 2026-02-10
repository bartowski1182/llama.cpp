#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"

#include "../src/llama-model.h"
#include "../src/llama-quant.h"

#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// Helpers: mock model construction
// ---------------------------------------------------------------------------

// Build a minimal llama_model with uniform head counts across layers.
static std::unique_ptr<llama_model> build_mock_model(
        llm_arch arch,
        llm_type type,
        uint32_t n_layer,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_expert = 0) {
    struct llama_model_params mparams = llama_model_default_params();
    auto model = std::make_unique<llama_model>(mparams);

    model->arch = arch;
    model->type = type;

    model->hparams.n_layer  = n_layer;
    model->hparams.n_expert = n_expert;

    for (uint32_t i = 0; i < n_layer; i++) {
        model->hparams.n_head_arr[i]    = n_head;
        model->hparams.n_head_kv_arr[i] = n_head_kv;
    }

    return model;
}

// ---------------------------------------------------------------------------
// Helpers: mock tensor construction
// ---------------------------------------------------------------------------

// Bundle a ggml context and a tensor pointer together so the context lifetime
// outlives the tensor usage.
struct mock_tensor {
    ggml_context_ptr ctx;
    ggml_tensor *    tensor;
};

// Create a named 2D tensor (no memory allocated for data).
static mock_tensor make_mock_tensor(const std::string & name, int64_t ne0, int64_t ne1) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name.c_str());
    return { std::move(ctx), t };
}

// Create a named 3D tensor (for MoE expert tensors, no memory allocated for data).
static mock_tensor make_mock_tensor_3d(const std::string & name, int64_t ne0, int64_t ne1, int64_t ne2) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(t, name.c_str());
    return { std::move(ctx), t };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("test-quant-type-selection: infrastructure OK\n");
    return 0;
}
