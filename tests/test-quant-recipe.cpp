#include "llama-quant-recipe.h"

#include <cassert>
#include <cstdio>
#include <string>

static void test_parse_q4_k_m() {
    const char * recipe_text = R"(
name    Q4_K_M
default Q4_K

[output]
  *                              : Q6_K
  arch=falcon                    : Q8_0

[attn_v]
  more_bits                      : Q6_K
  model_type=70B                 : Q5_K
  n_expert>=8                    : Q8_0

[attn_k]
  n_expert>=8                    : Q8_0

[attn_qkv]
  *                              : Q5_K

[ffn_down]
  layer=more_bits                : Q6_K
  arch=falcon, layer<1/16        : Q6_K
  arch=falcon, layer=more_bits   : Q5_K

[attn_output]
  n_expert>=8                    : Q5_K
)";

    quant_recipe recipe = recipe_parse_string(recipe_text);

    assert(recipe.name == "Q4_K_M");
    assert(recipe.default_type == GGML_TYPE_Q4_K);
    assert(recipe.categories.size() == 6);

    // output: 2 rules, first is unconditional default, second overrides for falcon
    auto & output = recipe.categories.at(tensor_category::OUTPUT);
    assert(output.size() == 2);
    assert(output[0].conditions.empty());
    assert(output[0].type == GGML_TYPE_Q6_K);
    assert(output[1].conditions.size() == 1);
    assert(output[1].conditions[0].type == recipe_condition_type::ARCH);
    assert(output[1].conditions[0].str_val == "falcon");
    assert(output[1].type == GGML_TYPE_Q8_0);

    // ffn_down: 3 rules, general default first, falcon overrides after
    auto & ffn = recipe.categories.at(tensor_category::FFN_DOWN);
    assert(ffn.size() == 3);
    assert(ffn[0].conditions.size() == 1);
    assert(ffn[0].conditions[0].type == recipe_condition_type::LAYER);
    assert(ffn[0].conditions[0].more_bits == true);

    assert(ffn[1].conditions.size() == 2);
    assert(ffn[1].conditions[0].type == recipe_condition_type::ARCH);
    assert(ffn[1].conditions[1].type == recipe_condition_type::LAYER);
    assert(ffn[1].conditions[1].cmp == recipe_comparison::LT);
    assert(ffn[1].conditions[1].frac_num == 1);
    assert(ffn[1].conditions[1].frac_den == 16);

    assert(ffn[2].conditions.size() == 2);
    assert(ffn[2].conditions[0].type == recipe_condition_type::ARCH);
    assert(ffn[2].conditions[1].type == recipe_condition_type::LAYER);
    assert(ffn[2].conditions[1].more_bits == true);

    // attn_qkv: unconditional Q5_K
    auto & qkv = recipe.categories.at(tensor_category::ATTENTION_QKV);
    assert(qkv.size() == 1);
    assert(qkv[0].conditions.empty());
    assert(qkv[0].type == GGML_TYPE_Q5_K);

    printf("  PASS: parse_q4_k_m\n");
}

static void test_parse_conditions() {
    const char * recipe_text = R"(
name    test
default Q4_K

[attn_v]
  n_expert>=8            : Q8_0
  index<2                : Q5_K
  n_gqa>=4               : Q4_K
  has_imatrix            : Q6_K
  !has_imatrix           : Q3_K
  arch!=falcon           : Q5_K
  layer<1/8              : Q6_K
  category<1/4           : Q5_K
  category=more_bits     : Q6_K
)";

    quant_recipe recipe = recipe_parse_string(recipe_text);
    auto & rules = recipe.categories.at(tensor_category::ATTENTION_V);
    assert(rules.size() == 9);

    // n_expert>=8
    assert(rules[0].conditions[0].type == recipe_condition_type::N_EXPERT);
    assert(rules[0].conditions[0].cmp == recipe_comparison::GTE);
    assert(rules[0].conditions[0].int_val == 8);

    // index<2
    assert(rules[1].conditions[0].type == recipe_condition_type::INDEX);
    assert(rules[1].conditions[0].cmp == recipe_comparison::LT);
    assert(rules[1].conditions[0].int_val == 2);

    // n_gqa>=4
    assert(rules[2].conditions[0].type == recipe_condition_type::N_GQA);
    assert(rules[2].conditions[0].cmp == recipe_comparison::GTE);
    assert(rules[2].conditions[0].int_val == 4);

    // has_imatrix
    assert(rules[3].conditions[0].type == recipe_condition_type::HAS_IMATRIX);
    assert(rules[3].conditions[0].cmp == recipe_comparison::EQ);

    // !has_imatrix
    assert(rules[4].conditions[0].type == recipe_condition_type::HAS_IMATRIX);
    assert(rules[4].conditions[0].cmp == recipe_comparison::NEG);

    // arch!=falcon
    assert(rules[5].conditions[0].type == recipe_condition_type::ARCH);
    assert(rules[5].conditions[0].cmp == recipe_comparison::NEG);
    assert(rules[5].conditions[0].str_val == "falcon");

    // layer<1/8
    assert(rules[6].conditions[0].type == recipe_condition_type::LAYER);
    assert(rules[6].conditions[0].cmp == recipe_comparison::LT);
    assert(rules[6].conditions[0].frac_num == 1);
    assert(rules[6].conditions[0].frac_den == 8);

    // category<1/4
    assert(rules[7].conditions[0].type == recipe_condition_type::CATEGORY);
    assert(rules[7].conditions[0].cmp == recipe_comparison::LT);
    assert(rules[7].conditions[0].frac_num == 1);
    assert(rules[7].conditions[0].frac_den == 4);

    // category=more_bits
    assert(rules[8].conditions[0].type == recipe_condition_type::CATEGORY);
    assert(rules[8].conditions[0].more_bits == true);

    printf("  PASS: parse_conditions\n");
}

static void test_parse_file() {
    quant_recipe recipe = recipe_parse_file("recipes/Q4_K_M.recipe");

    assert(recipe.name == "Q4_K_M");
    assert(recipe.default_type == GGML_TYPE_Q4_K);
    assert(recipe.categories.size() == 6);

    printf("  PASS: parse_file\n");
}

int main() {
    printf("Running quant recipe tests...\n");

    test_parse_q4_k_m();
    test_parse_conditions();
    test_parse_file();

    printf("All tests passed!\n");
    return 0;
}
