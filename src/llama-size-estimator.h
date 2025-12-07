#pragma once

#include "llama-model-loader.h"
#include "llama-hparams.h"
#include "llama-arch.h"
#include "ggml-backend.h"

#include <cstddef>
#include <vector>
#include <string>
#include <regex>

// Estimates memory requirements for model offloading
struct llama_size_estimator {
    struct layer_size_info {
        size_t total_size = 0;
        size_t moe_size = 0;      // Size of MoE expert weights
        size_t attn_size = 0;     // Size of attention weights
        size_t ffn_size = 0;      // Size of FFN weights (excluding MoE)
        int layer_idx = -1;
    };

    struct model_size_estimate {
        std::vector<layer_size_info> layers;
        size_t input_output_size = 0;  // tok_embd, output weights
        size_t total_model_size = 0;
        size_t total_moe_size = 0;
        int n_layers = 0;
    };

    // Estimate model weight sizes from loader metadata
    static model_size_estimate estimate_model_sizes(const llama_model_loader & ml);

    // Estimate KV cache size based on model hyperparameters
    static size_t estimate_kv_cache_size(
        const llama_hparams & hparams,
        uint32_t n_ctx,
        uint32_t n_seq_max,
        ggml_type type_k,
        ggml_type type_v);

    // Estimate overhead (compute buffers, temporary tensors, etc.)
    static size_t estimate_overhead(const llama_hparams & hparams, uint32_t n_ctx);

    // Calculate optimal n_gpu_layers and n_cpu_moe given available VRAM
    // Returns n_gpu_layers, sets n_cpu_moe (0 = all experts on GPU)
    static int calculate_optimal_gpu_layers(
        const model_size_estimate & sizes,
        size_t kv_cache_size,
        size_t overhead,
        size_t available_vram,
        int & n_cpu_moe);  // Output: number of layers to keep MoE on CPU (0 = all on GPU)

private:
    // Helper to classify tensors by layer
    static int parse_layer_idx(const std::string & tensor_name);

    // Helper to identify MoE tensors
    static bool is_moe_tensor(const std::string & tensor_name);

    // Helper to identify attention tensors
    static bool is_attn_tensor(const std::string & tensor_name);

    // Helper to identify FFN tensors
    static bool is_ffn_tensor(const std::string & tensor_name);
};
