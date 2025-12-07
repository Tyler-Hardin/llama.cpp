#include "llama-size-estimator.h"
#include "llama-hparams.h"
#include "llama-impl.h"

#include <cstring>
#include <algorithm>

// Parse layer index from tensor name (e.g., "blk.12.attn_q.weight" -> 12)
int llama_size_estimator::parse_layer_idx(const std::string & tensor_name) {
    int layer_idx = -1;
    sscanf(tensor_name.c_str(), "blk.%d.", &layer_idx);
    return layer_idx;
}

// Check if tensor is a MoE expert weight
bool llama_size_estimator::is_moe_tensor(const std::string & tensor_name) {
    // Matches patterns like: .ffn_gate_exps, .ffn_down_exps, .ffn_up_exps
    return tensor_name.find("_exps") != std::string::npos;
}

// Check if tensor is an attention weight
bool llama_size_estimator::is_attn_tensor(const std::string & tensor_name) {
    return tensor_name.find(".attn_") != std::string::npos ||
           tensor_name.find(".attn.") != std::string::npos;
}

// Check if tensor is a FFN weight (excluding MoE)
bool llama_size_estimator::is_ffn_tensor(const std::string & tensor_name) {
    return (tensor_name.find(".ffn_") != std::string::npos ||
            tensor_name.find(".ffn.") != std::string::npos) &&
           !is_moe_tensor(tensor_name);
}

// Estimate model weight sizes by analyzing the weights_map
llama_size_estimator::model_size_estimate
llama_size_estimator::estimate_model_sizes(const llama_model_loader & ml) {
    model_size_estimate result;

    // First pass: determine number of layers
    int max_layer_idx = -1;
    for (const auto & it : ml.weights_map) {
        int layer_idx = parse_layer_idx(it.first);
        if (layer_idx > max_layer_idx) {
            max_layer_idx = layer_idx;
        }
    }

    result.n_layers = max_layer_idx + 1;
    if (result.n_layers > 0) {
        result.layers.resize(result.n_layers);
        for (int i = 0; i < result.n_layers; i++) {
            result.layers[i].layer_idx = i;
        }
    }

    // Second pass: accumulate sizes
    for (const auto & it : ml.weights_map) {
        const std::string & tensor_name = it.first;
        const ggml_tensor * tensor = it.second.tensor;
        size_t tensor_size = ggml_nbytes(tensor);

        int layer_idx = parse_layer_idx(tensor_name);

        if (layer_idx >= 0 && layer_idx < result.n_layers) {
            // This is a layer-specific tensor
            auto & layer = result.layers[layer_idx];
            layer.total_size += tensor_size;

            if (is_moe_tensor(tensor_name)) {
                layer.moe_size += tensor_size;
                result.total_moe_size += tensor_size;
            } else if (is_attn_tensor(tensor_name)) {
                layer.attn_size += tensor_size;
            } else if (is_ffn_tensor(tensor_name)) {
                layer.ffn_size += tensor_size;
            }
        } else {
            // This is input/output layer (token_embd, output, etc.)
            result.input_output_size += tensor_size;
        }

        result.total_model_size += tensor_size;
    }

    LLAMA_LOG_INFO("%s: estimated %d layers, total model size %.2f MiB (MoE: %.2f MiB)\n",
        __func__, result.n_layers,
        result.total_model_size / 1024.0 / 1024.0,
        result.total_moe_size / 1024.0 / 1024.0);

    return result;
}

// Estimate KV cache size
size_t llama_size_estimator::estimate_kv_cache_size(
        const llama_hparams & hparams,
        uint32_t n_ctx,
        uint32_t n_seq_max,
        ggml_type type_k,
        ggml_type type_v) {

    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    // Size per layer for K and V caches
    const size_t kv_bytes_per_token_k = ggml_type_size(type_k) * n_embd_k_gqa;
    const size_t kv_bytes_per_token_v = ggml_type_size(type_v) * n_embd_v_gqa;

    // Total KV cache size
    size_t kv_cache_size = n_layer * n_ctx * (kv_bytes_per_token_k + kv_bytes_per_token_v);

    // Add overhead for multiple sequences if needed
    if (n_seq_max > 1) {
        // In unified mode, we use one stream; otherwise we need per-sequence overhead
        // This is a conservative estimate
        kv_cache_size = kv_cache_size * std::min(n_seq_max, (uint32_t)4) / 2;
    }

    LLAMA_LOG_INFO("%s: estimated KV cache size: %.2f MiB (n_ctx=%u, n_seq_max=%u)\n",
        __func__, kv_cache_size / 1024.0 / 1024.0, n_ctx, n_seq_max);

    return kv_cache_size;
}

// Estimate overhead (compute graphs, temporary buffers, etc.)
size_t llama_size_estimator::estimate_overhead(
        const llama_hparams & hparams,
        uint32_t n_ctx) {

    // Conservative estimate based on typical compute graph sizes
    // This includes:
    // - Compute graph allocations
    // - Temporary buffers for operations
    // - Backend-specific overhead

    const size_t base_overhead = 512 * 1024 * 1024; // 512 MiB base

    // Scale with context size (larger context = larger compute graphs)
    const size_t ctx_overhead = (n_ctx / 2048) * 128 * 1024 * 1024; // +128 MiB per 2K ctx

    // Scale with model size (wider models need bigger intermediate buffers)
    const size_t model_overhead = (hparams.n_embd / 4096) * 128 * 1024 * 1024; // +128 MiB per 4K embd

    size_t total_overhead = base_overhead + ctx_overhead + model_overhead;

    LLAMA_LOG_INFO("%s: estimated overhead: %.2f MiB\n",
        __func__, total_overhead / 1024.0 / 1024.0);

    return total_overhead;
}

// Calculate optimal number of GPU layers and MoE CPU offload
int llama_size_estimator::calculate_optimal_gpu_layers(
        const model_size_estimate & sizes,
        size_t kv_cache_size,
        size_t overhead,
        size_t available_vram,
        int & n_cpu_moe) {

    n_cpu_moe = 0;

    if (sizes.n_layers == 0) {
        LLAMA_LOG_WARN("%s: no layers detected, cannot optimize\n", __func__);
        return 0;
    }

    LLAMA_LOG_INFO("%s: optimizing for %.2f MiB available VRAM\n",
        __func__, available_vram / 1024.0 / 1024.0);

    // Strategy 1: Try full offload without MoE on CPU
    {
        size_t required = sizes.total_model_size + kv_cache_size + overhead;
        if (required <= available_vram) {
            LLAMA_LOG_INFO("%s: full offload possible (%.2f MiB required)\n",
                __func__, required / 1024.0 / 1024.0);
            return sizes.n_layers;
        }
    }

    // Strategy 2: Try full layer offload with PARTIAL MoE expert offload
    // This is preferred over full MoE on CPU (Strategy 3)
    if (sizes.total_moe_size > 0) {
        // Binary search for the optimal n_cpu_moe value
        // n_cpu_moe=X means layers 0..(X-1) have MoE on CPU, layers X..N have MoE on GPU

        int left = 0;
        int right = sizes.n_layers;
        int best_n_cpu_moe = -1;

        while (left <= right) {
            int mid = (left + right) / 2;

            // Calculate MoE size on CPU for first 'mid' layers
            size_t moe_on_cpu = 0;
            for (int i = 0; i < mid && i < sizes.n_layers; i++) {
                moe_on_cpu += sizes.layers[i].moe_size;
            }

            size_t required = (sizes.total_model_size - moe_on_cpu) + kv_cache_size + overhead;

            if (required <= available_vram) {
                // This config fits! Try to reduce n_cpu_moe further (more experts on GPU)
                best_n_cpu_moe = mid;
                right = mid - 1;
            } else {
                // Doesn't fit, need more MoE on CPU
                left = mid + 1;
            }
        }

        if (best_n_cpu_moe >= 0) {
            // Calculate final size for logging
            size_t moe_on_cpu = 0;
            for (int i = 0; i < best_n_cpu_moe && i < sizes.n_layers; i++) {
                moe_on_cpu += sizes.layers[i].moe_size;
            }
            size_t required = (sizes.total_model_size - moe_on_cpu) + kv_cache_size + overhead;

            LLAMA_LOG_INFO("%s: full layer offload with partial MoE - %d layers with %d layers MoE on CPU (%.2f MiB used, saves %.2f MiB MoE)\n",
                __func__, sizes.n_layers, best_n_cpu_moe,
                required / 1024.0 / 1024.0, moe_on_cpu / 1024.0 / 1024.0);

            n_cpu_moe = best_n_cpu_moe;
            return sizes.n_layers;
        }
    }

    // Strategy 3: Partial layer offload - find maximum number of layers that fit
    // Only reach here if even full layers + all MoE on CPU doesn't work
    size_t kv_per_layer = kv_cache_size / sizes.n_layers;
    size_t fixed_cost = sizes.input_output_size + overhead;
    size_t available_for_layers = available_vram > fixed_cost ? available_vram - fixed_cost : 0;

    int optimal_layers = 0;
    size_t accumulated = 0;

    // Start from the last layer and work backwards (last layers are processed first)
    for (int i = sizes.n_layers - 1; i >= 0; i--) {
        const auto & layer = sizes.layers[i];
        size_t layer_cost = layer.total_size + kv_per_layer;

        if (accumulated + layer_cost <= available_for_layers) {
            accumulated += layer_cost;
            optimal_layers++;
        } else {
            break;
        }
    }

    LLAMA_LOG_INFO("%s: partial layer offload - %d/%d layers (%.2f MiB used)\n",
        __func__, optimal_layers, sizes.n_layers,
        (accumulated + fixed_cost) / 1024.0 / 1024.0);

    return optimal_layers;
}
