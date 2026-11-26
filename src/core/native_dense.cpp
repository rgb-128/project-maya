#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>

namespace strata::core {
bool NativeDense::hc_q8_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("MAYA_HC_Q8");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

namespace {
bool is_hc_q8_tensor(const strata::TensorInfo& tensor) {
    if (!NativeDense::hc_q8_enabled() || tensor.type != 8) return false;
    const auto& name = tensor.name;
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* hc_suffixes[] = {
        ".hc_attn_down.weight", ".hc_attn_up.weight",
        ".hc_ffn_down.weight", ".hc_ffn_up.weight"
    };
    for (const char* s : hc_suffixes) if (name.ends_with(s)) return true;
    return false;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    if (is_hc_q8_tensor(tensor)) return true;
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS, and Q8_0 (repacked to Q8_0_SOA on P100).
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
int layer_of_tensor(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    return std::atoi(name.c_str() + 4);
}
// the trunk's layers: a GGUF that ships its MTP block counts it into block_count (nextn_predict_layers more) - the
// runner serves the trunk, so blk.<trunk>.* and above are not its (and have no canonical entry)
int trunk_layers(const strata::GgufFile& gguf) {
    const auto* bc = gguf.get("qwen4exp.block_count");
    const auto* nextn = gguf.get("qwen4exp.nextn_predict_layers");
    return bc ? (int) (bc->u - (nextn ? nextn->u : 0)) : INT_MAX;
}
bool in_mtp_block(const strata::TensorInfo& tensor, int trunk) {
    const auto& name = tensor.name;
    if (trunk == INT_MAX || name.rfind("blk.", 0) != 0) return false;
    return std::atoi(name.c_str() + 4) >= trunk;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    int64_t ne0, ne1;
    DevicePtr data;
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const int trunk = trunk_layers(gguf);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && !in_mtp_block(tensor, trunk) &&
                    strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

NativeDense::~NativeDense() {
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int lb, int le, int tp_rank, int tp_size) {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    if (tp_size < 1 || tp_rank < 0 || tp_rank >= tp_size) { err = "native dense: invalid tp_rank/tp_size"; return false; }
    try {
        std::vector<Pending> pending;
        std::vector<uint8_t> shard_stage;
        std::vector<uint8_t> soa_stage;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        int trunk = INT_MAX;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            if (trunk == INT_MAX) trunk = trunk_layers(gguf);
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || in_mtp_block(tensor, trunk)) continue;
                if (le > 0) {
                    const int lyr = layer_of_tensor(tensor.name);
                    if (lyr >= 0 && (lyr < lb || lyr >= le)) continue;
                    if (lyr < 0 && le < trunk) continue;   // output_hc_* only on the last stage
                }
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                const bool hc_q8 = is_hc_q8_tensor(tensor);
                const bool ple_key_q8 = (tensor.name == "blk.1.ple_key.weight" && tensor.type == 8);
                if ((!ref.quantized() && !hc_q8 && !ple_key_q8) || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                int64_t eff_ne0 = ref.ne0;
                int64_t eff_ne1 = ref.ne1;
                int upload_type = (int) tensor.type;
                const void* upload_src = gguf.tensor_data(tensor);

                if (tp_size > 1) {
                    if (tensor.name == "blk.1.ple_key.weight" && tp_rank > 0) continue;
                    int be = 0, bb = 0;
                    if (!strata::block_geometry(tensor.type, be, bb)) {
                        err = "native dense TP: unsupported block geometry for " + tensor.name;
                        return false;
                    }
                    const uint8_t* raw = static_cast<const uint8_t*>(upload_src);
                    const size_t full_row_b = (size_t) (ref.ne0 / be) * (size_t) bb;
                    auto slice_rows_contig = [&](int64_t r0, int64_t nr) {
                        eff_ne1 = nr;
                        shard_stage.resize((size_t) nr * full_row_b);
                        std::memcpy(shard_stage.data(), raw + (size_t) r0 * full_row_b, (size_t) nr * full_row_b);
                        upload_src = shard_stage.data();
                    };
                    auto slice_cols_contig = [&](int64_t c0, int64_t nc) -> bool {
                        if (c0 % be != 0 || nc % be != 0) {
                            err = "native dense TP: column slice not block-aligned for " + tensor.name;
                            return false;
                        }
                        eff_ne0 = nc;
                        const size_t c0_b = (size_t) (c0 / be) * (size_t) bb;
                        const size_t nc_b = (size_t) (nc / be) * (size_t) bb;
                        shard_stage.resize((size_t) ref.ne1 * nc_b);
                        for (int64_t row = 0; row < ref.ne1; ++row) {
                            std::memcpy(shard_stage.data() + (size_t) row * nc_b,
                                        raw + (size_t) row * full_row_b + c0_b, nc_b);
                        }
                        upload_src = shard_stage.data();
                        return true;
                    };

                    if (tensor.name.ends_with(".attn_qkv.weight")) {
                        // [Q(2048) | K(2048) | V(6144)], S=128, HK=16, HV=48, modulo head pairing!
                        const int64_t S = 128, HK = 16, HV = 48;
                        const int64_t hk_l = HK / tp_size, hv_l = HV / tp_size, gqa = HV / HK;
                        eff_ne1 = (2 * hk_l + hv_l) * S;
                        shard_stage.resize((size_t) eff_ne1 * full_row_b);
                        uint8_t* dst = shard_stage.data();
                        std::memcpy(dst, raw + (size_t) ((int64_t) tp_rank * hk_l * S) * full_row_b, (size_t) (hk_l * S) * full_row_b);
                        dst += (size_t) (hk_l * S) * full_row_b;
                        std::memcpy(dst, raw + (size_t) ((HK + (int64_t) tp_rank * hk_l) * S) * full_row_b, (size_t) (hk_l * S) * full_row_b);
                        dst += (size_t) (hk_l * S) * full_row_b;
                        for (int64_t g = 0; g < gqa; ++g) {
                            const int64_t r0 = (2 * HK + g * HK + (int64_t) tp_rank * hk_l) * S;
                            std::memcpy(dst, raw + (size_t) r0 * full_row_b, (size_t) (hk_l * S) * full_row_b);
                            dst += (size_t) (hk_l * S) * full_row_b;
                        }
                        upload_src = shard_stage.data();
                    } else if (tensor.name.ends_with(".attn_gate.weight")) {
                        const int64_t S = 128, HK = 16, HV = 48;
                        const int64_t hk_l = HK / tp_size, hv_l = HV / tp_size, gqa = HV / HK;
                        eff_ne1 = hv_l * S;
                        shard_stage.resize((size_t) eff_ne1 * full_row_b);
                        uint8_t* dst = shard_stage.data();
                        for (int64_t g = 0; g < gqa; ++g) {
                            const int64_t r0 = (g * HK + (int64_t) tp_rank * hk_l) * S;
                            std::memcpy(dst, raw + (size_t) r0 * full_row_b, (size_t) (hk_l * S) * full_row_b);
                            dst += (size_t) (hk_l * S) * full_row_b;
                        }
                        upload_src = shard_stage.data();
                    } else if (tensor.name.ends_with(".ssm_out.weight")) {
                        const int64_t S = 128, HK = 16, HV = 48;
                        const int64_t hk_l = HK / tp_size, hv_l = HV / tp_size, gqa = HV / HK;
                        if ((hk_l * S) % be != 0) { err = "native dense TP: ssm_out block mismatch"; return false; }
                        eff_ne0 = hv_l * S;
                        const size_t chunk_b = (size_t) ((hk_l * S) / be) * (size_t) bb;
                        const size_t new_row_b = (size_t) (eff_ne0 / be) * (size_t) bb;
                        shard_stage.resize((size_t) ref.ne1 * new_row_b);
                        for (int64_t row = 0; row < ref.ne1; ++row) {
                            const uint8_t* srow = raw + (size_t) row * full_row_b;
                            uint8_t* drow = shard_stage.data() + (size_t) row * new_row_b;
                            for (int64_t g = 0; g < gqa; ++g) {
                                const size_t c0_b = (size_t) (((g * HK + (int64_t) tp_rank * hk_l) * S) / be) * (size_t) bb;
                                std::memcpy(drow + (size_t) g * chunk_b, srow + c0_b, chunk_b);
                            }
                        }
                        upload_src = shard_stage.data();
                    } else if (tensor.name.ends_with(".attn_q.weight") ||
                               tensor.name.ends_with(".ffn_gate_shexp.weight") ||
                               tensor.name.ends_with(".ffn_up_shexp.weight")) {
                        const int64_t nr = ref.ne1 / tp_size;
                        slice_rows_contig((int64_t) tp_rank * nr, nr);
                    } else if (tensor.name.ends_with(".attn_k.weight") || tensor.name.ends_with(".attn_v.weight")) {
                        // 2 KV heads total (512 rows): TP=2 -> 1 KV head/rank; TP=4 -> ranks 0,1 = KV0, ranks 2,3 = KV1
                        const int64_t kv_r = ((int64_t) tp_rank * 2) / tp_size;
                        const int64_t nr = ref.ne1 / 2;
                        slice_rows_contig(kv_r * nr, nr);
                    } else if (tensor.name.ends_with(".attn_output.weight") ||
                               tensor.name.ends_with(".ffn_down_shexp.weight")) {
                        const int64_t nc = ref.ne0 / tp_size;
                        if (!slice_cols_contig((int64_t) tp_rank * nc, nc)) return false;
                    }
                }

                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    upload_type, (int) eff_ne0, (int) eff_ne1);
                if (strata::kernels::p100_soa_enabled() &&
                    (upload_type == 8 || upload_type == 14) &&
                    (eff_ne1 % 32 == 0)) {
                    // Wave 6: Pack hc_*_down & output_hc_down as K-Major Q8_0_SOA (108) too!
                    // Eliminates 10,240B uncoalesced strided reads in prefill AND enables 160x320t Split-K in decode.
                    if (soa_stage.size() < bytes) soa_stage.resize(bytes);
                    const int repacked = strata::kernels::native_repack_to_soa_host(
                        upload_type, upload_src, soa_stage.data(), (int) eff_ne0, (int) eff_ne1, false);
                    if (repacked != upload_type) {
                        upload_type = repacked;
                        upload_src = soa_stage.data();
                    }
                }
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), upload_src, bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) eff_ne0);
                total += bytes;
                pending.push_back(Pending{&ref, upload_type, bytes, eff_ne0, eff_ne1, std::move(data)});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            item.ref->ne0 = item.ne0;
            item.ref->ne1 = item.ne1;
            item.ref->elements = item.ne0 * item.ne1;
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core
