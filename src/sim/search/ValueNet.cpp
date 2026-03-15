#include "sim/search/ValueNet.h"
#include "combat/BattleContext.h"
#include "constants/MonsterEncounters.h"
#include "constants/PlayerStatusEffects.h"

#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

// nlohmann/json for manifest parsing
#include <nlohmann/json.hpp>

namespace sts::search {

// PlayerStatus IDs matching Python generate_training_data.py PLAYER_STATUS_IDS order
static constexpr PlayerStatus STATUS_ORDER[] = {
    PlayerStatus::WEAK,              // 0
    PlayerStatus::VULNERABLE,        // 1
    PlayerStatus::FRAIL,             // 2
    PlayerStatus::INTANGIBLE,        // 3
    PlayerStatus::DRAW_REDUCTION,    // 4
    PlayerStatus::ENTANGLED,         // 5
    PlayerStatus::NO_DRAW,           // 6
    PlayerStatus::WRAITH_FORM,       // 7
    PlayerStatus::BARRICADE,         // 8
    PlayerStatus::CORRUPTION,        // 9
    PlayerStatus::BLUR,              // 10
    PlayerStatus::BUFFER,            // 11
    PlayerStatus::DOUBLE_TAP,        // 12
    PlayerStatus::ECHO_FORM,         // 13
    PlayerStatus::BRUTALITY,         // 14
    PlayerStatus::COMBUST,           // 15
    PlayerStatus::DARK_EMBRACE,      // 16
    PlayerStatus::DEMON_FORM,        // 17
    PlayerStatus::DRAW_CARD_NEXT_TURN,// 18
    PlayerStatus::ENERGIZED,         // 19
    PlayerStatus::EVOLVE,            // 20
    PlayerStatus::FEEL_NO_PAIN,      // 21
    PlayerStatus::FIRE_BREATHING,    // 22
    PlayerStatus::FLAME_BARRIER,     // 23
    PlayerStatus::JUGGERNAUT,        // 24
    PlayerStatus::METALLICIZE,       // 25
    PlayerStatus::PLATED_ARMOR,      // 26
    PlayerStatus::RAGE,              // 27
    PlayerStatus::REGEN,             // 28
    PlayerStatus::RUPTURE,           // 29
    PlayerStatus::THORNS,            // 30
    PlayerStatus::VIGOR,             // 31
    PlayerStatus::ARTIFACT,          // 32
};
static_assert(sizeof(STATUS_ORDER)/sizeof(STATUS_ORDER[0]) == VN_NUM_PLAYER_STATUSES);

static std::vector<float> readBinaryFile(const std::string &path, size_t expectedFloats) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ValueNet: failed to open " << path << std::endl;
        return {};
    }
    std::vector<float> data(expectedFloats);
    f.read(reinterpret_cast<char*>(data.data()), expectedFloats * sizeof(float));
    if (!f) {
        std::cerr << "ValueNet: failed to read " << expectedFloats << " floats from " << path << std::endl;
        return {};
    }
    return data;
}

bool ValueNet::load(const std::string &modelDir) {
    std::string manifestPath = modelDir + "/manifest.json";
    std::ifstream mf(manifestPath);
    if (!mf.is_open()) {
        std::cerr << "ValueNet: failed to open " << manifestPath << std::endl;
        return false;
    }

    nlohmann::json manifest;
    mf >> manifest;

    int inputDim = manifest["input_dim"];
    if (inputDim != VN_FEATURE_DIM) {
        std::cerr << "ValueNet: input_dim mismatch: " << inputDim << " vs " << VN_FEATURE_DIM << std::endl;
        return false;
    }

    // Load layers
    layers.clear();
    for (const auto &layerInfo : manifest["layers"]) {
        Layer layer;
        auto wShape = layerInfo["weight_shape"];
        layer.out_features = wShape[0];
        layer.in_features = wShape[1];
        layer.is_output = layerInfo["is_output"];

        layer.weight = readBinaryFile(modelDir + "/" + layerInfo["weight_file"].get<std::string>(),
                                       layer.out_features * layer.in_features);
        layer.bias = readBinaryFile(modelDir + "/" + layerInfo["bias_file"].get<std::string>(),
                                     layer.out_features);

        if (layer.weight.empty() || layer.bias.empty()) return false;
        layers.push_back(std::move(layer));
    }

    // Load normalization
    normalize = manifest.value("normalize", false);
    if (normalize) {
        norm_mean = readBinaryFile(modelDir + "/norm_mean.bin", VN_FEATURE_DIM);
        norm_std = readBinaryFile(modelDir + "/norm_std.bin", VN_FEATURE_DIM);
        if (norm_mean.empty() || norm_std.empty()) return false;
    }

    loaded = true;
    std::cerr << "ValueNet: loaded " << layers.size() << " layers from " << modelDir << std::endl;
    return true;
}

void ValueNet::extractFeatures(const BattleContext &bc, float *f) const {
    std::memset(f, 0, VN_FEATURE_DIM * sizeof(float));
    int idx = 0;

    // --- Player stats (7) ---
    f[idx+0] = static_cast<float>(bc.player.curHp);
    f[idx+1] = static_cast<float>(bc.player.maxHp);
    f[idx+2] = static_cast<float>(bc.player.block);
    f[idx+3] = static_cast<float>(bc.player.energy);
    f[idx+4] = static_cast<float>(bc.player.strength);
    f[idx+5] = static_cast<float>(bc.player.dexterity);
    f[idx+6] = static_cast<float>(bc.turn);
    idx += VN_PLAYER_STATS;

    // --- Monster IDs (5) ---
    int mc = std::min(bc.monsters.monsterCount, VN_MAX_MONSTERS);
    for (int m = 0; m < mc; ++m) {
        f[idx + m] = static_cast<float>(static_cast<int>(bc.monsters.arr[m].id));
    }
    idx += VN_MONSTER_IDS;

    // --- Monster stats (5 × 10 = 50) ---
    for (int m = 0; m < mc; ++m) {
        const auto &mon = bc.monsters.arr[m];
        int base = idx + m * 10;
        f[base+0] = static_cast<float>(mon.curHp);
        f[base+1] = static_cast<float>(mon.maxHp);
        f[base+2] = static_cast<float>(mon.block);
        f[base+3] = static_cast<float>(mon.strength);
        f[base+4] = static_cast<float>(mon.weak);
        f[base+5] = static_cast<float>(mon.vulnerable);
        f[base+6] = mon.isAlive() ? 1.0f : 0.0f;

        bool attacking = mon.isAttacking();
        f[base+7] = attacking ? 1.0f : 0.0f;
        if (attacking) {
            auto dmgInfo = mon.getMoveBaseDamage(bc);
            f[base+8] = static_cast<float>(dmgInfo.damage);
            f[base+9] = static_cast<float>(dmgInfo.attackCount);
        }
    }
    idx += VN_MONSTER_STATS;

    // --- Player statuses (33) ---
    for (int i = 0; i < VN_NUM_PLAYER_STATUSES; ++i) {
        f[idx + i] = static_cast<float>(bc.player.getStatusRuntime(STATUS_ORDER[i]));
    }
    idx += VN_NUM_PLAYER_STATUSES;

    // --- Hand card counts (371) ---
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        int cardId = static_cast<int>(bc.cards.hand[i].getId());
        if (cardId >= 0 && cardId < VN_NUM_CARD_IDS) {
            f[idx + cardId] += 1.0f;
        }
    }
    idx += VN_HAND;

    // --- Draw pile card counts (371) ---
    for (const auto &card : bc.cards.drawPile) {
        int cardId = static_cast<int>(card.getId());
        if (cardId >= 0 && cardId < VN_NUM_CARD_IDS) {
            f[idx + cardId] += 1.0f;
        }
    }
    idx += VN_DRAW_PILE;

    // --- Discard pile card counts (371) ---
    for (const auto &card : bc.cards.discardPile) {
        int cardId = static_cast<int>(card.getId());
        if (cardId >= 0 && cardId < VN_NUM_CARD_IDS) {
            f[idx + cardId] += 1.0f;
        }
    }
    idx += VN_DISCARD_PILE;

    // --- Exhaust pile card counts (371) ---
    for (const auto &card : bc.cards.exhaustPile) {
        int cardId = static_cast<int>(card.getId());
        if (cardId >= 0 && cardId < VN_NUM_CARD_IDS) {
            f[idx + cardId] += 1.0f;
        }
    }
    idx += VN_EXHAUST_PILE;

    // --- Relic flags (181) ---
    // Use pre-set relic flags (set from GameContext before combat search)
    std::memcpy(&f[idx], relicFlags.data(), VN_NUM_RELIC_IDS * sizeof(float));
    idx += VN_RELICS;

    // --- Potion slots (5) ---
    int pc = std::min(bc.potionCapacity, VN_MAX_POTION_SLOTS);
    for (int i = 0; i < pc && i < bc.potionCount; ++i) {
        f[idx + i] = static_cast<float>(static_cast<int>(bc.potions[i]));
    }
    idx += VN_POTIONS;

    // --- Meta (3): act, floor_num, fight_type ---
    f[idx+0] = static_cast<float>(this->act);
    f[idx+1] = static_cast<float>(bc.floorNum);
    // Fight type: 0=hallway, 1=elite, 2=boss
    if (isBossEncounter(bc.encounter)) f[idx+2] = 2.0f;
    else if (isEliteEncounter(bc.encounter)) f[idx+2] = 1.0f;
    else f[idx+2] = 0.0f;
}

float ValueNet::forward(const float *input) const {
    // Temporary buffers for layer activations
    thread_local std::vector<float> buf_in, buf_out;
    // Cache of nonzero input indices for sparse first layer
    thread_local std::vector<int> nonzeroIdx;

    const float *cur = input;

    // Apply normalization (only first 95 features are affected; rest have mean=0, std=1)
    static constexpr int DENSE_END = 95;
    if (normalize) {
        buf_in.resize(VN_FEATURE_DIM);
        // Dense features: apply normalization
        for (int i = 0; i < DENSE_END; ++i) {
            buf_in[i] = (input[i] - norm_mean[i]) / norm_std[i];
        }
        // Sparse features: copy directly (norm is identity for these)
        std::memcpy(&buf_in[DENSE_END], &input[DENSE_END], (VN_FEATURE_DIM - DENSE_END) * sizeof(float));
        cur = buf_in.data();
    }

    // First layer: sparse optimization (97.7% of inputs are zero)
    {
        const auto &layer = layers[0];
        buf_out.resize(layer.out_features);

        // Find nonzero inputs
        nonzeroIdx.clear();
        for (int i = 0; i < layer.in_features; ++i) {
            if (cur[i] != 0.0f) {
                nonzeroIdx.push_back(i);
            }
        }

        // Sparse matrix multiply: only accumulate nonzero columns
        for (int o = 0; o < layer.out_features; ++o) {
            float sum = layer.bias[o];
            const float *w_row = &layer.weight[o * layer.in_features];
            for (int nzi : nonzeroIdx) {
                sum += w_row[nzi] * cur[nzi];
            }
            buf_out[o] = std::max(0.0f, sum);  // ReLU
        }

        std::swap(buf_in, buf_out);
        cur = buf_in.data();
    }

    // Remaining layers: dense (512->512->512->1, all inputs nonzero after ReLU)
    for (size_t li = 1; li < layers.size(); ++li) {
        const auto &layer = layers[li];
        buf_out.resize(layer.out_features);

        for (int o = 0; o < layer.out_features; ++o) {
            float sum = layer.bias[o];
            const float *w_row = &layer.weight[o * layer.in_features];
            for (int i = 0; i < layer.in_features; ++i) {
                sum += w_row[i] * cur[i];
            }
            buf_out[o] = layer.is_output ? sum : std::max(0.0f, sum);
        }

        std::swap(buf_in, buf_out);
        cur = buf_in.data();
    }

    return cur[0];
}

double ValueNet::evaluate(const BattleContext &bc) const {
    thread_local std::vector<float> features(VN_FEATURE_DIM);
    extractFeatures(bc, features.data());
    float predictedHp = forward(features.data());

    // Scale to match evaluateEndState range:
    // Win: 100 * (35 + hp + potionScore - turn*0.01)
    // The MLP predicts HP remaining (0 = death, >0 = survival with that HP)
    // Convert to same scale as evaluateEndState for compatibility
    double potionScore = bc.potionCount * 4.0;
    if (predictedHp > 0.5) {
        // Predicted survival
        return 100.0 * (35.0 + predictedHp + potionScore - bc.turn * 0.01);
    } else {
        // Predicted death — use the loss formula from evaluateEndState
        // but weight by how negative the prediction is
        return predictedHp * 10.0;  // Will be near zero for death predictions
    }
}

} // namespace sts::search
