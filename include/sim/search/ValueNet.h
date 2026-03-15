#ifndef STS_LIGHTSPEED_VALUENET_H
#define STS_LIGHTSPEED_VALUENET_H

#include <vector>
#include <string>
#include <array>

namespace sts {
    class BattleContext;
}

namespace sts::search {

    // Feature vector layout (must match Python generate_training_data.py)
    static constexpr int VN_NUM_CARD_IDS = 371;
    static constexpr int VN_NUM_RELIC_IDS = 181;
    static constexpr int VN_MAX_MONSTERS = 5;
    static constexpr int VN_MAX_POTION_SLOTS = 5;
    static constexpr int VN_NUM_PLAYER_STATUSES = 33;

    static constexpr int VN_PLAYER_STATS = 7;
    static constexpr int VN_MONSTER_IDS = 5;
    static constexpr int VN_MONSTER_STATS = 50;  // 5 * 10
    static constexpr int VN_HAND = VN_NUM_CARD_IDS;
    static constexpr int VN_DRAW_PILE = VN_NUM_CARD_IDS;
    static constexpr int VN_DISCARD_PILE = VN_NUM_CARD_IDS;
    static constexpr int VN_EXHAUST_PILE = VN_NUM_CARD_IDS;
    static constexpr int VN_RELICS = VN_NUM_RELIC_IDS;
    static constexpr int VN_POTIONS = VN_MAX_POTION_SLOTS;
    static constexpr int VN_META = 3;

    static constexpr int VN_FEATURE_DIM = VN_PLAYER_STATS + VN_MONSTER_IDS + VN_MONSTER_STATS +
        VN_NUM_PLAYER_STATUSES + VN_HAND + VN_DRAW_PILE + VN_DISCARD_PILE + VN_EXHAUST_PILE +
        VN_RELICS + VN_POTIONS + VN_META;  // = 1768

    struct ValueNet {
        struct Layer {
            std::vector<float> weight;  // [out_features * in_features], row-major
            std::vector<float> bias;    // [out_features]
            int in_features;
            int out_features;
            bool is_output;
        };

        std::vector<Layer> layers;
        std::vector<float> norm_mean;  // [VN_FEATURE_DIM]
        std::vector<float> norm_std;   // [VN_FEATURE_DIM]
        bool normalize = false;
        bool loaded = false;

        // Per-combat context (set once before search begins)
        std::array<float, VN_NUM_RELIC_IDS> relicFlags{};
        int act = 1;

        // Load model from directory containing manifest.json + binary weight files
        bool load(const std::string &model_dir);

        // Set per-combat context that BattleContext doesn't carry
        void setRelicFlags(const std::array<float, VN_NUM_RELIC_IDS> &flags) { relicFlags = flags; }
        void setAct(int a) { act = a; }

        // Extract features from BattleContext into pre-allocated buffer
        void extractFeatures(const BattleContext &bc, float *features) const;

        // Run forward pass: features -> predicted HP remaining
        float forward(const float *features) const;

        // Full pipeline: BattleContext -> predicted HP value (scaled for MCTS)
        double evaluate(const BattleContext &bc) const;
    };

}

#endif // STS_LIGHTSPEED_VALUENET_H
