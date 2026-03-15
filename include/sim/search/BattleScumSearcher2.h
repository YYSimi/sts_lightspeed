//
// Created by keega on 9/17/2021.
//

#ifndef STS_LIGHTSPEED_BATTLESCUMSEARCHER2_H
#define STS_LIGHTSPEED_BATTLESCUMSEARCHER2_H

#include "sim/search/Action.h"
#include "sim/search/SimpleAgent.h"

#include <functional>
#include <memory>
#include <random>
#include <iostream>
#include <limits>

namespace sts::search {

    struct ValueNet;  // forward declaration

    typedef std::function<double (const BattleContext&)> EvalFnc;

    // to find a solution to a battle with tree pruning
    struct BattleScumSearcher2 {
        class Edge;
        struct Node {
            std::int64_t simulationCount = 0;
            double evaluationSum = 0;
            std::vector<Edge> edges;
        };

        struct Edge {
            Action action;
            Node node;
        };

        // Result of batched greedy search (value net)
        struct GreedyResult {
            std::vector<Action> bestActions;  // best action sequence for this turn
            double bestValue = -1e18;
            int leavesEvaluated = 0;
        };

        std::unique_ptr<const BattleContext> rootState;
        Node root;

        EvalFnc evalFnc;
        double explorationParameter = 3*sqrt(2);

        double bestActionValue = std::numeric_limits<double>::min();
        double minActionValue = std::numeric_limits<double>::max();
        int outcomePlayerHp = 0;

        std::vector<Action> bestActionSequence;
        std::default_random_engine randGen;

        bool fairRng = false;
        bool searchPotions = true;
        bool useHeuristicPlayouts = false;
        const ValueNet *valueNet = nullptr;  // if set, use MLP evaluation
        int valueNetPlayoutTurns = 0;       // if > 0, do partial heuristic playout before value net eval
        std::int64_t simCounter = 0;

        std::vector<Node*> searchStack;
        std::vector<Action> actionStack;

        explicit BattleScumSearcher2(const BattleContext &bc, EvalFnc evalFnc=&evaluateEndState);

        // public methods
        void search(int64_t simulations);
        void step();
        GreedyResult searchBatchedGreedy(int maxDepth = 10, int maxLeaves = 200);

        // private helpers
        void updateFromPlayout(const std::vector<Node*> &stack, const std::vector<Action> &actionStack, const BattleContext &endState);
        void updateFromValueNet(const std::vector<Node*> &stack, const std::vector<Action> &actionStack, const BattleContext &state);
        [[nodiscard]] bool isTerminalState(const BattleContext &bc) const;

        double evaluateEdge(const Node &parent, int edgeIdx);
        int selectBestEdgeToSearch(const Node &cur);
        int selectFirstActionForLeafNode(const Node &leafNode);

        void playoutRandom(BattleContext &state, std::vector<Action> &actionStack);
        void playoutHeuristic(BattleContext &state, std::vector<Action> &actionStack);
        void playoutHybrid(BattleContext &state, std::vector<Action> &actionStack, int maxTurns);

        void enumerateActionsForNode(Node &node, const BattleContext &bc);
        void enumerateCardActions(Node &node, const BattleContext &bc);
        void enumeratePotionActions(Node &node, const BattleContext &bc);
        void enumerateCardSelectActions(Node &node, const BattleContext &bc);
        static double evaluateEndState(const BattleContext &bc);

        // Batched greedy DFS helper
        void dfsEnumerateTurn(const BattleContext &state, std::vector<Action> &actions,
                              GreedyResult &result, int depth, int maxDepth, int maxLeaves);

        void printSearchTree(std::ostream &os, int levels);
        void printSearchStack(std::ostream &os, bool skipLast=false);
    };

    extern thread_local BattleScumSearcher2 *g_debug_scum_search;

}


#endif //STS_LIGHTSPEED_BATTLESCUMSEARCHER2_H
