//
// Created by keega on 9/19/2021.
//

#ifndef STS_LIGHTSPEED_SCUMSEARCHAGENT2_H
#define STS_LIGHTSPEED_SCUMSEARCHAGENT2_H

#include "game/GameContext.h"
#include "sim/search/Action.h"
#include "sim/search/GameAction.h"

#include <memory>
#include <random>

namespace sts::search {

    class BattleScumSearcher2;
    struct ValueNet;

    struct ScumSearchAgent2 {
        std::int64_t simulationCountTotal;
        std::vector<int> gameActionHistory;

        int stepCount = 0;
        bool paused = false;
        bool pauseOnCardReward = false;

        bool printActions = false;
        bool printLogs = false;
        bool fairRng = false;
        bool searchPotions = true;
        bool skipHallwayPotions = false;  // If true, only search potions for elite/boss fights
        bool heuristicPlayouts = false;
        double explorationParameter = -1;  // -1 = use searcher default
        ValueNet *valueNet = nullptr;  // if set, use value net for combat evaluation
        int valueNetPlayoutTurns = 0;       // if > 0, do partial heuristic playout before value net eval
        bool valueNetGreedy = false;        // if true, use greedy DFS; if false, use value net within MCTS

        int simulationCountBase = 50000;
        double bossSimulationMultiplier = 3;
        int stepsNoSolution = 5;
        int stepsWithSolution = 15;

        std::default_random_engine rng;


        // public interface
        void playout(GameContext &gc);

        // private methods
        void playoutBattle(BattleContext &bc);
        void playoutBattleValueNet(BattleContext &bc);

        void takeAction(GameContext &gc, GameAction a);
        void takeAction(BattleContext &bc, Action a);

        void stepThroughSolution(BattleContext &bc, std::vector<search::Action> &actions);
        void stepThroughSearchTree(BattleContext &bc, const search::BattleScumSearcher2 &s);

        void stepOutOfCombatPolicy(GameContext &gc);
        void cardSelectPolicy(GameContext &gc);
        void stepEventPolicy(GameContext &gc);
        void stepRandom(GameContext &gc);
        void stepRewardsPolicy(GameContext &gc);
        void weightedCardRewardPolicy(GameContext &gc);
    };

}


#endif //STS_LIGHTSPEED_SCUMSEARCHAGENT2_H
