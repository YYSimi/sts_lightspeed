//
// Created by keega on 9/16/2021.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/search/BattleScumSearcher2.h"
#include "sim/search/ValueNet.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "constants/MonsterEncounters.h"

#include "slaythespire.h"


using namespace sts;

// ---- Step-by-step battle wrapper ----
// Wraps BattleContext for Python access to combat decisions.
struct PyBattleContext {
    BattleContext bc;

    explicit PyBattleContext(GameContext &gc) {
        bc.init(gc);
        bc.executeActions(); // process initial draw/relic triggers → PLAYER_NORMAL
    }

    void exitBattle(GameContext &gc) {
        bc.exitBattle(gc);
    }

    // Play card at hand_idx, targeting monster at target_idx.
    // If card doesn't require a target, target_idx is ignored.
    // If card requires a target and target_idx == -1, auto-picks first targetable.
    void playCard(int handIdx, int targetIdx) {
        const auto card = bc.cards.hand[handIdx]; // copy before queue alters hand
        int target = targetIdx;
        if (target == -1) {
            if (card.requiresTarget()) {
                for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                    if (bc.monsters.arr[i].isTargetable()) { target = i; break; }
                }
            } else {
                target = 0;
            }
        }
        bc.addToBotCard(CardQueueItem(card, target, bc.player.energy));
        bc.setState(InputState::EXECUTING_ACTIONS);
        bc.executeActions();
    }

    void endTurn() {
        bc.endTurn();
        bc.setState(InputState::EXECUTING_ACTIONS);
        bc.executeActions();
    }

    void drinkPotion(int potionIdx, int targetIdx) {
        bc.drinkPotion(potionIdx, targetIdx);
        bc.setState(InputState::EXECUTING_ACTIONS);
        bc.executeActions();
    }

    void discardPotion(int potionIdx) {
        bc.discardPotion(potionIdx);
    }

    // Handle in-combat card selection screens (Armaments, Discovery, Headbutt, etc.)
    void chooseCardSelect(int idx) {
        switch (bc.cardSelectInfo.cardSelectTask) {
            case CardSelectTask::ARMAMENTS:   bc.chooseArmamentsCard(idx); break;
            case CardSelectTask::DUAL_WIELD:  bc.chooseDualWieldCard(idx); break;
            case CardSelectTask::EXHAUST_ONE: bc.chooseExhaustOneCard(idx); break;
            case CardSelectTask::FORETHOUGHT: bc.chooseForethoughtCard(idx); break;
            case CardSelectTask::HEADBUTT:    bc.chooseHeadbuttCard(idx); break;
            case CardSelectTask::WARCRY:      bc.chooseWarcryCard(idx); break;
            case CardSelectTask::DISCOVERY:   bc.chooseDiscoveryCard(bc.cardSelectInfo.cards[idx]); break;
            case CardSelectTask::EXHUME:      bc.chooseExhumeCard(idx); break;
            case CardSelectTask::RECYCLE:     bc.chooseRecycleCard(idx); break;
            default: break;
        }
        bc.setState(InputState::EXECUTING_ACTIONS);
        bc.executeActions();
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << bc;
        return oss.str();
    }

    static std::string debug_layout() {
        std::ostringstream oss;
        oss << "sizeof(PyBattleContext) = " << sizeof(PyBattleContext) << "\n";
        oss << "sizeof(BattleContext) = " << sizeof(BattleContext) << "\n";
        BattleContext bc2;
        char* base = reinterpret_cast<char*>(&bc2);
        oss << "potionCount offset: " << (reinterpret_cast<char*>(&bc2.potionCount) - base) << "\n";
        oss << "potionCapacity offset: " << (reinterpret_cast<char*>(&bc2.potionCapacity) - base) << "\n";
        oss << "potions offset: " << (reinterpret_cast<char*>(&bc2.potions) - base) << "\n";
        oss << "potions end: " << (reinterpret_cast<char*>(&bc2.potions[4]) - base + sizeof(Potion)) << "\n";
        oss << "turn offset: " << (reinterpret_cast<char*>(&bc2.turn) - base) << "\n";
        oss << "player offset: " << (reinterpret_cast<char*>(&bc2.player) - base) << "\n";
        oss << "miscBits offset: " << (reinterpret_cast<char*>(&bc2.miscBits) - base) << "\n";
        oss << "sizeof(Potion): " << sizeof(Potion) << "\n";
        oss << "sizeof(Player): " << sizeof(Player) << "\n";
        return oss.str();
    }
};
// ---- end PyBattleContext ----

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");

    pybind11::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    // ValueNet binding
    pybind11::class_<search::ValueNet>(m, "ValueNet")
        .def(pybind11::init<>())
        .def("load", &search::ValueNet::load, pybind11::arg("model_dir"),
             "Load model weights from directory with manifest.json")
        .def_readonly("loaded", &search::ValueNet::loaded)
        .def("set_act", &search::ValueNet::setAct, pybind11::arg("act"),
             "Set current act number (1-3)")
        .def("set_relic_flags", [](search::ValueNet &vn, const std::vector<int> &relicIds) {
            vn.relicFlags.fill(0.0f);
            for (int rid : relicIds) {
                if (rid >= 0 && rid < search::VN_NUM_RELIC_IDS) {
                    vn.relicFlags[rid] = 1.0f;
                }
            }
        }, pybind11::arg("relic_ids"),
             "Set relic flags from list of relic ID integers");

    pybind11::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def_readwrite("fair_rng", &search::ScumSearchAgent2::fairRng, "resample RNG per simulation for fair MCTS (no perfect foresight)")
        .def_readwrite("search_potions", &search::ScumSearchAgent2::searchPotions, "include potion actions in MCTS search tree (disable to reduce crashes)")
        .def_readwrite("skip_hallway_potions", &search::ScumSearchAgent2::skipHallwayPotions, "only search potions for elite/boss fights (reduces action space for hallway fights)")
        .def_readwrite("exploration_parameter", &search::ScumSearchAgent2::explorationParameter, "UCB1 exploration constant (-1 = default 3*sqrt(2))")
        .def_readwrite("heuristic_playouts", &search::ScumSearchAgent2::heuristicPlayouts, "use SimpleAgent heuristic instead of random for MCTS playouts")
        .def("set_value_net", [](search::ScumSearchAgent2 &a, search::ValueNet &vn) {
            a.valueNet = &vn;
        }, pybind11::arg("value_net"), "Set value network for batched greedy combat evaluation")
        .def("clear_value_net", [](search::ScumSearchAgent2 &a) {
            a.valueNet = nullptr;
        }, "Clear value network, reverting to playout-based evaluation")
        .def_readwrite("value_net_playout_turns", &search::ScumSearchAgent2::valueNetPlayoutTurns,
             "Number of turns to playout with SimpleAgent before value net evaluation (0 = no playout)")
        .def_readwrite("value_net_greedy", &search::ScumSearchAgent2::valueNetGreedy,
             "If true, use greedy DFS with value net; if false (default), use value net within MCTS")
        .def("playout", &search::ScumSearchAgent2::playout)
        .def("playout_battle", [](search::ScumSearchAgent2 &a, PyBattleContext &pbc) {
            a.playoutBattle(pbc.bc);
        }, "play out a complete battle using MCTS search")
        .def("search_step", [](search::ScumSearchAgent2 &a, PyBattleContext &pbc, bool execute) -> pybind11::dict {
            auto &bc = pbc.bc;
            pybind11::dict result;

            if (bc.outcome != Outcome::UNDECIDED) {
                result["type"] = "done";
                return result;
            }

            // Determine simulation count (boss multiplier)
            const std::int64_t simulationCount = isBossEncounter(bc.encounter) ?
                (static_cast<std::int64_t>(a.bossSimulationMultiplier * a.simulationCountBase)) : a.simulationCountBase;

            // Create searcher with agent settings
            search::BattleScumSearcher2 searcher(bc);
            searcher.fairRng = a.fairRng;
            searcher.searchPotions = a.searchPotions;
            if (a.skipHallwayPotions && !isEliteOrBossEncounter(bc.encounter)) {
                searcher.searchPotions = false;
            }
            searcher.useHeuristicPlayouts = a.heuristicPlayouts;
            searcher.valueNet = a.valueNet;
            searcher.valueNetPlayoutTurns = a.valueNetPlayoutTurns;
            if (a.explorationParameter >= 0) {
                searcher.explorationParameter = a.explorationParameter;
            }

            // Run MCTS search
            searcher.search(simulationCount);

            // Find most-visited action at root
            std::int64_t maxSimulations = -1;
            const search::BattleScumSearcher2::Edge *maxEdge = nullptr;
            for (const auto &edge : searcher.root.edges) {
                if (edge.node.simulationCount > maxSimulations) {
                    maxSimulations = edge.node.simulationCount;
                    maxEdge = &edge;
                }
            }

            if (maxEdge == nullptr) {
                result["type"] = "done";
                return result;
            }

            // Extract action info before executing
            auto action = maxEdge->action;
            auto actionType = action.getActionType();

            switch (actionType) {
                case search::ActionType::CARD: {
                    int handIdx = action.getSourceIdx();
                    int targetIdx = action.getTargetIdx();
                    result["type"] = "card";
                    result["hand_idx"] = handIdx;
                    result["target_idx"] = targetIdx;
                    if (handIdx >= 0 && handIdx < bc.cards.cardsInHand) {
                        result["card_id"] = static_cast<int>(bc.cards.hand[handIdx].getId());
                        result["upgraded"] = bc.cards.hand[handIdx].isUpgraded();
                    }
                    break;
                }
                case search::ActionType::POTION: {
                    int slot = action.getSourceIdx();
                    int targetIdx = action.getTargetIdx();
                    result["type"] = "potion";
                    result["slot"] = slot;
                    result["target_idx"] = targetIdx;
                    if (slot >= 0 && slot < std::min(bc.potionCapacity, 5)) {
                        result["potion_id"] = static_cast<int>(bc.potions[slot]);
                    }
                    break;
                }
                case search::ActionType::SINGLE_CARD_SELECT: {
                    result["type"] = "card_select";
                    result["select_idx"] = action.getSelectIdx();
                    break;
                }
                case search::ActionType::MULTI_CARD_SELECT: {
                    result["type"] = "multi_card_select";
                    auto idxs = action.getSelectedIdxs();
                    pybind11::list pyIdxs;
                    for (int idx : idxs) { pyIdxs.append(idx); }
                    result["select_idxs"] = pyIdxs;
                    break;
                }
                case search::ActionType::END_TURN: {
                    result["type"] = "end_turn";
                    break;
                }
            }

            // Get human-readable description
            std::ostringstream oss;
            action.printDesc(oss, bc);
            result["description"] = oss.str();

            // Store action bits for deferred execution
            result["action_bits"] = static_cast<std::int32_t>(action.bits);

            // Execute the action on the battle context (unless deferred)
            if (execute) {
                action.execute(bc);
            }

            return result;
        }, pybind11::arg("bc"), pybind11::arg("execute") = true,
           "Run one MCTS search step: search, pick best action, optionally execute it, return action info dict")

        .def("execute_action", [](search::ScumSearchAgent2 &a, PyBattleContext &pbc, int bits) {
            auto &bc = pbc.bc;
            search::Action action;
            action.bits = static_cast<std::int32_t>(bits);
            action.execute(bc);
        }, pybind11::arg("bc"), pybind11::arg("action_bits"),
           "Execute a previously returned action (by its action_bits) on the battle context");

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())
        .def("copy", [](const GameContext &gc) { return GameContext(gc); }, "creates a deep copy of the game context")
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        .def_readwrite("cur_room", &GameContext::curRoom)
//        .def_readwrite("cur_event", &GameContext::curEvent) // todo standardize event names
        .def_readwrite("boss", &GameContext::boss)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard);

    // ---- Step-by-step control methods ----

    // Map navigation
    gameContext
        .def("transition_to_map_node", [](GameContext &gc, int x) {
            gc.transitionToMapNode(x);
        }, "Move to the map node at the given x coordinate")
        .def("get_available_map_nodes", [](const GameContext &gc) {
            std::vector<std::pair<int, Room>> result;
            if (!gc.map) return result;
            if (gc.curMapNodeY == 14) {
                result.push_back({0, Room::BOSS});
            } else if (gc.curMapNodeY == -1) {
                for (const auto &node : gc.map->nodes[0]) {
                    if (node.edgeCount > 0) result.push_back({node.x, node.room});
                }
            } else {
                auto node = gc.map->getNode(gc.curMapNodeX, gc.curMapNodeY);
                for (int i = 0; i < node.edgeCount; ++i) {
                    int nx = node.edges[i];
                    result.push_back({nx, gc.map->getNode(nx, gc.curMapNodeY + 1).room});
                }
            }
            return result;
        }, "Returns list of (x_coord, Room) pairs for valid next map nodes");

    // Neow event (floor 0)
    gameContext
        .def("choose_neow_option", [](GameContext &gc, int idx) {
            gc.chooseNeowOption(gc.info.neowRewards[idx]);
        }, "Choose Neow bonus at game start (idx 0-3)")
        .def_property_readonly("neow_options", [](const GameContext &gc) {
            std::vector<std::string> opts;
            for (int i = 0; i < 4; ++i) {
                std::string s = Neow::bonusStrings[static_cast<int>(gc.info.neowRewards[i].r)];
                std::string d = Neow::drawbackStrings[static_cast<int>(gc.info.neowRewards[i].d)];
                if (!d.empty()) s += " / " + d;
                opts.push_back(s);
            }
            return opts;
        }, "Human-readable Neow option strings");

    // Event screen
    gameContext
        .def_property_readonly("cur_event", [](const GameContext &gc) { return gc.curEvent; })
        .def_property_readonly("event_data", [](const GameContext &gc) { return gc.info.eventData; },
             "Phase/counter for multi-phase events (CURSED_TOME, DEAD_ADVENTURER, etc.)")
        .def("choose_event_option", [](GameContext &gc, int idx) { gc.chooseEventOption(idx); },
             "Choose event option by index")
        .def("choose_match_and_keep", [](GameContext &gc, int idx1, int idx2) {
            gc.chooseMatchAndKeepCards(idx1, idx2);
        }, "Choose two cards in the Match and Keep event");

    // Card select screen (non-combat: transform, remove, upgrade, etc.)
    gameContext
        .def_property_readonly("card_select_type",
            [](const GameContext &gc) { return gc.info.selectScreenType; })
        .def_property_readonly("card_select_count",
            [](const GameContext &gc) { return gc.info.toSelectCount; })
        .def_property_readonly("card_select_cards", [](const GameContext &gc) {
            std::vector<Card> cards;
            for (int i = 0; i < (int)gc.info.toSelectCards.size(); ++i)
                cards.push_back(gc.info.toSelectCards[i].card);
            return cards;
        }, "Cards available to choose from on the card select screen")
        .def("choose_card_select", [](GameContext &gc, int idx) {
            gc.chooseSelectCardScreenOption(idx);
        }, "Choose a card by index on the card select screen");

    // Rest room (campfire)
    gameContext.def("choose_campfire_option", [](GameContext &gc, int idx) {
        gc.chooseCampfireOption(idx);
    }, "0=Rest, 1=Smith(upgrade), 2=Recall(key), 3=Lift(girya), 4=Toke(remove), 5=Dig, 6=Skip");

    // Treasure room
    gameContext.def("choose_treasure_room_option", [](GameContext &gc, bool openChest) {
        gc.chooseTreasureRoomOption(openChest);
    }, "True to open chest, False to skip");

    // Boss relic selection
    gameContext
        .def_property_readonly("boss_relics", [](const GameContext &gc) {
            return std::vector<RelicId>{gc.info.bossRelics[0], gc.info.bossRelics[1], gc.info.bossRelics[2]};
        }, "Three boss relics to choose from (after act boss)")
        .def("choose_boss_relic", [](GameContext &gc, int idx) { gc.chooseBossRelic(idx); },
             "Choose boss relic by index (0-2), or 3 to skip");

    // Rewards screen
    gameContext
        .def_property_readonly("rewards_gold_count",
            [](const GameContext &gc) { return gc.info.rewardsContainer.goldRewardCount; })
        .def("rewards_gold", [](const GameContext &gc, int idx) {
            return gc.info.rewardsContainer.gold[idx];
        }, "Gold amount for reward at idx")
        .def_property_readonly("rewards_card_count",
            [](const GameContext &gc) { return gc.info.rewardsContainer.cardRewardCount; })
        .def("rewards_cards", [](const GameContext &gc, int rewardIdx) {
            std::vector<Card> cards;
            const auto &cr = gc.info.rewardsContainer.cardRewards[rewardIdx];
            for (int i = 0; i < (int)cr.size(); ++i) cards.push_back(cr[i]);
            return cards;
        }, "Cards available in reward slot rewardIdx")
        .def_property_readonly("rewards_relic_count",
            [](const GameContext &gc) { return gc.info.rewardsContainer.relicCount; })
        .def("rewards_relic", [](const GameContext &gc, int idx) {
            return gc.info.rewardsContainer.relics[idx];
        }, "Relic at reward idx")
        .def_property_readonly("rewards_potion_count",
            [](const GameContext &gc) { return gc.info.rewardsContainer.potionCount; })
        .def("rewards_potion", [](const GameContext &gc, int idx) {
            return gc.info.rewardsContainer.potions[idx];
        }, "Potion at reward idx")
        .def_property_readonly("rewards_emerald_key",
            [](const GameContext &gc) { return gc.info.rewardsContainer.emeraldKey; })
        .def_property_readonly("rewards_sapphire_key",
            [](const GameContext &gc) { return gc.info.rewardsContainer.sapphireKey; })
        .def("take_reward_gold", [](GameContext &gc, int idx) {
            gc.obtainGold(gc.info.rewardsContainer.gold[idx]);
            gc.info.rewardsContainer.removeGoldReward(idx);
        })
        .def("take_reward_card", [](GameContext &gc, int rewardIdx, int cardIdx) {
            gc.deck.obtain(gc, gc.info.rewardsContainer.cardRewards[rewardIdx][cardIdx]);
            gc.info.rewardsContainer.removeCardReward(rewardIdx);
        })
        .def("take_reward_singing_bowl", [](GameContext &gc, int rewardIdx) {
            gc.playerIncreaseMaxHp(2);
            gc.info.rewardsContainer.removeCardReward(rewardIdx);
        }, "Take +2 max HP instead of a card (requires Singing Bowl relic)")
        .def("take_reward_relic", [](GameContext &gc, int idx) {
            gc.obtainRelic(gc.info.rewardsContainer.relics[idx]);
            if (gc.info.rewardsContainer.sapphireKey && idx == gc.info.rewardsContainer.relicCount - 1)
                gc.info.rewardsContainer.sapphireKey = false;
            gc.info.rewardsContainer.removeRelicReward(idx);
        })
        .def("take_reward_potion", [](GameContext &gc, int idx) {
            gc.obtainPotion(gc.info.rewardsContainer.potions[idx]);
            gc.info.rewardsContainer.removePotionReward(idx);
        })
        .def("take_reward_emerald_key", [](GameContext &gc) {
            gc.obtainKey(Key::EMERALD_KEY);
            gc.info.rewardsContainer.emeraldKey = false;
        })
        .def("take_reward_sapphire_key", [](GameContext &gc) {
            if (gc.info.rewardsContainer.relicCount > 0)
                gc.info.rewardsContainer.removeRelicReward(gc.info.rewardsContainer.relicCount - 1);
            gc.obtainKey(Key::SAPPHIRE_KEY);
            gc.info.rewardsContainer.sapphireKey = false;
        })
        .def("proceed", [](GameContext &gc) { gc.regainControl(); },
             "Skip remaining rewards and proceed to map");

    // Shop
    gameContext
        .def("shop_cards", [](const GameContext &gc) {
            std::vector<Card> cards;
            for (int i = 0; i < 7; ++i) cards.push_back(gc.info.shop.cards[i]);
            return cards;
        }, "List of 7 shop cards (CardId::INVALID means sold out)")
        .def("shop_card_price", [](const GameContext &gc, int idx) {
            return gc.info.shop.cardPrice(idx);
        }, "Price of shop card at idx (-1 if unavailable)")
        .def("shop_relics", [](const GameContext &gc) {
            return std::vector<RelicId>{gc.info.shop.relics[0], gc.info.shop.relics[1], gc.info.shop.relics[2]};
        })
        .def("shop_relic_price", [](const GameContext &gc, int idx) {
            return gc.info.shop.relicPrice(idx);
        }, "Price of shop relic at idx (-1 if unavailable)")
        .def("shop_potions", [](const GameContext &gc) {
            return std::vector<Potion>{gc.info.shop.potions[0], gc.info.shop.potions[1], gc.info.shop.potions[2]};
        })
        .def("shop_potion_price", [](const GameContext &gc, int idx) {
            return gc.info.shop.potionPrice(idx);
        })
        .def_property_readonly("shop_remove_card_cost",
            [](const GameContext &gc) { return gc.info.shop.removeCost; },
            "Cost to remove a card (-1 if unavailable)")
        .def("shop_buy_card", [](GameContext &gc, int idx) { gc.info.shop.buyCard(gc, idx); })
        .def("shop_buy_relic", [](GameContext &gc, int idx) { gc.info.shop.buyRelic(gc, idx); })
        .def("shop_buy_potion", [](GameContext &gc, int idx) { gc.info.shop.buyPotion(gc, idx); })
        .def("shop_buy_remove", [](GameContext &gc) { gc.info.shop.buyCardRemove(gc); })
        .def("proceed_from_shop", [](GameContext &gc) { gc.screenState = ScreenState::MAP_SCREEN; },
             "Leave the shop and return to the map");

    // Potions (on GameContext, usable between battles)
    gameContext
        .def_property_readonly("potion_count",
            [](const GameContext &gc) { return gc.potionCount; })
        .def_property_readonly("potion_capacity",
            [](const GameContext &gc) { return gc.potionCapacity; })
        .def("get_potion", [](const GameContext &gc, int idx) { return gc.potions[idx]; },
             "Potion enum value at slot idx")
        .def("drink_potion", [](GameContext &gc, int idx) { gc.drinkPotionAtIdx(idx); })
        .def("discard_potion", [](GameContext &gc, int idx) { gc.discardPotionAtIdx(idx); })
        .def("set_potion", [](GameContext &gc, int slot, Potion p) {
            if (slot < 0 || slot >= std::min(gc.potionCapacity, 5)) return;
            gc.potions[slot] = p;
            // recount
            int count = 0;
            for (int i = 0; i < std::min(gc.potionCapacity, 5); ++i) {
                if (gc.potions[i] != Potion::EMPTY_POTION_SLOT) ++count;
            }
            gc.potionCount = count;
        }, pybind11::arg("slot"), pybind11::arg("potion"),
           "Set potion at slot idx and update count")
        .def("set_potions", [](GameContext &gc, std::vector<Potion> potions) {
            int cap = std::min(gc.potionCapacity, 5);
            int count = 0;
            for (int i = 0; i < cap; ++i) {
                Potion p = (i < (int)potions.size()) ? potions[i] : Potion::EMPTY_POTION_SLOT;
                gc.potions[i] = p;
                if (p != Potion::EMPTY_POTION_SLOT) ++count;
            }
            gc.potionCount = count;
        }, pybind11::arg("potions"),
           "Replace all potion slots from a list of Potion enums")
        .def("sync_deck", [](GameContext &gc, std::vector<Card> cards) {
            gc.deck.cards.clear();
            std::array<int,4> typeCounts = {0,0,0,0};
            for (auto &c : cards) {
                gc.deck.cards.push_back(c);
                int t = static_cast<int>(c.getType());
                if (t >= 0 && t < 4) typeCounts[t]++;
            }
            gc.deck.cardTypeCounts = typeCounts;
            gc.deck.upgradeableCount = 0;
            gc.deck.transformableCount = 0;
            for (int i = 0; i < gc.deck.size(); ++i) {
                if (gc.deck.cards[i].canUpgrade()) gc.deck.upgradeableCount++;
                if (gc.deck.cards[i].canTransform()) gc.deck.transformableCount++;
            }
        }, pybind11::arg("cards"),
           "Replace entire deck with given cards list")
        .def("add_relic", [](GameContext &gc, RelicId r) {
            if (!gc.relics.has(r)) {
                gc.relics.add({r, 0});
            }
        }, pybind11::arg("relic_id"),
           "Add a relic if not already present")
        .def("remove_relic", [](GameContext &gc, RelicId r) {
            gc.relics.remove(r);
        }, pybind11::arg("relic_id"),
           "Remove a relic by ID")
        .def("has_relic", [](const GameContext &gc, RelicId r) {
            return gc.relics.has(r);
        }, pybind11::arg("relic_id"),
           "Check if a relic is present");

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    pybind11::class_<Map> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int,int,bool>());
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState");
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room");
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<CardId>(m, "CardId")
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter");
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId");
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    // ---- BattleContext class ----
    pybind11::class_<PyBattleContext>(m, "BattleContext")
        .def(pybind11::init<GameContext&>(),
             "Initialize a battle from the current GameContext (must be in BATTLE screen state)")
        .def("exit_battle", &PyBattleContext::exitBattle,
             "Apply battle results back to GameContext (call after outcome != UNDECIDED)")
        .def("play_card", &PyBattleContext::playCard,
             pybind11::arg("hand_idx"), pybind11::arg("target_idx") = -1,
             "Play card at hand_idx. target_idx=-1 auto-picks first targetable monster.")
        .def("end_turn", &PyBattleContext::endTurn,
             "End player turn; monsters act automatically via executeActions()")
        .def("drink_potion", &PyBattleContext::drinkPotion,
             pybind11::arg("potion_idx"), pybind11::arg("target_idx") = 0)
        .def("discard_potion", &PyBattleContext::discardPotion)
        .def("choose_card_select", &PyBattleContext::chooseCardSelect,
             "Handle in-combat CARD_SELECT state (Armaments, Discovery, Headbutt, etc.)")
        // Outcome / state
        .def_property_readonly("outcome",
            [](const PyBattleContext &pbc) { return pbc.bc.outcome; })
        .def_property_readonly("input_state",
            [](const PyBattleContext &pbc) { return pbc.bc.inputState; })
        .def_property_readonly("turn",
            [](const PyBattleContext &pbc) { return pbc.bc.turn; })
        .def_property_readonly("card_select_task",
            [](const PyBattleContext &pbc) { return pbc.bc.cardSelectInfo.cardSelectTask; },
            "When input_state==CARD_SELECT, indicates what kind of selection is needed")
        // Player
        .def_property_readonly("player_cur_hp",
            [](const PyBattleContext &pbc) { return pbc.bc.player.curHp; })
        .def_property_readonly("player_max_hp",
            [](const PyBattleContext &pbc) { return pbc.bc.player.maxHp; })
        .def_property_readonly("player_block",
            [](const PyBattleContext &pbc) { return pbc.bc.player.block; })
        .def_property_readonly("player_energy",
            [](const PyBattleContext &pbc) { return pbc.bc.player.energy; })
        .def_property_readonly("player_strength",
            [](const PyBattleContext &pbc) { return pbc.bc.player.strength; })
        .def_property_readonly("player_dexterity",
            [](const PyBattleContext &pbc) { return pbc.bc.player.dexterity; })
        // Monsters
        .def_property_readonly("monster_count",
            [](const PyBattleContext &pbc) { return pbc.bc.monsters.monsterCount; })
        .def("monster_cur_hp",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].curHp; })
        .def("monster_max_hp",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].maxHp; })
        .def("monster_block",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].block; })
        .def("is_monster_alive",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].isAlive(); })
        .def("is_monster_targetable",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].isTargetable(); })
        // Monster intent & status
        .def("monster_is_attacking",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].isAttacking(); })
        .def("monster_move_damage",
            [](const PyBattleContext &pbc, int idx) {
                auto &m = pbc.bc.monsters.arr[idx];
                if (!m.isAttacking()) return 0;
                auto dInfo = m.getMoveBaseDamage(pbc.bc);
                return m.calculateDamageToPlayer(pbc.bc, dInfo.damage);
            })
        .def("monster_move_hits",
            [](const PyBattleContext &pbc, int idx) {
                auto &m = pbc.bc.monsters.arr[idx];
                if (!m.isAttacking()) return 0;
                return m.getMoveBaseDamage(pbc.bc).attackCount;
            })
        .def("monster_strength",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].strength; })
        .def("monster_weak",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].weak; })
        .def("monster_vulnerable",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.monsters.arr[idx].vulnerable; })
        .def("monster_id",
            [](const PyBattleContext &pbc, int idx) { return (int)pbc.bc.monsters.arr[idx].id; })
        .def("monster_move_id",
            [](const PyBattleContext &pbc, int idx) { return (int)pbc.bc.monsters.arr[idx].moveHistory[0]; })
        // Hand / card piles
        .def_property_readonly("hand_size",
            [](const PyBattleContext &pbc) { return pbc.bc.cards.cardsInHand; })
        .def_property_readonly("draw_pile_size",
            [](const PyBattleContext &pbc) { return (int)pbc.bc.cards.drawPile.size(); })
        .def_property_readonly("discard_pile_size",
            [](const PyBattleContext &pbc) { return (int)pbc.bc.cards.discardPile.size(); })
        .def("hand_card_id",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.hand[idx].getId(); })
        .def("hand_card_upgraded",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.hand[idx].isUpgraded(); })
        .def("hand_card_cost",
            [](const PyBattleContext &pbc, int idx) { return (int)pbc.bc.cards.hand[idx].costForTurn; })
        .def("hand_card_requires_target",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.hand[idx].requiresTarget(); })
        .def("can_play_card",
            [](const PyBattleContext &pbc, int idx) {
                return pbc.bc.cards.hand[idx].canUseOnAnyTarget(pbc.bc);
            })
        // Draw pile card accessors
        .def("draw_pile_card_id",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.drawPile[idx].getId(); })
        .def("draw_pile_card_upgraded",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.drawPile[idx].isUpgraded(); })
        // Discard pile card accessors
        .def("discard_pile_card_id",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.discardPile[idx].getId(); })
        .def("discard_pile_card_upgraded",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.discardPile[idx].isUpgraded(); })
        // Exhaust pile accessors
        .def_property_readonly("exhaust_pile_size",
            [](const PyBattleContext &pbc) { return (int)pbc.bc.cards.exhaustPile.size(); })
        .def("exhaust_pile_card_id",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.exhaustPile[idx].getId(); })
        .def("exhaust_pile_card_upgraded",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.cards.exhaustPile[idx].isUpgraded(); })
        // Player status effects
        .def("player_has_status",
            [](const PyBattleContext &pbc, int status_id) {
                return pbc.bc.player.hasStatusRuntime(static_cast<PlayerStatus>(status_id));
            }, pybind11::arg("status_id"),
            "Check if player has a status effect (by PlayerStatus enum int value)")
        .def("player_get_status",
            [](const PyBattleContext &pbc, int status_id) {
                return pbc.bc.player.getStatusRuntime(static_cast<PlayerStatus>(status_id));
            }, pybind11::arg("status_id"),
            "Get player status effect value (by PlayerStatus enum int value)")
        // Potions (in combat)
        .def_property_readonly("potion_count",
            [](const PyBattleContext &pbc) { return pbc.bc.potionCount; })
        .def("get_potion",
            [](const PyBattleContext &pbc, int idx) { return pbc.bc.potions[idx]; })
        // --- Setters for per-turn sync ---
        .def("set_player_hp",
            [](PyBattleContext &pbc, int hp) { pbc.bc.player.curHp = hp; },
            "Set player current HP")
        .def("set_player_max_hp",
            [](PyBattleContext &pbc, int hp) { pbc.bc.player.maxHp = hp; },
            "Set player max HP")
        .def("set_player_block",
            [](PyBattleContext &pbc, int block) { pbc.bc.player.block = block; },
            "Set player block")
        .def("set_player_energy",
            [](PyBattleContext &pbc, int energy) { pbc.bc.player.energy = energy; },
            "Set player energy")
        .def("set_player_strength",
            [](PyBattleContext &pbc, int str) { pbc.bc.player.strength = str; },
            "Set player strength")
        .def("set_player_dexterity",
            [](PyBattleContext &pbc, int dex) { pbc.bc.player.dexterity = dex; },
            "Set player dexterity")
        .def("set_monster_hp",
            [](PyBattleContext &pbc, int idx, int hp) {
                if (idx >= 0 && idx < pbc.bc.monsters.monsterCount) {
                    pbc.bc.monsters.arr[idx].curHp = hp;
                }
            }, pybind11::arg("idx"), pybind11::arg("hp"),
            "Set monster current HP at index")
        .def("set_monster_block",
            [](PyBattleContext &pbc, int idx, int block) {
                if (idx >= 0 && idx < pbc.bc.monsters.monsterCount) {
                    pbc.bc.monsters.arr[idx].block = block;
                }
            }, pybind11::arg("idx"), pybind11::arg("block"),
            "Set monster block at index")
        .def("__repr__", &PyBattleContext::repr)
        .def_static("debug_layout", &PyBattleContext::debug_layout);

    // ---- New enums ----

    pybind11::enum_<Outcome>(m, "BattleOutcome")
        .value("UNDECIDED", Outcome::UNDECIDED)
        .value("PLAYER_VICTORY", Outcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", Outcome::PLAYER_LOSS);

    pybind11::enum_<InputState>(m, "InputState")
        .value("EXECUTING_ACTIONS", InputState::EXECUTING_ACTIONS)
        .value("PLAYER_NORMAL", InputState::PLAYER_NORMAL)
        .value("CARD_SELECT", InputState::CARD_SELECT);

    pybind11::enum_<CardSelectScreenType>(m, "CardSelectScreenType")
        .value("INVALID", CardSelectScreenType::INVALID)
        .value("TRANSFORM", CardSelectScreenType::TRANSFORM)
        .value("TRANSFORM_UPGRADE", CardSelectScreenType::TRANSFORM_UPGRADE)
        .value("UPGRADE", CardSelectScreenType::UPGRADE)
        .value("REMOVE", CardSelectScreenType::REMOVE)
        .value("DUPLICATE", CardSelectScreenType::DUPLICATE)
        .value("OBTAIN", CardSelectScreenType::OBTAIN)
        .value("BOTTLE", CardSelectScreenType::BOTTLE)
        .value("BONFIRE_SPIRITS", CardSelectScreenType::BONFIRE_SPIRITS);

    pybind11::enum_<CardSelectTask>(m, "CardSelectTask")
        .value("INVALID", CardSelectTask::INVALID)
        .value("ARMAMENTS", CardSelectTask::ARMAMENTS)
        .value("CODEX", CardSelectTask::CODEX)
        .value("DISCOVERY", CardSelectTask::DISCOVERY)
        .value("DUAL_WIELD", CardSelectTask::DUAL_WIELD)
        .value("EXHAUST_ONE", CardSelectTask::EXHAUST_ONE)
        .value("EXHAUST_MANY", CardSelectTask::EXHAUST_MANY)
        .value("EXHUME", CardSelectTask::EXHUME)
        .value("FORETHOUGHT", CardSelectTask::FORETHOUGHT)
        .value("GAMBLE", CardSelectTask::GAMBLE)
        .value("HEADBUTT", CardSelectTask::HEADBUTT)
        .value("HOLOGRAM", CardSelectTask::HOLOGRAM)
        .value("LIQUID_MEMORIES_POTION", CardSelectTask::LIQUID_MEMORIES_POTION)
        .value("WARCRY", CardSelectTask::WARCRY);

    pybind11::enum_<Event>(m, "Event")
        .value("INVALID", Event::INVALID)
        .value("NEOW", Event::NEOW)
        .value("OMINOUS_FORGE", Event::OMINOUS_FORGE)
        .value("PLEADING_VAGRANT", Event::PLEADING_VAGRANT)
        .value("ANCIENT_WRITING", Event::ANCIENT_WRITING)
        .value("OLD_BEGGAR", Event::OLD_BEGGAR)
        .value("BIG_FISH", Event::BIG_FISH)
        .value("BONFIRE_SPIRITS", Event::BONFIRE_SPIRITS)
        .value("CURSED_TOME", Event::CURSED_TOME)
        .value("DEAD_ADVENTURER", Event::DEAD_ADVENTURER)
        .value("DESIGNER_IN_SPIRE", Event::DESIGNER_IN_SPIRE)
        .value("AUGMENTER", Event::AUGMENTER)
        .value("DUPLICATOR", Event::DUPLICATOR)
        .value("FACE_TRADER", Event::FACE_TRADER)
        .value("FALLING", Event::FALLING)
        .value("FORGOTTEN_ALTAR", Event::FORGOTTEN_ALTAR)
        .value("THE_DIVINE_FOUNTAIN", Event::THE_DIVINE_FOUNTAIN)
        .value("GHOSTS", Event::GHOSTS)
        .value("GOLDEN_IDOL", Event::GOLDEN_IDOL)
        .value("GOLDEN_SHRINE", Event::GOLDEN_SHRINE)
        .value("WING_STATUE", Event::WING_STATUE)
        .value("KNOWING_SKULL", Event::KNOWING_SKULL)
        .value("LAB", Event::LAB)
        .value("THE_SSSSSERPENT", Event::THE_SSSSSERPENT)
        .value("LIVING_WALL", Event::LIVING_WALL)
        .value("MASKED_BANDITS", Event::MASKED_BANDITS)
        .value("MATCH_AND_KEEP", Event::MATCH_AND_KEEP)
        .value("MINDBLOOM", Event::MINDBLOOM)
        .value("HYPNOTIZING_COLORED_MUSHROOMS", Event::HYPNOTIZING_COLORED_MUSHROOMS)
        .value("MYSTERIOUS_SPHERE", Event::MYSTERIOUS_SPHERE)
        .value("THE_NEST", Event::THE_NEST)
        .value("NLOTH", Event::NLOTH)
        .value("NOTE_FOR_YOURSELF", Event::NOTE_FOR_YOURSELF)
        .value("PURIFIER", Event::PURIFIER)
        .value("SCRAP_OOZE", Event::SCRAP_OOZE)
        .value("SECRET_PORTAL", Event::SECRET_PORTAL)
        .value("SENSORY_STONE", Event::SENSORY_STONE)
        .value("SHINING_LIGHT", Event::SHINING_LIGHT)
        .value("THE_CLERIC", Event::THE_CLERIC)
        .value("THE_JOUST", Event::THE_JOUST)
        .value("THE_LIBRARY", Event::THE_LIBRARY)
        .value("THE_MAUSOLEUM", Event::THE_MAUSOLEUM)
        .value("THE_MOAI_HEAD", Event::THE_MOAI_HEAD)
        .value("THE_WOMAN_IN_BLUE", Event::THE_WOMAN_IN_BLUE)
        .value("TOMB_OF_LORD_RED_MASK", Event::TOMB_OF_LORD_RED_MASK)
        .value("TRANSMORGRIFIER", Event::TRANSMORGRIFIER)
        .value("UPGRADE_SHRINE", Event::UPGRADE_SHRINE)
        .value("VAMPIRES", Event::VAMPIRES)
        .value("WE_MEET_AGAIN", Event::WE_MEET_AGAIN)
        .value("WHEEL_OF_CHANGE", Event::WHEEL_OF_CHANGE)
        .value("WINDING_HALLS", Event::WINDING_HALLS)
        .value("WORLD_OF_GOOP", Event::WORLD_OF_GOOP);

    pybind11::enum_<Potion>(m, "Potion")
        .value("INVALID", Potion::INVALID)
        .value("EMPTY_POTION_SLOT", Potion::EMPTY_POTION_SLOT)
        .value("AMBROSIA", Potion::AMBROSIA)
        .value("ANCIENT_POTION", Potion::ANCIENT_POTION)
        .value("ATTACK_POTION", Potion::ATTACK_POTION)
        .value("BLESSING_OF_THE_FORGE", Potion::BLESSING_OF_THE_FORGE)
        .value("BLOCK_POTION", Potion::BLOCK_POTION)
        .value("BLOOD_POTION", Potion::BLOOD_POTION)
        .value("BOTTLED_MIRACLE", Potion::BOTTLED_MIRACLE)
        .value("COLORLESS_POTION", Potion::COLORLESS_POTION)
        .value("CULTIST_POTION", Potion::CULTIST_POTION)
        .value("CUNNING_POTION", Potion::CUNNING_POTION)
        .value("DEXTERITY_POTION", Potion::DEXTERITY_POTION)
        .value("DISTILLED_CHAOS", Potion::DISTILLED_CHAOS)
        .value("DUPLICATION_POTION", Potion::DUPLICATION_POTION)
        .value("ELIXIR_POTION", Potion::ELIXIR_POTION)
        .value("ENERGY_POTION", Potion::ENERGY_POTION)
        .value("ENTROPIC_BREW", Potion::ENTROPIC_BREW)
        .value("ESSENCE_OF_DARKNESS", Potion::ESSENCE_OF_DARKNESS)
        .value("ESSENCE_OF_STEEL", Potion::ESSENCE_OF_STEEL)
        .value("EXPLOSIVE_POTION", Potion::EXPLOSIVE_POTION)
        .value("FAIRY_POTION", Potion::FAIRY_POTION)
        .value("FEAR_POTION", Potion::FEAR_POTION)
        .value("FIRE_POTION", Potion::FIRE_POTION)
        .value("FLEX_POTION", Potion::FLEX_POTION)
        .value("FOCUS_POTION", Potion::FOCUS_POTION)
        .value("FRUIT_JUICE", Potion::FRUIT_JUICE)
        .value("GAMBLERS_BREW", Potion::GAMBLERS_BREW)
        .value("GHOST_IN_A_JAR", Potion::GHOST_IN_A_JAR)
        .value("HEART_OF_IRON", Potion::HEART_OF_IRON)
        .value("LIQUID_BRONZE", Potion::LIQUID_BRONZE)
        .value("LIQUID_MEMORIES", Potion::LIQUID_MEMORIES)
        .value("POISON_POTION", Potion::POISON_POTION)
        .value("POTION_OF_CAPACITY", Potion::POTION_OF_CAPACITY)
        .value("POWER_POTION", Potion::POWER_POTION)
        .value("REGEN_POTION", Potion::REGEN_POTION)
        .value("SKILL_POTION", Potion::SKILL_POTION)
        .value("SMOKE_BOMB", Potion::SMOKE_BOMB)
        .value("SNECKO_OIL", Potion::SNECKO_OIL)
        .value("SPEED_POTION", Potion::SPEED_POTION)
        .value("STANCE_POTION", Potion::STANCE_POTION)
        .value("STRENGTH_POTION", Potion::STRENGTH_POTION)
        .value("SWIFT_POTION", Potion::SWIFT_POTION)
        .value("WEAK_POTION", Potion::WEAK_POTION);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")


