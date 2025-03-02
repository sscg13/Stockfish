#include <iostream>
#include <string>
#include <algorithm>

#include "simplified_threats.h"

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../../uci.h"

using namespace Stockfish;
//including this for now for debug purposes
int main(int argc, char* argv[]) {
    Bitboards::init();
    Position::init();

    Position pos;
    std::string fen = std::string(argv[1]);
    StateListPtr states;
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, false, &states->back());
    Eval::NNUE::Features::Simplified_Threats test;
    Eval::NNUE::Features::Simplified_Threats::IndexList white;
    Eval::NNUE::Features::Simplified_Threats::IndexList black;
    test.init_threat_offsets();
    test.append_active_threats<WHITE>(pos, white);
    test.append_active_threats<BLACK>(pos, black);
    /*std::vector<int> features;
    for (auto feature : added) {
        features.push_back(feature);
    }
    std::sort(features.begin(), features.end());
    std::cout << features.size() << " features:\n";
    for (auto feature : features) {
        std::cout << feature << ", ";
    }*/
    std::cout << white.size() << " white perspective features:\n";
    for (auto feature : white) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black.size() << " black perspective features:\n";
    for (auto feature : black) {
        std::cout << feature << ", ";
    }
    return 0;
}


/*
clang++ -pthread -static -o test.exe test.cpp simplified_threats.cpp ../network.cpp ../nnue_misc.cpp ../../benchmark.cpp ../../bitboard.cpp 
*/