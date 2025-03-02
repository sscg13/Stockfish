#include <iostream>
#include <string>
#include <algorithm>

#include "simplified_threats.h"

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"

using namespace Stockfish;
//including this for now for debug purposes
int main(int argc, char* argv[]) {
    Bitboards::init();
    Position::init();

    Position pos;
    std::string fen = std::string(argv[1]);
    pos.set(fen, false, pos.state());
    Eval::NNUE::Features::Simplified_Threats test;
    test.init_threat_offsets();
    Eval::NNUE::Features::Simplified_Threats::IndexList added;
    test.append_active_indices<WHITE>(pos, added);
    //std::sort(added.begin(), added.end());
    for (auto feature : added) {
        std::cout << feature << std::endl;
    }
    return 0;
}


/*
clang++ -pthread -static -o test.exe test.cpp simplified_threats.cpp ../network.cpp ../nnue_misc.cpp ../../benchmark.cpp ../../bitboard.cpp 
*/