#include <iostream>
#include <string>
#include <algorithm>
#include <type_traits>


//#define private public

#include "full_threats.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../../uci.h"
#include "../network.h"
#include "../nnue_misc.h"

using namespace Stockfish;
//including this for now for debug purposes
int main(int argc, char* argv[]) {
    Bitboards::init();
    Position::init();
    /*
    std::cout << std::is_trivial_v<Eval::NNUE::FeatureSet::IndexList> << std::endl;
    std::cout << std::is_trivial_v<StateInfo> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eval::NNUE::IndexType> << std::endl;
    UCIEngine engin(argc, argv);
    std::vector<std::string> moves;
    engin.engine.set_position("r3kb1r/pp1npppp/5n2/3p4/3P1B2/2N1P3/Pq2NPPP/R2QK2R b KQkq - 1 9", moves);
    engin.engine.networks->big.featureTransformer->print_accumulator<WHITE>(engin.engine.pos);
    engin.engine.states->emplace_back();
    engin.engine.pos.do_move(engin.to_move(engin.engine.pos, "b2a3"), engin.engine.states->back());
    engin.engine.networks->big.featureTransformer->print_accumulator<WHITE>(engin.engine.pos);
    //bigft.update_accumulator_scratch<WHITE>(pos);
    */
    std::cout << "early debug print\n";
    Position pos;
    Eval::NNUE::Features::Full_Threats test;
    std::string fen1 = std::string(argv[1]);
    std::string fen2 = std::string(argv[2]);
    StateListPtr states;
    states = StateListPtr(new std::deque<StateInfo>(1));
    Eval::NNUE::Features::Full_Threats::IndexList white1;
    Eval::NNUE::Features::Full_Threats::IndexList white2;
    Eval::NNUE::Features::Full_Threats::IndexList black1;
    Eval::NNUE::Features::Full_Threats::IndexList black2;
    Eval::NNUE::Features::Full_Threats::IndexList white3;
    Eval::NNUE::Features::Full_Threats::IndexList black3;
    Eval::NNUE::Features::Full_Threats::IndexList white4;
    Eval::NNUE::Features::Full_Threats::IndexList black4;
    white1.clear();
    white2.clear();
    black1.clear();
    black2.clear();
    white3.clear();
    white4.clear();
    black3.clear();
    black4.clear();
    pos.set(fen1, false, &states->back());
    std::cout << "set up stuff\n";
    test.append_active_threats<WHITE>(pos.byColorBB, pos.byTypeBB, pos.board, white1);
    test.append_active_threats<BLACK>(pos.byColorBB, pos.byTypeBB, pos.board, black1);
    std::cout << "Position 1 " << white1.size() << " white perspective features:\n";
    for (auto feature : white1) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black1.size() << " black perspective features:\n";
    for (auto feature : black1) {
        std::cout << feature << ", ";
    }
    pos.set(fen2, false, &states->back());
    test.append_active_threats<WHITE>(pos.byColorBB, pos.byTypeBB, pos.board, white2);
    test.append_active_threats<BLACK>(pos.byColorBB, pos.byTypeBB, pos.board, black2);
    std::cout << "\nPosition 2 " << white2.size() << " white perspective features:\n";
    for (auto feature : white2) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black2.size() << " black perspective features:\n";
    for (auto feature : black2) {
        std::cout << feature << ", ";
    }
    Eval::NNUE::write_difference(white1, white2, white3, white4);
    Eval::NNUE::write_difference(black1, black2, black3, black4);
    std::cout << "\n" << white3.size() << " removed white perspective features:\n";
    for (auto feature : white3) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black3.size() << " removed black perspective features:\n";
    for (auto feature : black3) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << white4.size() << " added white perspective features:\n";
    for (auto feature : white4) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black4.size() << " added black perspective features:\n";
    for (auto feature : black4) {
        std::cout << feature << ", ";
    }
    return 0;
}


/*
clang++ -pthread -static -o test.exe test.cpp simplified_threats.cpp ../network.cpp ../nnue_misc.cpp ../../benchmark.cpp ../../bitboard.cpp 
*/