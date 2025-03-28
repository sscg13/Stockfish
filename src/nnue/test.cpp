#define private public
#include "network.h"
#include "nnue_misc.h"
#include "nnue_common.h"
#include "nnue_architecture.h"
#include "nnue_accumulator.h"
#include "../uci.h"
#include "../position.h"
#include "../bitboard.h"
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    Stockfish::UCIEngine engin(argc, argv);
    std::vector<std::string> moves;
    engin.engine.load_big_network("nn-f8b4dc417908.nnue");
    std::cout << "loaded net" << std::endl;
    std::ofstream output;
    output.open("l1b0.params");
    engin.engine.networks->big.network[0].fc_0.write_parameters(output);
    std::cout << "wrote parameters" << std::endl;
    return 0;
}