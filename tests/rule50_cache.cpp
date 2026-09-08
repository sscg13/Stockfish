// Compare lazy/incremental, reused-cache and fresh evaluations with clock inputs.
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include "attacks.h"
#include "position.h"
#include "movegen.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

static int checks = 0;

static void compare(const Network& net, const Position& pos, AccumulatorStack& stack,
                    AccumulatorCaches& caches) {
    auto coldStack = std::make_unique<AccumulatorStack>();
    auto coldCache = std::make_unique<AccumulatorCaches>(net);
    coldStack->reset();
    auto expected = net.evaluate(pos, *coldStack, *coldCache);
    auto actual = net.evaluate(pos, stack, caches);
    if (actual != expected) {
        std::cerr << "Incremental mismatch: " << pos.fen() << '\n';
        std::exit(1);
    }
    ++checks;
}

static void visit(const Network& net, Position& pos, AccumulatorStack& stack,
                  AccumulatorCaches& caches, int depth) {
    if (depth == 0) {
        compare(net, pos, stack, caches);
        if (!pos.checkers()) {
            StateInfo nullState;
            const int clock = pos.rule50_count();
            pos.do_null_move(nullState);
            if (clock != pos.rule50_count()) std::abort();
            compare(net, pos, stack, caches);
            pos.undo_null_move();
            compare(net, pos, stack, caches);
        }
        return;
    }
    for (Move move : MoveList<LEGAL>(pos)) {
        StateInfo state;
        auto& dirty = stack.push();
        pos.do_move(move, state, pos.gives_check(move), dirty, nullptr, nullptr);
        // Do not evaluate the intermediate ply: exercise lazy reconstruction.
        visit(net, pos, stack, caches, depth - 1);
        pos.undo_move(move);
        stack.pop();
        compare(net, pos, stack, caches);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    Attacks::init();
    Position::init();
    auto net = std::make_unique<Network>();
    EvalFile file;
    net->load("", argv[1], file);
    if (!file.current || *file.current != argv[1]) return 3;
    auto caches = std::make_unique<AccumulatorCaches>(*net);
    auto stack = std::make_unique<AccumulatorStack>();
    for (const char* board : {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "8/8/3k4/8/3K4/8/4R3/8 w - -",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6",
        "4k3/P7/8/8/8/8/7p/4K3 w - -"
    }) {
        for (int clock : {0, 63, 99, 100}) {
            Position pos;
            StateInfo state;
            if (pos.set(std::string(board) + " " + std::to_string(clock) + " 42", false, &state)) return 4;
            stack->reset();
            compare(*net, pos, *stack, *caches);
            visit(*net, pos, *stack, *caches, 2);
        }
    }
    std::cout << checks << " incremental/cache/null/undo comparisons passed\n";
}
