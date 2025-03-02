/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//Definition of input features Simplified_Threats of NNUE evaluation function

#include "simplified_threats.h"

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_accumulator.h"

namespace Stockfish::Eval::NNUE::Features {
//lookup array for indexing threats
void Simplified_Threats::init_threat_offsets() {
    int pieceoffset = 0;
    PieceType piecetbl[PIECE_NB] = {NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE,
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE};
    for (int piece = 0; piece < 16; piece++) {
        if (piecetbl[piece] == NO_PIECE_TYPE) {
            continue;
        }
        threatoffsets[piece][65] = pieceoffset;
        int squareoffset = 0;
        for (int from = SQ_A1; from <= SQ_H8; from++) {
            Square sq = static_cast<Square>(from);
            threatoffsets[piece][from-SQ_A1] = squareoffset;
            if (piecetbl[piece] != PAWN) {
                Bitboard attacks = attacks_bb(piecetbl[piece], sq, 0ULL);
                squareoffset += popcount(attacks);
            }
            else if (from >= SQ_A2 && from <= SQ_H7) {
                Bitboard attacks = pawn_attacks_bb(static_cast<Color>(piece/8), sq);
                squareoffset += popcount(attacks);
            }
        }
        threatoffsets[piece][64] = squareoffset;
        pieceoffset += 2*squareoffset;
    }
}

// Index of a feature for a given king position and another piece on some square
template<Color Perspective>
inline IndexType Simplified_Threats::make_index(Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
    bool enemy = (type_of(attkr) == type_of(attkd));
    from = (Square)(int(from) ^ OrientTBL[Perspective][ksq]);
    to = (Square)(int(to) ^ OrientTBL[Perspective][ksq]);
    if (Perspective == BLACK) {
        attkr = ~attkr;
        attkd = ~attkd;
    }
    if (from == to) {
        return IndexType(PieceSquareIndex[attkr]+from-SQ_A1);
    }
    Bitboard attacks = (type_of(attkr) == PAWN) ? pawn_attacks_bb(color_of(attkr), from) : attacks_bb(type_of(attkr), from, 0ULL);
    return IndexType(768 + threatoffsets[attkr][65] + enemy*threatoffsets[attkr][64]
                     + threatoffsets[attkr][from-SQ_A1] + popcount((square_bb(to)-1) & attacks));
}

// Get a list of indices for active features
template<Color Perspective>
void Simplified_Threats::append_active_indices(const Position& pos, IndexList& active) {
    Square   ksq = pos.square<KING>(Perspective);
    Bitboard bb  = pos.pieces();
    while (bb)
    {
        Square from = pop_lsb(bb);
        Piece attkr = pos.piece_on(from);
        PieceType attkrtype = type_of(attkr);
        Color attkrcolor = color_of(attkr);
        active.push_back(make_index<Perspective>(attkr, from, from, attkr, ksq));
        Bitboard attacks = ((attkrtype == PAWN) ? pawn_attacks_bb(attkrcolor, from) : attacks_bb(attkrtype, from, pos.pieces())) & pos.pieces();
        while (attacks) {
            Square to = pop_lsb(attacks);
            Piece attkd = pos.piece_on(to);
            active.push_back(make_index<Perspective>(attkr, from, to, attkd, ksq));
        }
    }
}

// Explicit template instantiations
template void Simplified_Threats::append_active_indices<WHITE>(const Position& pos, IndexList& active);
template void Simplified_Threats::append_active_indices<BLACK>(const Position& pos, IndexList& active);
template IndexType Simplified_Threats::make_index<WHITE>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);
template IndexType Simplified_Threats::make_index<BLACK>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);
/*
// Get a list of indices for recently changed features
template<Color Perspective>
void Simplified_Threats::append_changed_indices(Square            ksq,
                                         const DirtyPiece& dp,
                                         IndexList&        removed,
                                         IndexList&        added) {
    for (int i = 0; i < dp.dirty_num; ++i)
    {
        if (dp.from[i] != SQ_NONE)
            removed.push_back(make_index<Perspective>(dp.from[i], dp.piece[i], ksq));
        if (dp.to[i] != SQ_NONE)
            added.push_back(make_index<Perspective>(dp.to[i], dp.piece[i], ksq));
    }
}

// Explicit template instantiations
template void Simplified_Threats::append_changed_indices<WHITE>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);
template void Simplified_Threats::append_changed_indices<BLACK>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);

bool Simplified_Threats::requires_refresh(const StateInfo* st, Color perspective) {
    return st->dirtyPiece.piece[0] == make_piece(perspective, KING);
}
*/
}  // namespace Stockfish::Eval::NNUE::Features


