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

#include "full_threats.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_accumulator.h"

namespace Stockfish::Eval::NNUE::Features {
//lookup array for indexing threats
void Full_Threats::init_threat_offsets() {
    int pieceoffset = 0;
    PieceType piecetbl[PIECE_NB] = {NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE,
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE};
    IndexType numvalidtargets[PIECE_NB] = {0, 6, 12, 10, 10, 12, 8, 0, 0, 6, 12, 10, 10, 12, 8, 0};
    for (int piece = 0; piece < 16; piece++) {
        if (piecetbl[piece] != NO_PIECE_TYPE) {
            threatoffsets[piece][65] = pieceoffset;
            int squareoffset = 0;
            for (int from = SQ_A1; from <= SQ_H8; from++) {
                threatoffsets[piece][from] = squareoffset;
                if (piecetbl[piece] != PAWN) {
                    Bitboard attacks = attacks_bb(piecetbl[piece], Square(from), 0ULL);
                    squareoffset += popcount(attacks);
                }
                else if (from >= SQ_A2 && from <= SQ_H7) {
                    Bitboard attacks = pawn_attacks_bb(Color(piece/8), Square(from));
                    squareoffset += popcount(attacks);
                }
            }
            threatoffsets[piece][64] = squareoffset;
            pieceoffset += numvalidtargets[piece]*squareoffset;
        }
    }
}

// Index of a feature for a given king position and another piece on some square
template<Color Perspective>
IndexType Full_Threats::make_index(Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
    bool enemy = (color_of(attkr) != color_of(attkd));
    from = (Square)(int(from) ^ OrientTBL[Perspective][ksq]);
    to = (Square)(int(to) ^ OrientTBL[Perspective][ksq]);
    if (Perspective == BLACK) {
        attkr = ~attkr;
        attkd = ~attkd;
    }
    if (from == to) {
        return IndexType(PieceSquareIndex[attkr]+from);
    }
    Bitboard attacks = (type_of(attkr) == PAWN) ? pawn_attacks_bb(color_of(attkr), from) : attacks_bb(type_of(attkr), from, 0ULL);
    return IndexType(768 + threatoffsets[attkr][65] + enemy*threatoffsets[attkr][64]
                     + threatoffsets[attkr][from] + popcount((square_bb(to)-1) & attacks));
}

bool Full_Threats::is_duplicate(Piece attkr, Square from, Square to, Piece attkd) {
    return false;
}

// Get a list of indices for active features in ascending order
template<Color Perspective>
void Full_Threats::append_active_threats(const Position& pos, IndexList& active) {
    Square ksq = pos.square<KING>(Perspective);
    Color order[2][2] = {{WHITE, BLACK}, {BLACK, WHITE}};
    std::vector<int> pieces;
    for (int i = WHITE; i <= BLACK; i++) {
        for (int j = PAWN; j <= KING; j++) {
            Color c = order[Perspective][i];
            PieceType pt = PieceType(j);
            Piece attkr = make_piece(c, pt);
            Bitboard bb  = pos.pieces(c, pt);
            while (bb)
            {
                Square from = pop_lsb(bb);
                Bitboard attacks = ((pt == PAWN) ? pawn_attacks_bb(c, from) : attacks_bb(pt, from, pos.pieces())) & pos.pieces();
                while (attacks) {
                    Square to = pop_lsb(attacks);
                    Piece attkd = pos.piece_on(to);
                    pieces.push_back(make_index<Perspective>(attkr, from, to, attkd, ksq));
                }
            }
            std::sort(pieces.begin(), pieces.end());
            for (auto threat : pieces) {
                active.push_back(threat);
            }
            pieces.clear();
        }
    }
}

template<Color Perspective>
void Full_Threats::append_active_psq(const Position& pos, IndexList& active) {
    Square   ksq = pos.square<KING>(Perspective);
    Bitboard bb  = pos.pieces();
    while (bb)
    {
        Square s = pop_lsb(bb);
        Piece pc = pos.piece_on(s);
        active.push_back(make_index<Perspective>(pc, s, s, pc, ksq));
    }
}

// Explicit template instantiations
template void Full_Threats::append_active_threats<WHITE>(const Position& pos, IndexList& active);
template void Full_Threats::append_active_threats<BLACK>(const Position& pos, IndexList& active);
template void Full_Threats::append_active_psq<WHITE>(const Position& pos, IndexList& active);
template void Full_Threats::append_active_psq<BLACK>(const Position& pos, IndexList& active);
template IndexType Full_Threats::make_index<WHITE>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);
template IndexType Full_Threats::make_index<BLACK>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);
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


