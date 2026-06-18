/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

// Definition of pawn-pair input features for the NNUE evaluation function

#include "pp_3wide.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "full_threats.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

constexpr IndexType make_pawn_id(Color color, Square square) {
    assert(square >= SQ_A2 && square <= SQ_H7);
    return 48 * int(color) + square - SQ_A2;
}

inline sf_always_inline IndexType PP_3Wide::make_index(
  Color perspective, Color color, Square from, Square to, Color pairedColor, Square ksq) {
    const i8 orientation   = FullThreats::OrientTBL[ksq] ^ (56 * perspective);
    unsigned from_oriented = u8(from) ^ orientation;
    unsigned to_oriented   = u8(to) ^ orientation;

    Color color_oriented       = Color(color ^ perspective);
    Color pairedColor_oriented = Color(pairedColor ^ perspective);

    if (from_oriented < SQ_A2 || from_oriented > SQ_H7 || to_oriented < SQ_A2
        || to_oriented > SQ_H7)
        return Dimensions;

    const IndexType idA = make_pawn_id(color_oriented, Square(from_oriented));
    const IndexType idB = make_pawn_id(pairedColor_oriented, Square(to_oriented));
    const IndexType hi  = std::max(idA, idB);
    const IndexType lo  = std::min(idA, idB);

    return hi * (hi - 1) / 2 + lo;
}

void PP_3Wide::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq   = pos.square<KING>(perspective);
    const Bitboard pawns = pos.pieces(PAWN);

    for (Color color : {WHITE, BLACK})
    {
        const Color c  = Color(perspective ^ color);
        Bitboard    bb = pos.pieces(c, PAWN);

        while (bb)
        {
            Square   from    = pop_lsb(bb);
            Bitboard targets = pawn_pair_bb(from) & pawns;

            while (targets)
            {
                Square to = pop_lsb(targets);
                if (from < to)
                    continue;
                Color     pairedColor = color_of(pos.piece_on(to));
                IndexType index       = make_index(perspective, c, from, to, pairedColor, ksq);
                active.push_back_if_lt(index, Dimensions);
            }
        }
    }
}

void PP_3Wide::append_changed_indices(Color                   perspective,
                                      Square                  ksq,
                                      const DiffType&         diff,
                                      IndexList&              removed,
                                      IndexList&              added,
                                      const ThreatWeightType* prefetchBase,
                                      IndexType               prefetchStride) {

    for (const auto& dirty : diff.list)
    {
        auto color       = dirty.color();
        auto pairedColor = dirty.paired_color();
        auto from        = dirty.pc_sq();
        auto to          = dirty.paired_sq();
        auto add         = dirty.add();

        auto&           insert = add ? added : removed;
        const IndexType index  = make_index(perspective, color, from, to, pairedColor, ksq);

        if (index < Dimensions)
        {
            if (prefetchBase)
                prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
                  reinterpret_cast<uintptr_t>(prefetchBase) + index * prefetchStride));
            insert.push_back(index);
        }
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
