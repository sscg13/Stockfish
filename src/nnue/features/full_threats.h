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

//Definition of input features Full_Threats of NNUE evaluation function

#ifndef NNUE_FEATURES_FULL_THREATS_INCLUDED
#define NNUE_FEATURES_FULL_THREATS_INCLUDED


#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class FullThreats {
   public:
    // Hash value embedded in the evaluation file
    static constexpr u32 HashValue = 0x4b1f6a2du;

    // Number of feature dimensions
    static constexpr IndexType Dimensions = 59808;

    // clang-format off
    // Orient a square according to perspective (rotates by 180 for black)
    static constexpr i8 OrientTBL[SQUARE_NB] = {
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
    };

    // Slot index for each (AttackType, TargetType) pair.
    // -1 = fully excluded (never a valid feature).
    // >= 0 = contiguous slot index used to compute the feature base offset.
    // The gap AT rows (6,7 and 14,15) and gap TT columns (5-7) are all -1.
    // PAWN_DIAG does not target pawns; pawn-on-pawn co-presence is captured by PP_3Wide.
    static constexpr std::int8_t slot_map[ATTACK_TYPE_NB][TARGET_TYPE_NB] = {
      //                  W_P  W_N  W_B  W_R  W_Q  g5   g6   g7   B_P  B_N  B_B  B_R  B_Q
      /* W_PAWN_DIAG */ { -1,   0,  -1,   1,  -1,  -1,  -1,  -1,  -1,   2,  -1,   3,  -1},
      /* W_PAWN_PAIR */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
      /* W_KNIGHT    */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
      /* W_BISHOP    */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
      /* W_ROOK      */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
      /* W_QUEEN     */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
      /* gap_6       */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
      /* gap_7       */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
      /* B_PAWN_DIAG */ { -1,   0,  -1,   1,  -1,  -1,  -1,  -1,  -1,   2,  -1,   3,  -1},
      /* B_PAWN_PAIR */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
      /* B_KNIGHT    */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
      /* B_BISHOP    */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
      /* B_ROOK      */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
      /* B_QUEEN     */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
    };

    // Semi-exclusion: true = only active when from_oriented >= to_oriented.
    // Encodes symmetric pairs (mutual attacks / same-type defences) so each
    // pair is represented by exactly one feature index.
    static constexpr bool semi_map[ATTACK_TYPE_NB][TARGET_TYPE_NB] = {
      //                  W_P    W_N    W_B    W_R    W_Q    g5     g6     g7     B_P    B_N    B_B    B_R    B_Q
      /* W_PAWN_DIAG */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
      /* W_PAWN_PAIR */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
      /* W_KNIGHT    */ {false,  true, false, false, false, false, false, false, false,  true, false, false, false},
      /* W_BISHOP    */ {false, false,  true, false, false, false, false, false, false, false,  true, false, false},
      /* W_ROOK      */ {false, false, false,  true, false, false, false, false, false, false, false,  true, false},
      /* W_QUEEN     */ {false, false, false, false,  true, false, false, false, false, false, false, false,  true},
      /* gap_6       */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
      /* gap_7       */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
      /* B_PAWN_DIAG */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
      /* B_PAWN_PAIR */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
      /* B_KNIGHT    */ {false,  true, false, false, false, false, false, false, false,  true, false, false, false},
      /* B_BISHOP    */ {false, false,  true, false, false, false, false, false, false, false,  true, false, false},
      /* B_ROOK      */ {false, false, false,  true, false, false, false, false, false, false, false,  true, false},
      /* B_QUEEN     */ {false, false, false, false,  true, false, false, false, false, false, false, false,  true},
    };
    // clang-format on

    // Maximum number of simultaneously active features.
    static constexpr IndexType MaxActiveDimensions = 224;
    using IndexList                                = ValueList<IndexType, MaxActiveDimensions>;
    using DiffType                                 = DirtyThreats;

    static IndexType
    make_index(Color perspective, AttackType at, Square from, Square to, TargetType tt, Square ksq);

    // Get a list of indices for active features
    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Get a list of indices for recently changed features
    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const DiffType&         diff,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_FULL_THREATS_INCLUDED
