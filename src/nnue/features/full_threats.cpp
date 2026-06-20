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

//Definition of input features FullThreats of NNUE evaluation function

#include "full_threats.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

struct HelperOffsets {
    int cumulativePieceOffset, cumulativeOffset;
};

constexpr Bitboard pseudo_attack_bb(AttackType at, Square s) {
    switch (at)
    {
    case W_PAWN_DIAG_AT :
        return Attacks::PseudoAttacks[WHITE][s];
    case B_PAWN_DIAG_AT :
        return Attacks::PseudoAttacks[BLACK][s];
    case W_KNIGHT_AT :
    case B_KNIGHT_AT :
        return Attacks::PseudoAttacks[KNIGHT][s];
    case W_BISHOP_AT :
    case B_BISHOP_AT :
        return Attacks::PseudoAttacks[BISHOP][s];
    case W_ROOK_AT :
    case B_ROOK_AT :
        return Attacks::PseudoAttacks[ROOK][s];
    case W_QUEEN_AT :
    case B_QUEEN_AT :
        return Attacks::PseudoAttacks[QUEEN][s];
    default :
        return 0;
    }
}

constexpr auto init_threat_offsets() {
    std::array<HelperOffsets, ATTACK_TYPE_NB>                    indices{};
    std::array<std::array<IndexType, SQUARE_NB>, ATTACK_TYPE_NB> offsets{};

    auto num_slots = [](int at) constexpr {
        int count = 0;
        for (int tt = 0; tt < TARGET_TYPE_NB; ++tt)
            if (FullThreats::slot_map[at][tt] >= 0)
                ++count;
        return count;
    };

    int cumulativeOffset = 0;
    for (int at = 0; at < ATTACK_TYPE_NB; ++at)
    {
        int  cumulativePieceOffset = 0;
        bool isPawnType            = (at == W_PAWN_DIAG_AT || at == B_PAWN_DIAG_AT);

        for (Square from = SQ_A1; from <= SQ_H8; ++from)
        {
            offsets[at][from] = cumulativePieceOffset;
            bool inPawnRange  = (from >= SQ_A2 && from <= SQ_H7);
            if (!isPawnType || inPawnRange)
                cumulativePieceOffset += constexpr_popcount(pseudo_attack_bb(AttackType(at), from));
        }

        indices[at] = {cumulativePieceOffset, cumulativeOffset};
        cumulativeOffset += num_slots(at) * cumulativePieceOffset;
    }

    return std::pair{indices, offsets};
}

constexpr auto helper_offsets = init_threat_offsets().first;
// Lookup array for indexing threats
constexpr auto offsets = init_threat_offsets().second;

constexpr auto init_index_luts() {
    std::array<std::array<std::array<u32, 2>, 16>, ATTACK_TYPE_NB> lut{};

    for (int at = 0; at < ATTACK_TYPE_NB; ++at)
    {
        for (int tt = 0; tt < TARGET_TYPE_NB; ++tt)
        {
            int8_t slot = FullThreats::slot_map[at][tt];
            bool   semi = FullThreats::semi_map[at][tt];

            if (slot < 0)
            {
                lut[at][tt][0] = lut[at][tt][1] = FullThreats::Dimensions;
            }
            else
            {
                IndexType feature = helper_offsets[at].cumulativeOffset
                                  + slot * helper_offsets[at].cumulativePieceOffset;
                lut[at][tt][0]    = feature;
                lut[at][tt][1]    = semi ? FullThreats::Dimensions : feature;
            }
        }
    }

    return lut;
}

constexpr auto index_lut2_array() {
    std::array<std::array<std::array<uint8_t, SQUARE_NB>, SQUARE_NB>, ATTACK_TYPE_NB> out{};

    for (int at = 0; at < ATTACK_TYPE_NB; ++at)
        for (Square from = SQ_A1; from <= SQ_H8; ++from)
        {
            Bitboard attacks = pseudo_attack_bb(AttackType(at), from);
            for (Square to = SQ_A1; to <= SQ_H8; ++to)
                out[at][from][to] = constexpr_popcount(((1ULL << to) - 1) & attacks);
        }

    return out;
}

// The final index is calculated from summing data found in these two LUTs, as well
// as offsets[attacker][from]

// [attacker][attacked][from < to]
constexpr auto index_lut1 = init_index_luts();
// [attacker][from][to]
constexpr auto index_lut2 = index_lut2_array();

static_assert([] {
    auto [h, o] = init_threat_offsets();
    int total   = 0;
    for (int at = 0; at < ATTACK_TYPE_NB; ++at)
    {
        int slots = 0;
        for (int tt = 0; tt < TARGET_TYPE_NB; ++tt)
            if (FullThreats::slot_map[at][tt] >= 0)
                ++slots;
        total += slots * h[at].cumulativePieceOffset;
    }
    return total == FullThreats::Dimensions;
}());

// Index of a feature for a given king position and threat relationship
inline sf_always_inline IndexType FullThreats::make_index(
  Color perspective, AttackType at, Square from, Square to, TargetType tt, Square ksq) {
    const i8 orientation   = OrientTBL[ksq] ^ (56 * perspective);
    unsigned from_oriented = u8(from) ^ orientation;
    unsigned to_oriented   = u8(to) ^ orientation;

    AttackType at_oriented = AttackType(u8(at) ^ (u8(perspective) << 3));
    TargetType tt_oriented = TargetType(u8(tt) ^ (u8(perspective) << 3));

    return index_lut1[at_oriented][tt_oriented][from_oriented < to_oriented]
         + offsets[at_oriented][from_oriented]
         + index_lut2[at_oriented][from_oriented][to_oriented];
}

// Get a list of indices for active features in ascending order
void FullThreats::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq         = pos.square<KING>(perspective);
    const Bitboard occupied    = pos.pieces();
    const Bitboard occupiedNoK = occupied ^ pos.pieces(KING);
    const Bitboard pawns       = pos.pieces(PAWN);

    for (Color color : {WHITE, BLACK})
    {
        const Color c = Color(perspective ^ color);

        {
            const AttackType at     = make_attack_type(make_piece(c, PAWN));
            const Bitboard   cPawns = pos.pieces(c, PAWN);

            const Bitboard diagTargets  = occupiedNoK & ~pawns;
            auto           process_pawn = [&](Bitboard targets, Direction dir) {
                while (targets)
                {
                    Square     to       = pop_lsb(targets);
                    Square     from     = to - dir;
                    Piece      attacked = pos.piece_on(to);
                    TargetType tt       = make_target_type(attacked);
                    IndexType  index    = make_index(perspective, at, from, to, tt, ksq);
                    active.push_back_if_lt(index, Dimensions);
                }
            };

            if (c == WHITE)
            {
                process_pawn(shift<NORTH_EAST>(cPawns) & diagTargets, NORTH_EAST);
                process_pawn(shift<NORTH_WEST>(cPawns) & diagTargets, NORTH_WEST);
            }
            else
            {
                process_pawn(shift<SOUTH_WEST>(cPawns) & diagTargets, SOUTH_WEST);
                process_pawn(shift<SOUTH_EAST>(cPawns) & diagTargets, SOUTH_EAST);
            }
        }

        for (PieceType pt = KNIGHT; pt < KING; ++pt)
        {
            Piece      attacker = make_piece(c, pt);
            AttackType at       = make_attack_type(attacker);
            Bitboard   bb       = pos.pieces(c, pt);
            while (bb)
            {
                Square   from    = pop_lsb(bb);
                Bitboard attacks = Attacks::attacks_bb(pt, from, occupied) & occupiedNoK;
                while (attacks)
                {
                    Square     to       = pop_lsb(attacks);
                    Piece      attacked = pos.piece_on(to);
                    TargetType tt       = make_target_type(attacked);
                    IndexType  index    = make_index(perspective, at, from, to, tt, ksq);
                    active.push_back_if_lt(index, Dimensions);
                }
            }
        }
    }
}

// Get a list of indices for recently changed features
void FullThreats::append_changed_indices(Color                   perspective,
                                         Square                  ksq,
                                         const DiffType&         diff,
                                         IndexList&              removed,
                                         IndexList&              added,
                                         const ThreatWeightType* prefetchBase,
                                         IndexType               prefetchStride) {

    for (const auto& dirty : diff.list)
    {
        auto at   = dirty.attack_type();
        auto tt   = dirty.target_type();
        auto from = dirty.pc_sq();
        auto to   = dirty.threatened_sq();
        auto add  = dirty.add();

        auto&           insert = add ? added : removed;
        const IndexType index  = make_index(perspective, at, from, to, tt, ksq);

        if (prefetchBase)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + index * prefetchStride));
        insert.push_back_if_lt(index, Dimensions);
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
