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

#include "pp_3wide.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

// File-major IDs put the two colors and six pawn ranks of each file together.
alignas(64) static constexpr auto PawnIdsBySquare = [] {
    std::array<u8, SQUARE_NB> ids{};
    for (Square sq = SQ_A2; sq <= SQ_H7; ++sq)
        ids[sq] = u8(12 * file_of(sq) + rank_of(sq) - RANK_2);
    return ids;
}();

static constexpr IndexType make_pawn_id(Color color, Square square) {
    assert(square >= SQ_A2 && square <= SQ_H7);
    return PawnIdsBySquare[square] + 6 * int(color);
}

static constexpr IndexType pp_idx(IndexType a, IndexType b) {
    const IndexType hi = std::max(a, b);
    const IndexType lo = std::min(a, b);
    // Same- and adjacent-file partners differ by at most 23.
    assert(hi > lo && hi - lo <= 23);
    return 22 * lo + hi - 1;
}

#ifdef USE_AVX512ICL
static inline __m256i pp_idx_epi16(__m256i a, __m256i b) {
    const __m256i hi     = _mm256_max_epu16(a, b);
    const __m256i lo     = _mm256_min_epu16(a, b);
    const __m256i packed = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 8));
    // Multiply unsigned ID bytes by (22, 1). The sum is at most 23 * 95.
    return _mm256_sub_epi16(_mm256_maddubs_epi16(packed, _mm256_set1_epi16(0x0116)),
                            _mm256_set1_epi16(1));
}
#endif

sf_always_inline IndexType PP_3Wide::make_index(
  Color perspective, Color color, Square from, Square to, Color pairedColor, Square ksq) {
    const u8 orientation   = OrientTBL[ksq] ^ (56 * perspective);
    unsigned from_oriented = u8(from) ^ orientation;
    unsigned to_oriented   = u8(to) ^ orientation;

    Color color_oriented       = Color(color ^ perspective);
    Color pairedColor_oriented = Color(pairedColor ^ perspective);

    assert(from_oriented >= SQ_A2 && from_oriented <= SQ_H7);
    assert(to_oriented >= SQ_A2 && to_oriented <= SQ_H7);

    const IndexType idA = make_pawn_id(color_oriented, Square(from_oriented));
    const IndexType idB = make_pawn_id(pairedColor_oriented, Square(to_oriented));
    return pp_idx(idA, idB);
}

template<usize N>
static void append_indices_impl(Color perspective,
                                const Square (&ksq)[N],
                                Bitboard updated,
                                Bitboard pawnsW,
                                Bitboard pawnsB,
                                PP_3Wide::IndexList* const (&out)[N],
                                [[maybe_unused]] const ThreatWeightType* const (&prefetchBases)[N],
                                [[maybe_unused]] IndexType prefetchStride) {
    static_assert(N == 1 || N == 2);

    if (!updated)
        return;

    u8 orientation[N];
    for (usize i = 0; i < N; ++i)
        orientation[i] = OrientTBL[ksq[i]] ^ u8(56 * (perspective ^ i));

#ifdef USE_AVX512ICL
    __m512i ids[N];
    for (usize i = 0; i < N; ++i)
    {
        const Bitboard enemy = (perspective ^ i) == WHITE ? pawnsB : pawnsW;
        const __m512i  adjusted =
          _mm512_permutexvar_epi8(_mm512_xor_si512(AllSquares, _mm512_set1_epi8(orientation[i])),
                                  _mm512_load_si512(PawnIdsBySquare.data()));
        ids[i] = _mm512_mask_add_epi8(adjusted, enemy, adjusted, _mm512_set1_epi8(6));
    }
#endif

    // Enumerate changed pawns and their partners once for all requested perspectives.
    const Bitboard unchanged = (pawnsW | pawnsB) & ~updated;
    for (Bitboard u = updated; u;)
    {
        const Square   a        = pop_lsb(u);
        const Bitboard partners = pawn_pair_bb(a) & (unchanged | u);
        if (!partners)
            continue;

#ifdef USE_AVX512ICL
        const int n = popcount(partners);
        assert(n <= 16);

        for (usize i = 0; i < N; ++i)
        {
            const Bitboard enemy = (perspective ^ i) == WHITE ? pawnsB : pawnsW;
            const Color    color = (enemy & a) ? BLACK : WHITE;
            const u16      aId   = u16(make_pawn_id(color, Square(u8(a) ^ orientation[i])));
            const __m256i  pids  = _mm256_cvtepu8_epi16(
              _mm512_castsi512_si128(_mm512_maskz_compress_epi8(partners, ids[i])));
            const __m256i feats = pp_idx_epi16(_mm256_set1_epi16(aId), pids);
            // Position::set material limits bound refresh lists to 232 entries:
            // 80 + 8 * (16 - P) + 2 * P + P * (P - 1) / 2, for P <= 16 pawns.
            // Updates have at most 96 threats + 120 pairs. Both fit in 256 entries
            // including the 15 extra lanes written by an unmasked store.
            _mm256_storeu_epi16(out[i]->make_space(n), feats);
        }
#else
        const Color aCol = (pawnsB & a) ? BLACK : WHITE;
        IndexType   aId[N];
        for (usize i = 0; i < N; ++i)
            aId[i] = make_pawn_id(Color(aCol ^ perspective ^ i), Square(u8(a) ^ orientation[i]));

        auto push = [&](Square to, Color pairedColor) {
            for (usize i = 0; i < N; ++i)
            {
                const IndexType bId   = make_pawn_id(Color(pairedColor ^ perspective ^ i),
                                                     Square(u8(to) ^ orientation[i]));
                const IndexType index = pp_idx(aId[i], bId);
                if (prefetchBases[i])
                    prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(
                      reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(prefetchBases[i])
                                                    + uintptr_t(index) * prefetchStride));
                out[i]->push_back(index);
            }
        };

        for (Bitboard pb = pawnsB & partners; pb;)
            push(pop_lsb(pb), BLACK);
        for (Bitboard pw = pawnsW & partners; pw;)
            push(pop_lsb(pw), WHITE);
#endif
    }
}

template<usize N>
static void append_changed_indices_impl(Color perspective,
                                        const Square (&ksq)[N],
                                        const PP_3Wide::DiffType& diff,
                                        PP_3Wide::IndexList* const (&removed)[N],
                                        PP_3Wide::IndexList* const (&added)[N],
                                        const ThreatWeightType* const (&prefetchBases)[N],
                                        IndexType prefetchStride) {
    const Bitboard whiteBefore = diff.before[WHITE];
    const Bitboard blackBefore = diff.before[BLACK];
    const Bitboard whiteAfter  = diff.after[WHITE];
    const Bitboard blackAfter  = diff.after[BLACK];

    if (whiteBefore == whiteAfter && blackBefore == blackAfter)
        return;

    append_indices_impl<N>(perspective, ksq,
                           (whiteAfter & ~whiteBefore) | (blackAfter & ~blackBefore), whiteAfter,
                           blackAfter, added, prefetchBases, prefetchStride);
    append_indices_impl<N>(perspective, ksq,
                           (whiteBefore & ~whiteAfter) | (blackBefore & ~blackAfter), whiteBefore,
                           blackBefore, removed, prefetchBases, prefetchStride);
}

void PP_3Wide::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Bitboard white = pos.pieces(WHITE, PAWN);
    const Bitboard black = pos.pieces(BLACK, PAWN);
    append_indices_impl<1>(perspective, {pos.square<KING>(perspective)}, white | black, white,
                           black, {&active}, {nullptr}, 0);
}

void PP_3Wide::append_changed_indices(Color                   perspective,
                                      Square                  ksq,
                                      const DiffType&         diff,
                                      IndexList&              removed,
                                      IndexList&              added,
                                      const ThreatWeightType* prefetchBase,
                                      IndexType               prefetchStride) {
    append_changed_indices_impl<1>(perspective, {ksq}, diff, {&removed}, {&added}, {prefetchBase},
                                   prefetchStride);
}

void PP_3Wide::append_changed_indices_both(Square          white_ksq,
                                           Square          black_ksq,
                                           const DiffType& diff,
                                           IndexList&      white_removed,
                                           IndexList&      white_added,
                                           IndexList&      black_removed,
                                           IndexList&      black_added,
                                           const ThreatWeightType* const (&prefetchBases)[COLOR_NB],
                                           IndexType prefetchStride) {
    append_changed_indices_impl<2>(WHITE, {white_ksq, black_ksq}, diff,
                                   {&white_removed, &black_removed}, {&white_added, &black_added},
                                   prefetchBases, prefetchStride);
}

}  // namespace Stockfish::Eval::NNUE::Features
