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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <type_traits>
#include <utility>

#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

using BiasType       = std::int16_t;
using WeightType     = std::int16_t;

enum IncUpdateDirection {
    FORWARD,
    BACKWARDS
};

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
#define VECTOR

#ifdef USE_AVX512
using vec_t      = __m512i;
using psqt_vec_t = __m256i;
    #define vec_load(a) _mm512_load_si512(a)
    #define vec_store(a, b) _mm512_store_si512(a, b)
    #define vec_add_16(a, b) _mm512_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
    #define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
    #define vec_zero() _mm512_setzero_epi32()
    #define vec_set_16(a) _mm512_set1_epi16(a)
    #define vec_max_16(a, b) _mm512_max_epi16(a, b)
    #define vec_min_16(a, b) _mm512_min_epi16(a, b)
    #define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
    // Inverse permuted at load time
    #define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()
    #define NumRegistersSIMD 16
    #define MaxChunkSize 64

#elif USE_AVX2
using vec_t      = __m256i;
using psqt_vec_t = __m256i;
    #define vec_load(a) _mm256_load_si256(a)
    #define vec_store(a, b) _mm256_store_si256(a, b)
    #define vec_add_16(a, b) _mm256_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
    #define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
    #define vec_zero() _mm256_setzero_si256()
    #define vec_set_16(a) _mm256_set1_epi16(a)
    #define vec_max_16(a, b) _mm256_max_epi16(a, b)
    #define vec_min_16(a, b) _mm256_min_epi16(a, b)
    #define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
    // Inverse permuted at load time
    #define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()
    #define NumRegistersSIMD 16
    #define MaxChunkSize 32

#elif USE_SSE2
using vec_t      = __m128i;
using psqt_vec_t = __m128i;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) _mm_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm_sub_epi16(a, b)
    #define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
    #define vec_zero() _mm_setzero_si128()
    #define vec_set_16(a) _mm_set1_epi16(a)
    #define vec_max_16(a, b) _mm_max_epi16(a, b)
    #define vec_min_16(a, b) _mm_min_epi16(a, b)
    #define vec_slli_16(a, b) _mm_slli_epi16(a, b)
    #define vec_packus_16(a, b) _mm_packus_epi16(a, b)
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) _mm_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm_sub_epi32(a, b)
    #define vec_zero_psqt() _mm_setzero_si128()
    #define NumRegistersSIMD (Is64Bit ? 16 : 8)
    #define MaxChunkSize 16

#elif USE_NEON
using vec_t      = int16x8_t;
using psqt_vec_t = int32x4_t;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) vaddq_s16(a, b)
    #define vec_sub_16(a, b) vsubq_s16(a, b)
    #define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
    #define vec_zero() \
        vec_t { 0 }
    #define vec_set_16(a) vdupq_n_s16(a)
    #define vec_max_16(a, b) vmaxq_s16(a, b)
    #define vec_min_16(a, b) vminq_s16(a, b)
    #define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
    #define vec_packus_16(a, b) reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) vaddq_s32(a, b)
    #define vec_sub_psqt_32(a, b) vsubq_s32(a, b)
    #define vec_zero_psqt() \
        psqt_vec_t { 0 }
    #define NumRegistersSIMD 16
    #define MaxChunkSize 16

#else
    #undef VECTOR

#endif

// Returns the inverse of a permutation
template<std::size_t Len>
constexpr std::array<std::size_t, Len>
invert_permutation(const std::array<std::size_t, Len>& order) {
    std::array<std::size_t, Len> inverse{};
    for (std::size_t i = 0; i < order.size(); i++)
        inverse[order[i]] = i;
    return inverse;
}

// Divide a byte region of size TotalSize to chunks of size
// BlockSize, and permute the blocks by a given order
template<std::size_t BlockSize, typename T, std::size_t N, std::size_t OrderSize>
void permute(T (&data)[N], const std::array<std::size_t, OrderSize>& order) {
    constexpr std::size_t TotalSize = N * sizeof(T);

    static_assert(TotalSize % (BlockSize * OrderSize) == 0,
                  "ChunkSize * OrderSize must perfectly divide TotalSize");

    constexpr std::size_t ProcessChunkSize = BlockSize * OrderSize;

    std::array<std::byte, ProcessChunkSize> buffer{};

    std::byte* const bytes = reinterpret_cast<std::byte*>(data);

    for (std::size_t i = 0; i < TotalSize; i += ProcessChunkSize)
    {
        std::byte* const values = &bytes[i];

        for (std::size_t j = 0; j < OrderSize; j++)
        {
            auto* const buffer_chunk = &buffer[j * BlockSize];
            auto* const value_chunk  = &values[order[j] * BlockSize];

            std::copy(value_chunk, value_chunk + BlockSize, buffer_chunk);
        }

        std::copy(std::begin(buffer), std::end(buffer), values);
    }
}

// Compute optimal SIMD register count for feature transformer accumulation.
template<IndexType TransformedFeatureWidth, IndexType HalfDimensions>
class SIMDTiling {
#ifdef VECTOR
    // We use __m* types as template arguments, which causes GCC to emit warnings
    // about losing some attribute information. This is irrelevant to us as we
    // only take their size, so the following pragma are harmless.
    #if defined(__GNUC__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif

    template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
    static constexpr int BestRegisterCount() {
        constexpr std::size_t RegisterSize = sizeof(SIMDRegisterType);
        constexpr std::size_t LaneSize     = sizeof(LaneType);

        static_assert(RegisterSize >= LaneSize);
        static_assert(MaxRegisters <= NumRegistersSIMD);
        static_assert(MaxRegisters > 0);
        static_assert(NumRegistersSIMD > 0);
        static_assert(RegisterSize % LaneSize == 0);
        static_assert((NumLanes * LaneSize) % RegisterSize == 0);

        const int ideal = (NumLanes * LaneSize) / RegisterSize;
        if (ideal <= MaxRegisters)
            return ideal;

        // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
        for (int divisor = MaxRegisters; divisor > 1; --divisor)
            if (ideal % divisor == 0)
                return divisor;

        return 1;
    }

    #if defined(__GNUC__)
        #pragma GCC diagnostic pop
    #endif

   public:
    static constexpr int NumRegs =
      BestRegisterCount<vec_t, WeightType, TransformedFeatureWidth, NumRegistersSIMD>();
    static constexpr int NumPsqtRegs =
      BestRegisterCount<psqt_vec_t, PSQTWeightType, PSQTBuckets, NumRegistersSIMD>();

    static constexpr IndexType TileHeight     = NumRegs * sizeof(vec_t) / 2;
    static constexpr IndexType PsqtTileHeight = NumPsqtRegs * sizeof(psqt_vec_t) / 4;

    static_assert(HalfDimensions % TileHeight == 0, "TileHeight must divide HalfDimensions");
    static_assert(PSQTBuckets % PsqtTileHeight == 0, "PsqtTileHeight must divide PSQTBuckets");
#endif
};


// Input feature converter
template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
class FeatureTransformer {

    // Number of output dimensions for one side
    static constexpr IndexType HalfDimensions = TransformedFeatureDimensions;

   private:
    using Tiling = SIMDTiling<TransformedFeatureDimensions, HalfDimensions>;

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = FeatureSet::Dimensions;
    static constexpr IndexType OutputDimensions = TransformedFeatureDimensions;

    // Size of forward propagation buffer
    static constexpr std::size_t BufferSize = OutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        return FeatureSet::HashValue ^ (OutputDimensions * 2);
    }

    inline void scale_weights(bool read) {
        for (IndexType j = 0; j < InputDimensions; ++j)
        {
            WeightType* w = &weights[j * HalfDimensions];
            for (IndexType i = 0; i < HalfDimensions; ++i)
                w[i] = read ? w[i] * 2 : w[i] / 2;
        }

        for (IndexType i = 0; i < HalfDimensions; ++i)
            biases[i] = read ? biases[i] * 2 : biases[i] / 2;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<WeightType>(stream, weights, TransformedFeatureDimensions * InputDimensions);
        read_little_endian<BiasType>(stream, biases, TransformedFeatureDimensions);
        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) {
        write_little_endian<WeightType>(stream, weights, TransformedFeatureDimensions * InputDimensions);
        read_little_endian<BiasType>(stream, biases, TransformedFeatureDimensions);
        return !stream.fail();
    }

   private:
    // Given a computed accumulator, computes the accumulator of another position.
    template<Color Perspective, IncUpdateDirection Direction = FORWARD>
    void update_accumulator_incremental(const Square     ksq,
                                        StateInfo*       target_state,
                                        const StateInfo* computed) const {
        [[maybe_unused]] constexpr bool Forward   = Direction == FORWARD;
        [[maybe_unused]] constexpr bool Backwards = Direction == BACKWARDS;
        assert((computed->*accPtr).computed[Perspective]);

        StateInfo* next = Forward ? computed->next : computed->previous;

        assert(next != nullptr);
        assert(!(next->*accPtr).computed[Perspective]);

        // The size must be enough to contain the largest possible update.
        // That might depend on the feature set and generally relies on the
        // feature set's update cost calculation to be correct and never allow
        // updates with more added/removed features than MaxActiveDimensions.
        // In this case, the maximum size of both feature addition and removal
        // is 2, since we are incrementally updating one move at a time.
        FeatureSet::IndexList removed, added;
        if constexpr (Forward)
            FeatureSet::append_changed_indices<Perspective>(ksq, next->dirtyPiece, removed, added);
        else
            FeatureSet::append_changed_indices<Perspective>(ksq, computed->dirtyPiece, added,
                                                            removed);

        if (removed.size() == 0 && added.size() == 0)
        {
            std::memcpy((next->*accPtr).accumulation[Perspective],
                        (computed->*accPtr).accumulation[Perspective],
                        HalfDimensions * sizeof(BiasType));
            std::memcpy((next->*accPtr).psqtAccumulation[Perspective],
                        (computed->*accPtr).psqtAccumulation[Perspective],
                        PSQTBuckets * sizeof(PSQTWeightType));
        }
        else
        {
            assert(added.size() == 1 || added.size() == 2);
            assert(removed.size() == 1 || removed.size() == 2);
            if (Forward)
                assert(added.size() <= removed.size());
            else
                assert(removed.size() <= added.size());

#ifdef VECTOR
            auto* accIn =
              reinterpret_cast<const vec_t*>(&(computed->*accPtr).accumulation[Perspective][0]);
            auto* accOut = reinterpret_cast<vec_t*>(&(next->*accPtr).accumulation[Perspective][0]);

            const IndexType offsetA0 = HalfDimensions * added[0];
            auto*           columnA0 = reinterpret_cast<const vec_t*>(&weights[offsetA0]);
            const IndexType offsetR0 = HalfDimensions * removed[0];
            auto*           columnR0 = reinterpret_cast<const vec_t*>(&weights[offsetR0]);

            if ((Forward && removed.size() == 1) || (Backwards && added.size() == 1))
            {
                assert(added.size() == 1 && removed.size() == 1);
                for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                    accOut[i] = vec_add_16(vec_sub_16(accIn[i], columnR0[i]), columnA0[i]);
            }
            else if (Forward && added.size() == 1)
            {
                assert(removed.size() == 2);
                const IndexType offsetR1 = HalfDimensions * removed[1];
                auto*           columnR1 = reinterpret_cast<const vec_t*>(&weights[offsetR1]);

                for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                    accOut[i] = vec_sub_16(vec_add_16(accIn[i], columnA0[i]),
                                           vec_add_16(columnR0[i], columnR1[i]));
            }
            else if (Backwards && removed.size() == 1)
            {
                assert(added.size() == 2);
                const IndexType offsetA1 = HalfDimensions * added[1];
                auto*           columnA1 = reinterpret_cast<const vec_t*>(&weights[offsetA1]);

                for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                    accOut[i] = vec_add_16(vec_add_16(accIn[i], columnA0[i]),
                                           vec_sub_16(columnA1[i], columnR0[i]));
            }
            else
            {
                assert(added.size() == 2 && removed.size() == 2);
                const IndexType offsetA1 = HalfDimensions * added[1];
                auto*           columnA1 = reinterpret_cast<const vec_t*>(&weights[offsetA1]);
                const IndexType offsetR1 = HalfDimensions * removed[1];
                auto*           columnR1 = reinterpret_cast<const vec_t*>(&weights[offsetR1]);

                for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                    accOut[i] =
                      vec_add_16(accIn[i], vec_sub_16(vec_add_16(columnA0[i], columnA1[i]),
                                                      vec_add_16(columnR0[i], columnR1[i])));
            }

            auto* accPsqtIn = reinterpret_cast<const psqt_vec_t*>(
              &(computed->*accPtr).psqtAccumulation[Perspective][0]);
            auto* accPsqtOut =
              reinterpret_cast<psqt_vec_t*>(&(next->*accPtr).psqtAccumulation[Perspective][0]);

            const IndexType offsetPsqtA0 = PSQTBuckets * added[0];
            auto* columnPsqtA0 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtA0]);
            const IndexType offsetPsqtR0 = PSQTBuckets * removed[0];
            auto* columnPsqtR0 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR0]);

            if ((Forward && removed.size() == 1)
                || (Backwards && added.size() == 1))  // added.size() == removed.size() == 1
            {
                for (std::size_t i = 0;
                     i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t); ++i)
                    accPsqtOut[i] = vec_add_psqt_32(vec_sub_psqt_32(accPsqtIn[i], columnPsqtR0[i]),
                                                    columnPsqtA0[i]);
            }
            else if (Forward && added.size() == 1)
            {
                const IndexType offsetPsqtR1 = PSQTBuckets * removed[1];
                auto*           columnPsqtR1 =
                  reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR1]);

                for (std::size_t i = 0;
                     i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t); ++i)
                    accPsqtOut[i] =
                      vec_sub_psqt_32(vec_add_psqt_32(accPsqtIn[i], columnPsqtA0[i]),
                                      vec_add_psqt_32(columnPsqtR0[i], columnPsqtR1[i]));
            }
            else if (Backwards && removed.size() == 1)
            {
                const IndexType offsetPsqtA1 = PSQTBuckets * added[1];
                auto*           columnPsqtA1 =
                  reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtA1]);

                for (std::size_t i = 0;
                     i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t); ++i)
                    accPsqtOut[i] =
                      vec_add_psqt_32(vec_add_psqt_32(accPsqtIn[i], columnPsqtA0[i]),
                                      vec_sub_psqt_32(columnPsqtA1[i], columnPsqtR0[i]));
            }
            else
            {
                const IndexType offsetPsqtA1 = PSQTBuckets * added[1];
                auto*           columnPsqtA1 =
                  reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtA1]);
                const IndexType offsetPsqtR1 = PSQTBuckets * removed[1];
                auto*           columnPsqtR1 =
                  reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR1]);

                for (std::size_t i = 0;
                     i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t); ++i)
                    accPsqtOut[i] = vec_add_psqt_32(
                      accPsqtIn[i],
                      vec_sub_psqt_32(vec_add_psqt_32(columnPsqtA0[i], columnPsqtA1[i]),
                                      vec_add_psqt_32(columnPsqtR0[i], columnPsqtR1[i])));
            }
#else
            std::memcpy((next->*accPtr).accumulation[Perspective],
                        (computed->*accPtr).accumulation[Perspective],
                        HalfDimensions * sizeof(BiasType));
            std::memcpy((next->*accPtr).psqtAccumulation[Perspective],
                        (computed->*accPtr).psqtAccumulation[Perspective],
                        PSQTBuckets * sizeof(PSQTWeightType));

            // Difference calculation for the deactivated features
            for (const auto index : removed)
            {
                const IndexType offset = HalfDimensions * index;
                for (IndexType i = 0; i < HalfDimensions; ++i)
                    (next->*accPtr).accumulation[Perspective][i] -= weights[offset + i];

                for (std::size_t i = 0; i < PSQTBuckets; ++i)
                    (next->*accPtr).psqtAccumulation[Perspective][i] -=
                      psqtWeights[index * PSQTBuckets + i];
            }

            // Difference calculation for the activated features
            for (const auto index : added)
            {
                const IndexType offset = HalfDimensions * index;
                for (IndexType i = 0; i < HalfDimensions; ++i)
                    (next->*accPtr).accumulation[Perspective][i] += weights[offset + i];

                for (std::size_t i = 0; i < PSQTBuckets; ++i)
                    (next->*accPtr).psqtAccumulation[Perspective][i] +=
                      psqtWeights[index * PSQTBuckets + i];
            }
#endif
        }

        (next->*accPtr).computed[Perspective] = true;

        if (next != target_state)
            update_accumulator_incremental<Perspective, Direction>(ksq, target_state, next);
    }


    template<Color Perspective>
    void update_accumulator_refresh_cache(const Position&                           pos,
                                          AccumulatorCaches::Cache<HalfDimensions>* cache) const {
        assert(cache != nullptr);

        Square                ksq   = pos.square<KING>(Perspective);
        auto&                 entry = (*cache)[ksq][Perspective];
        FeatureSet::IndexList removed, added;

        for (Color c : {WHITE, BLACK})
        {
            for (PieceType pt = PAWN; pt <= KING; ++pt)
            {
                const Piece    piece    = make_piece(c, pt);
                const Bitboard oldBB    = entry.byColorBB[c] & entry.byTypeBB[pt];
                const Bitboard newBB    = pos.pieces(c, pt);
                Bitboard       toRemove = oldBB & ~newBB;
                Bitboard       toAdd    = newBB & ~oldBB;

                while (toRemove)
                {
                    Square sq = pop_lsb(toRemove);
                    removed.push_back(FeatureSet::make_index<Perspective>(sq, piece, ksq));
                }
                while (toAdd)
                {
                    Square sq = pop_lsb(toAdd);
                    added.push_back(FeatureSet::make_index<Perspective>(sq, piece, ksq));
                }
            }
        }

        auto& accumulator                 = pos.state()->*accPtr;
        accumulator.computed[Perspective] = true;

#ifdef VECTOR
        const bool combineLast3 = std::abs((int) removed.size() - (int) added.size()) == 1
                               && removed.size() + added.size() > 2;
        vec_t      acc[Tiling::NumRegs];
        psqt_vec_t psqt[Tiling::NumPsqtRegs];

        for (IndexType j = 0; j < HalfDimensions / Tiling::TileHeight; ++j)
        {
            auto* accTile = reinterpret_cast<vec_t*>(
              &accumulator.accumulation[Perspective][j * Tiling::TileHeight]);
            auto* entryTile = reinterpret_cast<vec_t*>(&entry.accumulation[j * Tiling::TileHeight]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = entryTile[k];

            std::size_t i = 0;
            for (; i < std::min(removed.size(), added.size()) - combineLast3; ++i)
            {
                IndexType       indexR  = removed[i];
                const IndexType offsetR = HalfDimensions * indexR + j * Tiling::TileHeight;
                auto*           columnR = reinterpret_cast<const vec_t*>(&weights[offsetR]);
                IndexType       indexA  = added[i];
                const IndexType offsetA = HalfDimensions * indexA + j * Tiling::TileHeight;
                auto*           columnA = reinterpret_cast<const vec_t*>(&weights[offsetA]);

                for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                    acc[k] = vec_add_16(acc[k], vec_sub_16(columnA[k], columnR[k]));
            }
            if (combineLast3)
            {
                IndexType       indexR  = removed[i];
                const IndexType offsetR = HalfDimensions * indexR + j * Tiling::TileHeight;
                auto*           columnR = reinterpret_cast<const vec_t*>(&weights[offsetR]);
                IndexType       indexA  = added[i];
                const IndexType offsetA = HalfDimensions * indexA + j * Tiling::TileHeight;
                auto*           columnA = reinterpret_cast<const vec_t*>(&weights[offsetA]);

                if (removed.size() > added.size())
                {
                    IndexType       indexR2  = removed[i + 1];
                    const IndexType offsetR2 = HalfDimensions * indexR2 + j * Tiling::TileHeight;
                    auto*           columnR2 = reinterpret_cast<const vec_t*>(&weights[offsetR2]);

                    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                        acc[k] = vec_sub_16(vec_add_16(acc[k], columnA[k]),
                                            vec_add_16(columnR[k], columnR2[k]));
                }
                else
                {
                    IndexType       indexA2  = added[i + 1];
                    const IndexType offsetA2 = HalfDimensions * indexA2 + j * Tiling::TileHeight;
                    auto*           columnA2 = reinterpret_cast<const vec_t*>(&weights[offsetA2]);

                    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                        acc[k] = vec_add_16(vec_sub_16(acc[k], columnR[k]),
                                            vec_add_16(columnA[k], columnA2[k]));
                }
            }
            else
            {
                for (; i < removed.size(); ++i)
                {
                    IndexType       index  = removed[i];
                    const IndexType offset = HalfDimensions * index + j * Tiling::TileHeight;
                    auto*           column = reinterpret_cast<const vec_t*>(&weights[offset]);

                    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                        acc[k] = vec_sub_16(acc[k], column[k]);
                }
                for (; i < added.size(); ++i)
                {
                    IndexType       index  = added[i];
                    const IndexType offset = HalfDimensions * index + j * Tiling::TileHeight;
                    auto*           column = reinterpret_cast<const vec_t*>(&weights[offset]);

                    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                        acc[k] = vec_add_16(acc[k], column[k]);
                }
            }

            for (IndexType k = 0; k < Tiling::NumRegs; k++)
                vec_store(&entryTile[k], acc[k]);
            for (IndexType k = 0; k < Tiling::NumRegs; k++)
                vec_store(&accTile[k], acc[k]);
        }

        for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
        {
            auto* accTilePsqt = reinterpret_cast<psqt_vec_t*>(
              &accumulator.psqtAccumulation[Perspective][j * Tiling::PsqtTileHeight]);
            auto* entryTilePsqt =
              reinterpret_cast<psqt_vec_t*>(&entry.psqtAccumulation[j * Tiling::PsqtTileHeight]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = entryTilePsqt[k];

            for (std::size_t i = 0; i < removed.size(); ++i)
            {
                IndexType       index  = removed[i];
                const IndexType offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
                auto* columnPsqt       = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

                for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                    psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
            }
            for (std::size_t i = 0; i < added.size(); ++i)
            {
                IndexType       index  = added[i];
                const IndexType offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
                auto* columnPsqt       = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

                for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                    psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
            }

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                vec_store_psqt(&entryTilePsqt[k], psqt[k]);
            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                vec_store_psqt(&accTilePsqt[k], psqt[k]);
        }

#else

        for (const auto index : removed)
        {
            const IndexType offset = HalfDimensions * index;
            for (IndexType j = 0; j < HalfDimensions; ++j)
                entry.accumulation[j] -= weights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                entry.psqtAccumulation[k] -= psqtWeights[index * PSQTBuckets + k];
        }
        for (const auto index : added)
        {
            const IndexType offset = HalfDimensions * index;
            for (IndexType j = 0; j < HalfDimensions; ++j)
                entry.accumulation[j] += weights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                entry.psqtAccumulation[k] += psqtWeights[index * PSQTBuckets + k];
        }

        // The accumulator of the refresh entry has been updated.
        // Now copy its content to the actual accumulator we were refreshing.

        std::memcpy(accumulator.accumulation[Perspective], entry.accumulation,
                    sizeof(BiasType) * HalfDimensions);

        std::memcpy(accumulator.psqtAccumulation[Perspective], entry.psqtAccumulation,
                    sizeof(int32_t) * PSQTBuckets);
#endif

        for (Color c : {WHITE, BLACK})
            entry.byColorBB[c] = pos.pieces(c);

        for (PieceType pt = PAWN; pt <= KING; ++pt)
            entry.byTypeBB[pt] = pos.pieces(pt);
    }


    template<Color Perspective>
    void update_accumulator(const Position&                           pos,
                            AccumulatorCaches::Cache<HalfDimensions>* cache) const {
        StateInfo* st = pos.state();
        if ((st->*accPtr).computed[Perspective])
            return;  // nothing to do

        // Look for a usable already computed accumulator of an earlier position.
        // Always try to do an incremental update as most accumulators will be reusable.
        do
        {
            if (FeatureSet::requires_refresh(st, Perspective) || !st->previous
                || st->previous->next != st)
            {
                // compute accumulator from scratch for this position
                update_accumulator_refresh_cache<Perspective>(pos, cache);
                if (st != pos.state())
                    // when computing an accumulator from scratch we can use it to
                    // efficiently compute the accumulator backwards, until we get to a king
                    // move. We expect that we will need these accumulators later anyway, so
                    // computing them now will save some work.
                    update_accumulator_incremental<Perspective, BACKWARDS>(
                      pos.square<KING>(Perspective), st, pos.state());
                return;
            }
            st = st->previous;
        } while (!(st->*accPtr).computed[Perspective]);

        // Start from the oldest computed accumulator, update all the
        // accumulators up to the current position.
        update_accumulator_incremental<Perspective>(pos.square<KING>(Perspective), pos.state(), st);
    }

    template<IndexType Size>
    friend struct AccumulatorCaches::Cache;

    alignas(CacheLineSize) BiasType biases[HalfDimensions];
    alignas(CacheLineSize) WeightType weights[HalfDimensions * InputDimensions];
};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
