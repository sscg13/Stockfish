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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <iterator>

#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE {

#if defined(VECTOR)

// Computes the feature transformer's pairwise activation for each 16-bit lane,
//
//     min(max(a, 0) * max(b, 0) >> FtProductShift, FtOutMaxVal)
//
// Neither operand is clamped from above, so the product does not fit in 16 bits
// and a plain 16-bit multiply would overflow. Computing it in 32 bits is possible
// but costs a widening and a narrowing step per vector, roughly tripling the cost
// of the loop on x86.
//
// Instead, note that the result saturates at FtOutMaxVal and
// FtOutMaxVal << FtProductShift < 65536: any product that does not fit in the low
// 16 bits is necessarily clamped. So mullo carries the exact product whenever the
// result is not clamped, and a non-zero mulhi is exactly the signal that it is.
// That turns the saturating 16-bit multiply we would like to have into a multiply
// pair plus a select, with the shift applied to the low half only.
//
// Platforms with a native widening multiply (NEON, wasm) do not need the trick and
// use a saturating narrowing shift instead.
[[maybe_unused]] static inline SIMD::vec_t ft_product_16(SIMD::vec_t a, SIMD::vec_t b) {
    using namespace SIMD;

    #if defined(USE_NEON)

    const int16x8_t q0 = vmaxq_s16(a, vdupq_n_s16(0));
    const int16x8_t q1 = vmaxq_s16(b, vdupq_n_s16(0));

    const int32x4_t plo = vmull_s16(vget_low_s16(q0), vget_low_s16(q1));
    const int32x4_t phi = vmull_s16(vget_high_s16(q0), vget_high_s16(q1));

    const uint16x8_t r = vcombine_u16(vqshrun_n_s32(plo, FtProductShift),
                                      vqshrun_n_s32(phi, FtProductShift));

    return vreinterpretq_s16_u16(vminq_u16(r, vdupq_n_u16(FtOutMaxVal)));

    #elif defined(__wasm__)

    const v128_t q0 = wasm_i16x8_max(a, wasm_i16x8_splat(0));
    const v128_t q1 = wasm_i16x8_max(b, wasm_i16x8_splat(0));

    const v128_t plo = wasm_i32x4_shr(wasm_i32x4_extmul_low_i16x8(q0, q1), FtProductShift);
    const v128_t phi = wasm_i32x4_shr(wasm_i32x4_extmul_high_i16x8(q0, q1), FtProductShift);

    return wasm_u16x8_min(wasm_u16x8_narrow_i32x4(plo, phi), wasm_u16x8_splat(FtOutMaxVal));

    #else

    const vec_t Zero = vec_zero();
    const vec_t Max  = vec_set_16(FtOutMaxVal);

    const vec_t q0 = vec_max_16(a, Zero);
    const vec_t q1 = vec_max_16(b, Zero);

    const vec_t lo = vec_mullo_16(q0, q1);
    const vec_t hi = vec_mulhi_u16(q0, q1);
    const vec_t v  = vec_srli_16(lo, FtProductShift);

        // hi != 0 means the product overflowed 16 bits, which means it is clamped
        #if defined(USE_AVX512)
    return _mm512_mask_blend_epi16(_mm512_test_epi16_mask(hi, hi), v, Max);
        #elif defined(USE_AVX2)
    return _mm256_blendv_epi8(v, Max, _mm256_cmpgt_epi16(hi, Zero));
        #elif defined(USE_SSE41)
    return _mm_blendv_epi8(v, Max, _mm_cmpgt_epi16(hi, Zero));
        #elif defined(USE_SSE2)
    const __m128i mask = _mm_cmpgt_epi16(hi, Zero);
    return _mm_or_si128(_mm_and_si128(mask, Max), _mm_andnot_si128(mask, v));
        #elif defined(USE_LASX)
    return __lasx_xvbitsel_v(Max, v, __lasx_xvseq_h(hi, Zero));
        #elif defined(USE_LSX)
    return __lsx_vbitsel_v(Max, v, __lsx_vseq_h(hi, Zero));
        #endif

    #endif
}

#endif  // defined(VECTOR)

// Returns the inverse of a permutation
template<usize Len>
constexpr std::array<usize, Len> invert_permutation(const std::array<usize, Len>& order) {
    std::array<usize, Len> inverse{};
    for (usize i = 0; i < order.size(); i++)
        inverse[order[i]] = i;
    return inverse;
}

// Divide a byte region of size TotalSize to chunks of size
// BlockSize, and permute the blocks by a given order
template<usize BlockSize, typename T, usize N, usize OrderSize>
void permute(std::array<T, N>& data, const std::array<usize, OrderSize>& order) {
    constexpr usize TotalSize = N * sizeof(T);

    static_assert(TotalSize % (BlockSize * OrderSize) == 0,
                  "ChunkSize * OrderSize must perfectly divide TotalSize");

    constexpr usize ProcessChunkSize = BlockSize * OrderSize;

    std::array<std::byte, ProcessChunkSize> buffer{};

    std::byte* const bytes = reinterpret_cast<std::byte*>(data.data());

    for (usize i = 0; i < TotalSize; i += ProcessChunkSize)
    {
        std::byte* const values = &bytes[i];

        for (usize j = 0; j < OrderSize; j++)
        {
            auto* const buffer_chunk = &buffer[j * BlockSize];
            auto* const value_chunk  = &values[order[j] * BlockSize];

            std::copy(value_chunk, value_chunk + BlockSize, buffer_chunk);
        }

        std::copy(std::begin(buffer), std::end(buffer), values);
    }
}

// Input feature converter
class FeatureTransformer {
    // Number of output dimensions for one side
    static constexpr IndexType HalfDimensions = L1;

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType ThreatInputDimensions = ThreatFeatureSet::Dimensions;
    static constexpr IndexType PairInputDimensions   = PairFeatureSet::Dimensions;
    static constexpr IndexType PsqDimensions         = PSQFeatureSet::Dimensions;
    static constexpr IndexType ThreatAndPpDimensions = ThreatInputDimensions + PairInputDimensions;
    static constexpr IndexType InputDimensions       = PsqDimensions + ThreatAndPpDimensions;
    static constexpr IndexType OutputDimensions      = HalfDimensions;
    static constexpr IndexType ThreatWeightSize      = ThreatInputDimensions * HalfDimensions;
    static constexpr IndexType ThreatPsqtWeightSize  = ThreatInputDimensions * PSQTBuckets;
    static constexpr IndexType PairWeightSize        = PairInputDimensions * HalfDimensions;
    static constexpr IndexType PairPsqtWeightSize    = PairInputDimensions * PSQTBuckets;
    static constexpr IndexType ThreatAndPpWeightSize = ThreatAndPpDimensions * HalfDimensions;
    static constexpr IndexType ThreatAndPpPsqtSize   = ThreatAndPpDimensions * PSQTBuckets;

    using BiasesArray            = std::array<BiasType, HalfDimensions>;
    using WeightArray            = std::array<WeightType, HalfDimensions * PsqDimensions>;
    using ThreatAndPpWeightArray = std::array<ThreatWeightType, ThreatAndPpWeightSize>;
    using PsqtWeightArray        = std::array<PSQTWeightType, PSQTBuckets * PsqDimensions>;
    using ThreatAndPpPsqtArray   = std::array<PSQTWeightType, ThreatAndPpPsqtSize>;

    // Size of forward propagation buffer
    static constexpr usize BufferSize = OutputDimensions * sizeof(OutputType);

    // Store the order by which 128-bit blocks of a 1024-bit data must
    // be permuted so that calling packus on adjacent vectors of 16-bit
    // integers loaded from the data results in the pre-permutation order
    static constexpr auto PackusEpi16Order = []() -> std::array<usize, 8> {
#if defined(USE_AVX512)
        // _mm512_packus_epi16 after permutation:
        // |   0   |   2   |   4   |   6   | // Vector 0
        // |   1   |   3   |   5   |   7   | // Vector 1
        // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 4, 6, 1, 3, 5, 7};
#elif defined(USE_AVX2) || defined(USE_LASX)
        // _mm256_packus_epi16 after permutation:
        // |   0   |   2   |  |   4   |   6   | // Vector 0, 2
        // |   1   |   3   |  |   5   |   7   | // Vector 1, 3
        // | 0 | 1 | 2 | 3 |  | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 1, 3, 4, 6, 5, 7};
#else
        return {0, 1, 2, 3, 4, 5, 6, 7};
#endif
    }();

    static constexpr auto InversePackusEpi16Order = invert_permutation(PackusEpi16Order);

    static constexpr u32 combine_hash(std::initializer_list<u32> hashes) {
        u32 hash = 0;
        for (const auto component_hash : hashes)
        {
            hash = (hash << 1) | (hash >> 31);
            hash ^= component_hash;
        }
        return hash;
    }

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        return combine_hash(
                 {ThreatFeatureSet::HashValue, PairFeatureSet::HashValue, PSQFeatureSet::HashValue})
             ^ (OutputDimensions * 2);
    }

    void permute_weights() {
        permute<16>(biases, PackusEpi16Order);
        permute<16>(weights, PackusEpi16Order);

        permute<8>(threatAndPpWeights, PackusEpi16Order);
    }

    void unpermute_weights() {
        permute<16>(biases, InversePackusEpi16Order);
        permute<16>(weights, InversePackusEpi16Order);
        permute<8>(threatAndPpWeights, InversePackusEpi16Order);
    }

    ThreatWeightType* threatWeightData() { return threatAndPpWeights.data(); }
    ThreatWeightType* pawnPairWeightData() { return threatWeightData() + ThreatWeightSize; }
    PSQTWeightType*   threatPsqtData() { return threatAndPpPsqtWeights.data(); }
    PSQTWeightType*   pawnPairPsqtData() { return threatPsqtData() + ThreatPsqtWeightSize; }


    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_leb_128(stream, biases);

        read_little_endian(stream, threatWeightData(), ThreatWeightSize);
        read_leb_128(stream, threatPsqtData(), ThreatPsqtWeightSize);
        read_little_endian(stream, pawnPairWeightData(), PairWeightSize);
        read_leb_128(stream, pawnPairPsqtData(), PairPsqtWeightSize);

        read_leb_128(stream, weights);
        read_leb_128(stream, psqtWeights);

        permute_weights();

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        std::unique_ptr<FeatureTransformer> copy = std::make_unique<FeatureTransformer>(*this);

        copy->unpermute_weights();

        write_leb_128<BiasType>(stream, copy->biases);


        write_little_endian(stream, copy->threatWeightData(), ThreatWeightSize);
        write_leb_128(stream, copy->threatPsqtData(), ThreatPsqtWeightSize);
        write_little_endian(stream, copy->pawnPairWeightData(), PairWeightSize);
        write_leb_128(stream, copy->pawnPairPsqtData(), PairPsqtWeightSize);

        write_leb_128<WeightType>(stream, copy->weights);
        write_leb_128<PSQTWeightType>(stream, copy->psqtWeights);

        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;

        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_raw_data_hash(psqtWeights));

        hash_combine(h, get_raw_data_hash(threatAndPpWeights));
        hash_combine(h, get_raw_data_hash(threatAndPpPsqtWeights));

        hash_combine(h, get_hash_value());

        return h;
    }

    // Convert input features
    i32 transform(const Position&                             pos,
                  AccumulatorStack&                           accumulatorStack,
                  AccumulatorCaches&                          cache,
                  OutputType*                                 output,
                  int                                         bucket,
                  [[maybe_unused]] NNZInfo<OutputDimensions>& nnzInfo) const {

        using namespace SIMD;
        accumulatorStack.evaluate(pos, *this, cache);
        const auto& accumulatorState = accumulatorStack.latest();

        const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& psqtAccumulation = accumulatorState.psqtAccumulation;
        const auto  psqt =
          (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket])
          / 2;

        const auto& accumulation = accumulatorState.accumulation;

        for (IndexType p = 0; p < 2; ++p)
        {
            const IndexType offset = (HalfDimensions / 2) * p;

#if defined(VECTOR)

            [[maybe_unused]] auto cursor = nnzInfo.make_cursor(p);

            constexpr IndexType OutputChunkSize = MaxChunkSize;
            static_assert((HalfDimensions / 2) % OutputChunkSize == 0);
            constexpr IndexType NumOutputChunks = HalfDimensions / 2 / OutputChunkSize;

            const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0]));
            const vec_t* in1 =
              reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][HalfDimensions / 2]));
            vec_t* out = reinterpret_cast<vec_t*>(output + offset);

            // Per the NNUE architecture, here we want the pairwise product of the
            // two ReLU'd accumulator halves, scaled back down to the int8 range
            // the following affine layer expects. See ft_product_16() above for
            // how the product is kept inside 16-bit lanes without clamping either
            // operand from above.

            for (IndexType j = 0; j < NumOutputChunks; j += 2)
            {
                vec_t packed[2];
                for (IndexType k = 0; k < 2; ++k)
                {
                    const IndexType i = (j + k) * 2;

                    const vec_t pa = ft_product_16(in0[i + 0], in1[i + 0]);
                    const vec_t pb = ft_product_16(in0[i + 1], in1[i + 1]);

                    packed[k] = out[j + k] = vec_packus_16(pa, pb);
                }

                cursor.record2(packed[0], packed[1]);
            }

#elif defined(USE_RVV)

            usize       j  = 0;
            usize       VL = __riscv_vsetvlmax_e8m1();
            vuint8m1_t  vid8;
            vuint16m2_t vid16;
            if (VL <= 256)
                vid8 = __riscv_vid_v_u8m1(VL);
            else
                vid16 = __riscv_vid_v_u16m2(VL);
            const auto& accp = accumulation[perspectives[p]];

            for (usize vl; j < HalfDimensions / 2; j += vl)
            {
                vl = __riscv_vsetvl_e16m2(HalfDimensions / 2 - j);

                vint16m2_t acc0 = __riscv_vle16_v_i16m2(&accp[j], vl);
                vint16m2_t acc1 = __riscv_vle16_v_i16m2(&accp[j + HalfDimensions / 2], vl);

                acc0 = __riscv_vmax(acc0, 0, vl);
                acc1 = __riscv_vmax(acc1, 0, vl);

                // Widening multiply, then a saturating narrowing shift by
                // FtProductShift; no need for the 16-bit overflow trick here.
                vuint32m4_t prod = __riscv_vwmulu(__riscv_vreinterpret_u16m2(acc0),
                                                  __riscv_vreinterpret_u16m2(acc1), vl);
                vuint16m2_t nar =
                  __riscv_vnclipu(prod, FtProductShift, __RISCV_VXRM_RDN, vl);
                nar = __riscv_vminu(nar, FtOutMaxVal, vl);

                vuint8m1_t result = __riscv_vnclipu(nar, 0, __RISCV_VXRM_RDN, vl);

                __riscv_vse8(&output[offset + j], result, vl);

                vbool8_t    m   = __riscv_vmsne(result, 0, vl);
                usize       cnt = __riscv_vcpop(m, vl);
                vuint16m2_t vidx;
                if (VL <= 256)
                    vidx = __riscv_vzext_vf2(__riscv_vcompress(vid8, m, vl), cnt);
                else
                    vidx = __riscv_vcompress(vid16, m, vl);
                __riscv_vse16(&nnzInfo.nnz[nnzInfo.count], __riscv_vadd(vidx, offset + j, cnt),
                              cnt);
                nnzInfo.count += cnt;
            }

#else

            for (IndexType j = 0; j < HalfDimensions / 2; ++j)
            {
                BiasType sum0 = accumulation[static_cast<int>(perspectives[p])][j + 0];
                BiasType sum1 =
                  accumulation[static_cast<int>(perspectives[p])][j + HalfDimensions / 2];

                sum0 = std::max<BiasType>(sum0, 0);
                sum1 = std::max<BiasType>(sum1, 0);

                output[offset + j] = static_cast<OutputType>(
                  std::min(unsigned(sum0 * sum1) >> FtProductShift, unsigned(FtOutMaxVal)));
            }

#endif
        }

        return psqt;
    }  // end of function transform()

    alignas(CacheLineSize) BiasesArray biases;
    alignas(CacheLineSize) WeightArray weights;

    // Threats and pawn-pair features are concatenated into one array to allow for a single index to address either.
    // The first pawn-pair feature is at index ThreatFeatureSet::Dimensions.
    static_assert(PairFeatureSet::IndexBase == ThreatFeatureSet::Dimensions);

    alignas(CacheLineSize) ThreatAndPpWeightArray threatAndPpWeights;
    alignas(CacheLineSize) PsqtWeightArray psqtWeights;
    alignas(CacheLineSize) ThreatAndPpPsqtArray threatAndPpPsqtWeights;
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::FeatureTransformer> {
    Stockfish::usize
    operator()(const Stockfish::Eval::NNUE::FeatureTransformer& ft) const noexcept {
        return ft.get_content_hash();
    }
};

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
