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
#include "nnue_misc.h"

namespace Stockfish::Eval::NNUE {

using BiasType       = std::int16_t;
using WeightType     = std::int16_t;

// Input feature converter
template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
class FeatureTransformer {

   public:
    // Output type
    using OutputType = TransformedFeatureType;
    FeatureSet threats;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = FeatureSet::Dimensions;
    static constexpr IndexType OutputDimensions = TransformedFeatureDimensions;

    // Size of forward propagation buffer
    static constexpr std::size_t BufferSize = OutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        return FeatureSet::HashValue ^ (OutputDimensions * 2);
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        std::cout << "FT::read_parameters()" << std::endl;
        read_little_endian<WeightType>(stream, weights, TransformedFeatureDimensions * InputDimensions);
        read_little_endian<BiasType>(stream, biases, TransformedFeatureDimensions);
        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) {
        write_little_endian<WeightType>(stream, weights, TransformedFeatureDimensions * InputDimensions);
        write_little_endian<BiasType>(stream, biases, TransformedFeatureDimensions);
        return !stream.fail();
    }
    //we start with non-ue
    void compute_accumulators_from_scratch(const Position& pos) {
        FeatureSet::IndexList white, black;
        threats.append_active_psq<WHITE>(pos, white);
        threats.append_active_threats<WHITE>(pos, white);
        threats.append_active_psq<BLACK>(pos, black);
        threats.append_active_threats<BLACK>(pos, black);
        auto& accumulation = (pos.state()->*accPtr).accumulation;
        for (IndexType j = 0; j < TransformedFeatureDimensions; j++) {
            accumulation[pos.side_to_move()][j] = biases[j];
        }
        for (const auto index : white)
        {
            const IndexType offset = TransformedFeatureDimensions * index;
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++)
                accumulation[pos.side_to_move()][j] += weights[offset + j];
        }
        for (IndexType j = 0; j < TransformedFeatureDimensions; j++) {
            accumulation[~pos.side_to_move()][j] = biases[j];
        }
        for (const auto index : black)
        {
            const IndexType offset = TransformedFeatureDimensions * index;
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++)
                accumulation[~pos.side_to_move()][j] += weights[offset + j];
        }
    }
    alignas(CacheLineSize) BiasType biases[TransformedFeatureDimensions];
    alignas(CacheLineSize) WeightType weights[TransformedFeatureDimensions * InputDimensions];


};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
