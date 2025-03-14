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

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_SCRELU_AFFINE_H_INCLUDED
#define NNUE_LAYERS_SCRELU_AFFINE_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE::Layers {

template<IndexType InDims>
class SCReLUAffine {
   public:
    // Input/output type
    using InputType  = std::int16_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputBuckets = 8;

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<std::int16_t>(stream, weights, OutputBuckets * InputDimensions);
        read_little_endian<std::int16_t>(stream, biases, OutputBuckets);
        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<std::int16_t>(stream, weights, OutputBuckets * InputDimensions);
        write_little_endian<std::int16_t>(stream, biases, OutputBuckets);
        return !stream.fail();
    }

    // Forward propagation
    OutputType evaluate(const InputType* input, IndexType bucket) {
        assert(bucket < OutputBuckets);
        const IndexType weightOffset = bucket * InputDimensions;
        const auto* weights_ptr = &(weights[weightOffset]);
        OutputType output = 255 * static_cast<OutputType>(biases[bucket]);
        for (IndexType i = 0; i < InputDimensions; i++) {
            const InputType clipped = std::clamp(input[i], static_cast<InputType>(0), static_cast<InputType>(255));
            output += clipped * clipped * weights_ptr[i];
        }
        return output / 255;
    }

    alignas(CacheLineSize) std::int16_t intermediate[InputDimensions];
    alignas(CacheLineSize) std::int16_t biases[OutputBuckets];
    alignas(CacheLineSize) std::int16_t weights[OutputBuckets * InputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED