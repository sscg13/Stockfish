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

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <cstdint>
#include <cstring>
#include <iosfwd>

#include "features/half_ka_v2_hm.h"
#include "features/simplified_threats.h"
#include "layers/screlu_affine.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using FeatureSet = Features::Simplified_Threats;

// Number of input feature dimensions after conversion
constexpr IndexType TransformedFeatureDimensionsBig = 256;

constexpr IndexType LayerStacks = 1;

template<IndexType L1>
struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    Layers::SCReLUAffine<TransformedFeatureDimensions * 2> fc_0;

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        return fc_0.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return fc_0.write_parameters(stream);
    }

    std::int32_t propagate(const TransformedFeatureType* transformedFeatures) {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;

            Buffer() { std::memset(this, 0, sizeof(*this)); }
        };

#if defined(__clang__) && (__APPLE__)
        // workaround for a bug reported with xcode 12
        static thread_local auto tlsBuffer = std::make_unique<Buffer>();
        // Access TLS only once, cache result.
        Buffer& buffer = *tlsBuffer;
#else
        alignas(CacheLineSize) static thread_local Buffer buffer;
#endif
        fc_0.propagate(transformedFeatures, buffer.fc_0_out);
        std::int32_t outputValue = buffer.fc_0_out[0];

        return outputValue;
    }
};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
