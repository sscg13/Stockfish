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
constexpr IndexType TransformedFeatureDimensionsBig = 1024;
constexpr IndexType LayerStacks = 1;

template<IndexType L1>
struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    Layers::SCReLUAffine<TransformedFeatureDimensions * 2> output;

    std::int32_t evaluate(TransformedFeatureType* transformedFeatures, const int bucket) {
      return output.evaluate(transformedFeatures, bucket);
    }
    // Read network parameters
    bool read_parameters(std::istream& stream) {
        return output.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return output.write_parameters(stream);
    }
};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
