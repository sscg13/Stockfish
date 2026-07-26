/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Definition of the Quantmoid4 NNUE activation function

#ifndef NNUE_LAYERS_QUANTMOID4_H_INCLUDED
#define NNUE_LAYERS_QUANTMOID4_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

template<IndexType InDims, int WeightScaleBitsLocal = WeightScaleBits>
class Quantmoid4 {
   public:
    using InputType  = i32;
    using OutputType = u8;

    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = 0x7B3C12D8u;
        hashValue += prevHash;
        return hashValue;
    }

    bool read_parameters(std::istream&) { return true; }

    bool write_parameters(std::ostream&) const { return true; }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_hash_value(0));
        return h;
    }

    void propagate(const InputType* input, OutputType* output) const {
        for (IndexType i = 0; i < InputDimensions; ++i)
        {
            const int quantizedInput = std::clamp(input[i] >> WeightScaleBitsLocal, -127, 127);
            const int distance       = 127 - std::abs(quantizedInput);
            const int lowerHalf      = (distance * distance) >> 8;
            output[i] = static_cast<OutputType>(quantizedInput < 0 ? lowerHalf : 126 - lowerHalf);
        }
    }
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_QUANTMOID4_H_INCLUDED
