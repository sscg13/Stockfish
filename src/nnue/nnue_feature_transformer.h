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

enum IncUpdateDirection {
    FORWARD,
    BACKWARDS
};

// Input feature converter
template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
class FeatureTransformer {

   public:
    // Output type
    using OutputType = TransformedFeatureType;
    FeatureSet feature_indexer;
    int acc_updates = 0;
    int pos_loops = 0;
    int threat_loops = 0;
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
    

    template<Color Perspective>
    void update_accumulator_scratch(const Position& pos) {

        // The size must be enough to contain the largest possible update.
        // That might depend on the feature set and generally relies on the
        // feature set's update cost calculation to be correct and never allow
        // updates with more added/removed features than MaxActiveDimensions.
        // In this case, the maximum size of both feature addition and removal
        // is 2, since we are incrementally updating one move at a time.
        auto& accumulator = pos.state()->*accPtr;
        auto& psq = pos.state()->psq;
        auto& threats = pos.state()->threats;
        psq.clear();
        threats.clear();
        feature_indexer.append_active_features<Perspective>(pos, psq, threats);
        acc_updates++;
        pos_loops++;
        threat_loops += (int)psq.size();
        threat_loops += (int)threats.size();
        for (IndexType j = 0; j < TransformedFeatureDimensions; j++) {
            accumulator.accumulation[Perspective][j] = biases[j];
        }
        for (auto index : psq)
        {
            const IndexType offset = TransformedFeatureDimensions * index;
            assert(offset < TransformedFeatureDimensions * InputDimensions);
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++)
                accumulator.accumulation[Perspective][j] += weights[offset + j];
        }
        for (auto index : threats)
        {
            const IndexType offset = TransformedFeatureDimensions * index;
            assert(offset < TransformedFeatureDimensions * InputDimensions);
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++)
                accumulator.accumulation[Perspective][j] += weights[offset + j];
        }
        accumulator.computed[Perspective] = true;
    }

    template<Color Perspective, IncUpdateDirection Direction = FORWARD>
    void update_accumulator_incremental(const Square ksq, StateInfo* target_state, const StateInfo* computed) {
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
        auto& oldthreats = computed->threats;
        auto& newthreats = next->threats;
        newthreats.clear();
        //feature_indexer.append_active_psq<Perspective>(next->colorBB, next->pieceBB, next->board, newfeatures);
        feature_indexer.append_active_threats<Perspective>(next->colorBB, next->pieceBB, next->board, newthreats);
        pos_loops += 2;

        /*
        FeatureSet::IndexList testthreats;
        feature_indexer.append_active_threats<Perspective>(computed->colorBB, computed->pieceBB, computed->board, testthreats);
        for (std::size_t i = 0; i < testthreats.size(); i++) {
            assert(testthreats[i] == oldthreats[i]);
        }
        */

        if constexpr (Forward)
            feature_indexer.append_changed_indices<Perspective>(ksq, next->dirtyPiece, removed, added);
        else
            feature_indexer.append_changed_indices<Perspective>(ksq, computed->dirtyPiece, added, removed);
        
        write_difference(oldthreats, newthreats, removed, added);


        if (removed.size() == 0 && added.size() == 0)
        {
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++) {
                (next->*accPtr).accumulation[Perspective][j] = (computed->*accPtr).accumulation[Perspective][j];
            }
        }/*
        else if (newfeatures.size() <= removed.size() + added.size()) {
            acc_updates++;
            threat_loops += (int)newfeatures.size();
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++) {
                (next->*accPtr).accumulation[Perspective][j] = biases[j];
            }
            for (auto index : newfeatures)
            {
                const IndexType offset = TransformedFeatureDimensions * index;
                assert(offset < TransformedFeatureDimensions * InputDimensions);
                for (IndexType j = 0; j < TransformedFeatureDimensions; j++)
                    (next->*accPtr).accumulation[Perspective][j] += weights[offset + j];
            }
        }*/
        else
        {
            for (IndexType j = 0; j < TransformedFeatureDimensions; j++) {
                (next->*accPtr).accumulation[Perspective][j] = (computed->*accPtr).accumulation[Perspective][j];
            }
            acc_updates++;
            threat_loops += (int)removed.size();
            threat_loops += (int)added.size();
            // Difference calculation for the activated features
            for (auto index : added)
            {
                const IndexType offset = TransformedFeatureDimensions * index;
                for (IndexType i = 0; i < TransformedFeatureDimensions; ++i)
                    (next->*accPtr).accumulation[Perspective][i] += weights[offset + i];
            }
            // Difference calculation for the deactivated features
            for (auto index : removed)
            {
                const IndexType offset = TransformedFeatureDimensions * index;
                for (IndexType i = 0; i < TransformedFeatureDimensions; ++i)
                    (next->*accPtr).accumulation[Perspective][i] -= weights[offset + i];
            }
        }

        (next->*accPtr).computed[Perspective] = true;

        if (next != target_state)
            update_accumulator_incremental<Perspective, Direction>(ksq, target_state, next);
    }

    template<Color Perspective>
    void update_accumulator(const Position& pos) {
        StateInfo* st = pos.state();
        if ((st->*accPtr).computed[Perspective])
            return;  // nothing to do

        // Look for a usable already computed accumulator of an earlier position.
        // Always try to do an incremental update as most accumulators will be reusable.
        do
        {
            if (!st->previous || st->previous->next != st)
            {
                // compute accumulator from scratch for this position
                update_accumulator_scratch<Perspective>(pos);/*
                if (st != pos.state())
                    // when computing an accumulator from scratch we can use it to
                    // efficiently compute the accumulator backwards, until we get to a king
                    // move. We expect that we will need these accumulators later anyway, so
                    // computing them now will save some work.
                    update_accumulator_incremental<Perspective, BACKWARDS>(
                      pos.square<KING>(Perspective), st, pos.state());*/
                return;
            }
            st = st->previous;
        } while (!(st->*accPtr).computed[Perspective]);
        // Start from the oldest computed accumulator, update all the
        // accumulators up to the current position.
        update_accumulator_incremental<Perspective>(pos.square<KING>(Perspective), pos.state(), st);
    }

    void transform(const Position& pos, OutputType* output) {
        update_accumulator<WHITE>(pos);
        update_accumulator<BLACK>(pos);

        const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& accumulation = (pos.state()->*accPtr).accumulation;

        for (IndexType p = 0; p < 2; ++p)
        {
            const IndexType offset = TransformedFeatureDimensions * p;

            for (IndexType j = 0; j < TransformedFeatureDimensions; ++j)
            {
                output[offset + j] = accumulation[perspectives[p]][j];
            }
        }
    }  // end of function transform()

    alignas(CacheLineSize) BiasType biases[TransformedFeatureDimensions];
    alignas(CacheLineSize) WeightType weights[TransformedFeatureDimensions * InputDimensions];


};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
