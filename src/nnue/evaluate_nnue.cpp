/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

// Code for calculating NNUE evaluation function

#include "evaluate_nnue.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "../uci.h"
#include "nnue_accumulator.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

// Input feature converter
LargePagePtr<FeatureTransformer<TransformedFeatureDimensionsBig, &StateInfo::accumulatorBig>>
  featureTransformerBig;
LargePagePtr<FeatureTransformer<TransformedFeatureDimensionsSmall, &StateInfo::accumulatorSmall>>
  featureTransformerSmall;

// Evaluation function
AlignedPtr<Network<TransformedFeatureDimensionsBig, L2Big, L3Big>>       networkBig[LayerStacks];
AlignedPtr<Network<TransformedFeatureDimensionsSmall, L2Small, L3Small>> networkSmall[LayerStacks];

// Evaluation function file names

namespace Detail {

// Initialize the evaluation function parameters
template<typename T>
void initialize(AlignedPtr<T>& pointer) {

    pointer.reset(reinterpret_cast<T*>(std_aligned_alloc(alignof(T), sizeof(T))));
    std::memset(pointer.get(), 0, sizeof(T));
}

template<typename T>
void initialize(LargePagePtr<T>& pointer) {

    static_assert(alignof(T) <= 4096,
                  "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");
    pointer.reset(reinterpret_cast<T*>(aligned_large_pages_alloc(sizeof(T))));
    std::memset(pointer.get(), 0, sizeof(T));
}

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    std::uint32_t header;
    header = read_little_endian<std::uint32_t>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<std::uint32_t>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail


// Initialize the evaluation function parameters
static void initialize(NetSize netSize) {

    if (netSize == Small)
    {
        Detail::initialize(featureTransformerSmall);
        for (std::size_t i = 0; i < LayerStacks; ++i)
            Detail::initialize(networkSmall[i]);
    }
    else
    {
        Detail::initialize(featureTransformerBig);
        for (std::size_t i = 0; i < LayerStacks; ++i)
            Detail::initialize(networkBig[i]);
    }
}

// Read network header
static bool read_header(std::istream& stream, std::uint32_t* hashValue, std::string* desc) {
    std::uint32_t version, size;

    version    = read_little_endian<std::uint32_t>(stream);
    *hashValue = read_little_endian<std::uint32_t>(stream);
    size       = read_little_endian<std::uint32_t>(stream);
    if (!stream || version != Version)
        return false;
    desc->resize(size);
    stream.read(&(*desc)[0], size);
    return !stream.fail();
}

// Write network header
static bool write_header(std::ostream& stream, std::uint32_t hashValue, const std::string& desc) {
    write_little_endian<std::uint32_t>(stream, Version);
    write_little_endian<std::uint32_t>(stream, hashValue);
    write_little_endian<std::uint32_t>(stream, std::uint32_t(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}

// Read network parameters
static bool read_parameters(std::istream& stream, NetSize netSize, std::string& netDescription) {

    std::uint32_t hashValue;
    if (!read_header(stream, &hashValue, &netDescription))
        return false;
    if (hashValue != HashValue[netSize])
        return false;
    if (netSize == Big && !Detail::read_parameters(stream, *featureTransformerBig))
        return false;
    if (netSize == Small && !Detail::read_parameters(stream, *featureTransformerSmall))
        return false;
    for (std::size_t i = 0; i < LayerStacks; ++i)
    {
        if (netSize == Big && !Detail::read_parameters(stream, *(networkBig[i])))
            return false;
        if (netSize == Small && !Detail::read_parameters(stream, *(networkSmall[i])))
            return false;
    }
    return stream && stream.peek() == std::ios::traits_type::eof();
}

// Write network parameters
static bool
write_parameters(std::ostream& stream, NetSize netSize, const std::string& netDescription) {

    if (!write_header(stream, HashValue[netSize], netDescription))
        return false;
    if (netSize == Big && !Detail::write_parameters(stream, *featureTransformerBig))
        return false;
    if (netSize == Small && !Detail::write_parameters(stream, *featureTransformerSmall))
        return false;
    for (std::size_t i = 0; i < LayerStacks; ++i)
    {
        if (netSize == Big && !Detail::write_parameters(stream, *(networkBig[i])))
            return false;
        if (netSize == Small && !Detail::write_parameters(stream, *(networkSmall[i])))
            return false;
    }
    return bool(stream);
}

void hint_common_parent_position(const Position& pos) {

    int simpleEvalAbs = std::abs(simple_eval(pos, pos.side_to_move()));
    if (simpleEvalAbs > 1050)
        featureTransformerSmall->hint_common_access(pos, simpleEvalAbs > 2500);
    else
        featureTransformerBig->hint_common_access(pos, false);
}

// Evaluation function. Perform differential calculation.
template<NetSize Net_Size>
Value evaluate(const Position& pos, bool adjusted, int* complexity, bool psqtOnly) {

    // We manually align the arrays on the stack because with gcc < 9.3
    // overaligning stack variables with alignas() doesn't work correctly.

    constexpr uint64_t alignment = CacheLineSize;
    constexpr int      delta     = 24;

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
    TransformedFeatureType transformedFeaturesUnaligned
      [FeatureTransformer < Net_Size == Small ? TransformedFeatureDimensionsSmall
                                              : TransformedFeatureDimensionsBig,
       nullptr > ::BufferSize + alignment / sizeof(TransformedFeatureType)];

    auto* transformedFeatures = align_ptr_up<alignment>(&transformedFeaturesUnaligned[0]);
#else

    alignas(alignment) TransformedFeatureType
      transformedFeatures[FeatureTransformer < Net_Size == Small ? TransformedFeatureDimensionsSmall
                                                                 : TransformedFeatureDimensionsBig,
                          nullptr > ::BufferSize];
#endif

    ASSERT_ALIGNED(transformedFeatures, alignment);

    const int  bucket = (pos.count<ALL_PIECES>() - 1) / 4;
    const auto psqt =
      Net_Size == Small
        ? featureTransformerSmall->transform(pos, transformedFeatures, bucket, psqtOnly)
        : featureTransformerBig->transform(pos, transformedFeatures, bucket, psqtOnly);

    const auto positional =
      !psqtOnly ? (Net_Size == Small ? networkSmall[bucket]->propagate(transformedFeatures)
                                     : networkBig[bucket]->propagate(transformedFeatures))
                : 0;

    if (complexity)
        *complexity = !psqtOnly ? std::abs(psqt - positional) / OutputScale : 0;

    // Give more value to positional evaluation when adjusted flag is set
    if (adjusted)
        return static_cast<Value>(((1024 - delta) * psqt + (1024 + delta) * positional)
                                  / (1024 * OutputScale));
    else
        return static_cast<Value>((psqt + positional) / OutputScale);
}

template Value evaluate<Big>(const Position& pos, bool adjusted, int* complexity, bool psqtOnly);
template Value evaluate<Small>(const Position& pos, bool adjusted, int* complexity, bool psqtOnly);

struct NnueEvalTrace {
    static_assert(LayerStacks == PSQTBuckets);

    Value       psqt[LayerStacks];
    Value       positional[LayerStacks];
    std::size_t correctBucket;
};

constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");

// Load eval, from a file stream or a memory stream
std::optional<std::string> load_eval(std::istream& stream, NetSize netSize) {

    initialize(netSize);
    std::string netDescription;
    return read_parameters(stream, netSize, netDescription) ? std::make_optional(netDescription)
                                                            : std::nullopt;
}

// Save eval, to a file stream or a memory stream
bool save_eval(std::ostream&      stream,
               NetSize            netSize,
               const std::string& name,
               const std::string& netDescription) {

    if (name.empty() || name == "None")
        return false;

    return write_parameters(stream, netSize, netDescription);
}

// Save eval, to a file given by its name
bool save_eval(const std::optional<std::string>&                              filename,
               NetSize                                                        netSize,
               const std::unordered_map<Eval::NNUE::NetSize, Eval::EvalFile>& evalFiles) {

    std::string actualFilename;
    std::string msg;

    if (filename.has_value())
        actualFilename = filename.value();
    else
    {
        if (evalFiles.at(netSize).current
            != (netSize == Small ? EvalFileDefaultNameSmall : EvalFileDefaultNameBig))
        {
            msg = "Failed to export a net. "
                  "A non-embedded net can only be saved if the filename is specified";

            sync_cout << msg << sync_endl;
            return false;
        }
        actualFilename = (netSize == Small ? EvalFileDefaultNameSmall : EvalFileDefaultNameBig);
    }

    std::ofstream stream(actualFilename, std::ios_base::binary);
    bool          saved = save_eval(stream, netSize, evalFiles.at(netSize).current,
                                    evalFiles.at(netSize).netDescription);

    msg = saved ? "Network saved successfully to " + actualFilename : "Failed to export a net";

    sync_cout << msg << sync_endl;
    return saved;
}


}  // namespace Stockfish::Eval::NNUE
