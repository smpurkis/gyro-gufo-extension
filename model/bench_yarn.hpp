#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "gufo/ext/text.hpp"
#include "model/rope_scaling.hpp"

namespace gufo::gyro {

[[nodiscard]] std::size_t FlashNextBenchContext(
    const ext::TextBenchOptions& options);

[[nodiscard]] bool ResolveFlashNextBenchYarn(
    const ext::TextBenchOptions& options, std::uint32_t native,
    std::optional<RopeScaling>* scaling, std::string* error);

}  // namespace gufo::gyro

namespace gufo::cli {
using gufo::gyro::FlashNextBenchContext;
using gufo::gyro::ResolveFlashNextBenchYarn;
}  // namespace gufo::cli
