#ifndef GYRO_FN_DRIVERS_HPP_
#define GYRO_FN_DRIVERS_HPP_

#include <chrono>
#include <memory>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>

namespace gufo::gyro {

#if defined(ENGINE_ENABLE_HIP)
/// The `bench`, `prompt` and `chat` drivers every Flash-Next provider wires
/// up. They select the runner through the same engine, so a provider only
/// differs in its probe and its weight formats.
int RunQwen38FlashNextBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

int RunFlashNextPrompt(const ext::TextPromptOptions& opt,
                       const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point model_load_start);

int RunFlashNextChat(const ext::TextPromptOptions& opt,
                     const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::gyro

#endif  // GYRO_FN_DRIVERS_HPP_
