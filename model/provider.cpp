#include <gufo/ext/generated/flash_next_gyro_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "model/drivers.hpp"
#include "model/probe.hpp"
#include "model/runner.hpp"

void flash_next_gyro_q4_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_flash_next_gyro::
                           kProvider_flash_next_q4);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flash-next-q4",
      .probe = &gufo::gyro::ProbeFlashNext,
      .probe_gguf = &gufo::gyro::ProbeFlashNextGguf,
      .create = &gufo::gyro::CreateQwenFlashNextTextRunner,
      .bench_driver =
          &gufo::gyro::RunQwen38FlashNextBenchmark,
      .prompt_driver = &gufo::gyro::RunFlashNextPrompt,
      .chat_driver = &gufo::gyro::RunFlashNextChat,
      .sampling_defaults =
          &gufo::gyro::FlashNextSamplingDefaults,
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1}},
  });
#endif
}
