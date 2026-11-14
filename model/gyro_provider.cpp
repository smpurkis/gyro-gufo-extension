#include <gufo/ext/generated/flash_next_gyro_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "model/drivers.hpp"
#include "model/probe.hpp"
#include "model/runner.hpp"

void flash_next_gyro_gyro_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_flash_next_gyro::
          kProvider_flash_next_gyro_m);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flash-next-gyro-m",
      .probe = &gufo::gyro::ProbeFlashNextGyro,
      .probe_gguf = &gufo::gyro::ProbeFlashNextGyroGguf,
      .create = &gufo::gyro::CreateQwenFlashNextTextRunner,
      .bench_driver =
          &gufo::gyro::RunQwen38FlashNextBenchmark,
      .prompt_driver = &gufo::gyro::RunFlashNextPrompt,
      .chat_driver = &gufo::gyro::RunFlashNextChat,
      .sampling_defaults =
          &gufo::gyro::FlashNextSamplingDefaults,
      // TQ2_T expert rows are this extension's storage layout: core maps them
      // without sizing them, and the trellis kernels read them directly.
      .accepts_unsized_tensors = true,
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1}},
  });
#endif
}
