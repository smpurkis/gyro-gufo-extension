# Qwen3.8 Flash-Next Gyro-M: a community model extension for Gufo.
#
# This tree is Flash-Next's engine, ROCm kernels, Qwen tokenizer/template/vision
# layer and the Agention Gyro trellis (TQ2_T) expert codec. Everything lives in
# the extension's own namespace, so it installs beside the official
# flash-next-q4 extension rather than replacing it: the trellis probe accepts
# only artifacts whose routed experts are all TQ2_T rows.

gufo_extension_target(
  NAME gufo_flash_next_gyro
  SOURCES
    model/config.cpp
    model/continuation_layout.cpp
    model/ngram.cpp
    model/rope_scaling.cpp
    model/weights.cpp
    model/bench_yarn.cpp
    model/probe.cpp
    model/continuation_identity.cpp
    model/gyro/tq_codebook.c
    model/gyro/tq_ref.c
    model/extension.cpp
    family/config/qwen_config.cpp
    family/text/tokenizer.cpp
    family/text/chat_template.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    Threads::Threads
)

# Gufo's audited surface forwards to its own header paths; the copy the build
# stages under the binary directory is what resolves those forwards.
set(GFN_PUBLIC_INCLUDE "${CMAKE_BINARY_DIR}/src/ext/public-include")

# The scalar oracle the numeric tests compare kernels against.
add_library(gufo_flash_next_gyro_reference STATIC EXCLUDE_FROM_ALL
  "${GUFO_EXTENSION_BUILD_ROOT}/model/cpu_ops.cpp"
  "${GUFO_EXTENSION_BUILD_ROOT}/model/reference.cpp")
target_include_directories(gufo_flash_next_gyro_reference BEFORE PUBLIC
  "${GUFO_EXTENSION_BUILD_ROOT}")
target_include_directories(gufo_flash_next_gyro_reference SYSTEM PUBLIC
  "${GFN_PUBLIC_INCLUDE}")
target_link_libraries(gufo_flash_next_gyro_reference PUBLIC gufo_flash_next_gyro)
set_source_files_properties(
  "${GUFO_EXTENSION_BUILD_ROOT}/model/cpu_ops.cpp"
  "${GUFO_EXTENSION_BUILD_ROOT}/model/reference.cpp"
  PROPERTIES COMPILE_OPTIONS "-O2")

if(ENGINE_ENABLE_HIP)
  set(GFN_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm")
  set(GFN_MMQ_DIR "${GFN_ROCM_DIR}/mmq")
  set(GFN_MMQ_SOURCES
    "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/mmq/gfn_ggml_stubs.hip.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/mmq/gfn_mmq.hip.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/mmq/mmid.hip.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/mmq/mmvq.hip.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/mmq/moe_wave64.hip.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/mmq/quantize.hip.cpp"
  )

  # Vendored llama.cpp quantized-matmul tier: its own C++17 dialect and tile
  # size, so it stays a separate target like it is in Gufo's own tree.
  add_library(gufo_flash_next_gyro_mmq STATIC ${GFN_MMQ_SOURCES})
  set_source_files_properties(${GFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${GFN_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_flash_next_gyro_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_flash_next_gyro_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  # One flag per generator expression: the extension validator checks every
  # option independently, so payloads must not carry ';' separators.
  target_compile_options(gufo_flash_next_gyro_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_flash_next_gyro_mmq SYSTEM PUBLIC ${GFN_MMQ_DIR})
  # The public surface forwards to Gufo's own header paths, which the audited
  # copy under the build tree provides. This target keeps its own C++17 dialect,
  # so it takes that directory directly instead of linking the runtime target.
  target_include_directories(gufo_flash_next_gyro_mmq SYSTEM PRIVATE
    "${CMAKE_BINARY_DIR}/src/ext/public-include")
  target_link_libraries(gufo_flash_next_gyro_mmq PUBLIC hip::host roc::hipblas)

  gufo_extension_target(
    NAME gufo_flash_next_gyro_hip
    SOURCES
      model/continuation_adapter.cpp
      model/engine.cpp
      model/runner.cpp
      model/drivers.cpp
      model/gyro_provider.cpp
      model/kernels/rocm/blaslt.cpp
      model/kernels/rocm/device_model.cpp
      model/kernels/rocm/executor.cpp
      model/kernels/rocm/batch.cpp
      family/vision/prompt.cpp
      family/vision/image_cache.cpp
      vendored/logit_comparator.cpp
    LINK_LIBRARIES
      gufo_flash_next_gyro
      gufo_ext_cache
      gufo_ext_hip
      gufo_flash_next_gyro_mmq
      hip::host
      roc::hipblas
      roc::hipblaslt
      ${CMAKE_DL_LIBS}
  )

  if(TARGET gufo_flash_next_gyro_hip)
    gufo_extension_hip_sources(gufo_flash_next_gyro_hip
      SOURCES
        family/kernels/attention_batched.hip
        family/kernels/attention_decode.hip
        family/kernels/attention_decode_graph.hip
        family/kernels/attention_decode_verify_wave64.hip
        family/kernels/attention_tile.hip
        family/kernels/attention_wmma.hip
        family/kernels/dflash_kernels.hip
        family/kernels/embed.hip
        family/kernels/fused.hip
        family/kernels/gemv.hip
        family/kernels/gemv_quant.hip
        family/kernels/hipblas_gemm.hip
        family/kernels/hipblaslt_gemm.hip
        family/kernels/norm.hip
        family/kernels/prefill_embed.hip
        family/kernels/prefill_fp16.hip
        family/kernels/prefill_gemm.hip
        family/kernels/prefill_norm.hip
        family/kernels/prefill_quant_gemm.hip
        family/kernels/prefill_quant_wave64.hip
        family/kernels/prefill_residual.hip
        family/kernels/prefill_rope.hip
        family/kernels/prefill_ssm.hip
        family/kernels/prefill_swiglu.hip
        family/kernels/prefill_unpack.hip
        family/kernels/qkv.hip
        family/kernels/residual.hip
        family/kernels/rope.hip
        family/kernels/sample.hip
        family/kernels/small_batch_quant16_wave64.hip
        family/kernels/small_batch_wave64.hip
        family/kernels/ssm.hip
        family/kernels/ssm_decode_recurrence.hip
        family/kernels/ssm_recurrence.hip
        family/kernels/ssm_row_split.hip
        family/kernels/swiglu.hip
        family/kernels/unpack.hip
        family/kernels/verify_gemm_27b.hip
        model/kernels/rocm/kernels.hip.cpp
        model/kernels/rocm/draft_head.hip.cpp
        model/kernels/rocm/trellis.hip.cpp
        model/kernels/rocm/w8a8_wave64.hip
        family/vision/encoder.hip
        family/vision/device_input.hip
    )
    # Wave-64 units and the per-unit scheduler options match Gufo's own tree:
    # moving code between these flags changes rounding and register pressure.
    foreach(gfn_wave64
        "${GUFO_EXTENSION_BUILD_ROOT}/model/kernels/rocm/w8a8_wave64.hip"
        "${GUFO_EXTENSION_BUILD_ROOT}/family/kernels/small_batch_wave64.hip"
        "${GUFO_EXTENSION_BUILD_ROOT}/family/kernels/prefill_quant_wave64.hip"
        "${GUFO_EXTENSION_BUILD_ROOT}/family/kernels/attention_decode_verify_wave64.hip")
      set_property(SOURCE ${gfn_wave64} APPEND PROPERTY COMPILE_OPTIONS
        "-mwavefrontsize64")
    endforeach()
    set_property(
      SOURCE "${GUFO_EXTENSION_BUILD_ROOT}/family/kernels/small_batch_quant16_wave64.hip"
      APPEND PROPERTY COMPILE_OPTIONS
      "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
    set_property(
      SOURCE
        "${GUFO_EXTENSION_BUILD_ROOT}/family/kernels/prefill_quant_gemm.hip"
        "${GUFO_EXTENSION_BUILD_ROOT}/family/kernels/prefill_quant_wave64.hip"
      APPEND PROPERTY COMPILE_OPTIONS ${GUFO_CLANG_OPTION}-fno-slp-vectorize)
    # The draft head quantizer keeps ggml's IEEE single-precision rounding.
    set_property(SOURCE ${GFN_ROCM_DIR}/draft_head.hip.cpp
      APPEND PROPERTY COMPILE_OPTIONS
        "${GUFO_CLANG_OPTION}-fno-fast-math;${GUFO_CLANG_OPTION}-ffp-contract=off")
    target_compile_definitions(gufo_flash_next_gyro_hip PRIVATE QFN_ROCM_BUILD=1)
    target_compile_options(gufo_flash_next_gyro_hip PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O2>
      $<$<COMPILE_LANGUAGE:HIP>:-O3>
      $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-ffast-math>
      $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-fno-finite-math-only>
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
    target_include_directories(gufo_flash_next_gyro_hip SYSTEM PUBLIC
      "${GFN_PUBLIC_INCLUDE}"
      ${HIPCUB_INCLUDE_DIR}
      ${ROCPRIM_INCLUDE_DIR}
      ${ROCWMMA_INCLUDE_DIR})
  endif()

  # The vendored Gyro decoder keeps its notices with the installed license tree.
  install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/model/gyro/NOTICE"
    "${GUFO_EXTENSION_BUILD_ROOT}/licenses/gyro.txt"
    DESTINATION share/licenses/gufo/third-party)
endif()

# Tests. Host contracts and kernel checks need no model weights.
if(BUILD_TESTING AND NOT TARGET flash_next_gyro_tests)
  add_custom_target(flash_next_gyro_tests)
endif()

function(add_flash_next_gyro_test name library labels)
  set(target flash_next_gyro_${name}_test)
  add_executable(${target} "${GUFO_EXTENSION_BUILD_ROOT}/tests/${name}_test.cpp")
  target_include_directories(${target} BEFORE PRIVATE
    "${GUFO_EXTENSION_BUILD_ROOT}")
  target_include_directories(${target} SYSTEM PRIVATE "${GFN_PUBLIC_INCLUDE}")
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME flash_next_gyro.${name} COMMAND ${target})
  set_tests_properties(flash_next_gyro.${name} PROPERTIES
    LABELS "flash_next_gyro;${labels}")
  add_dependencies(flash_next_gyro_tests ${target})
endfunction()

# The serialization check compiles the identity translation unit itself, so it
# needs neither HIP nor the serving tier.
add_executable(flash_next_gyro_continuation_identity_test
  "${GUFO_EXTENSION_BUILD_ROOT}/tests/continuation_identity_test.cpp"
  "${GUFO_EXTENSION_BUILD_ROOT}/model/continuation_identity.cpp")
target_include_directories(flash_next_gyro_continuation_identity_test BEFORE PRIVATE
  "${GUFO_EXTENSION_BUILD_ROOT}")
target_include_directories(flash_next_gyro_continuation_identity_test SYSTEM PRIVATE "${GFN_PUBLIC_INCLUDE}")
target_compile_definitions(flash_next_gyro_continuation_identity_test PRIVATE
  GUFO_IDENTITY_FIXTURES="${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures/continuation_identity")
target_link_libraries(flash_next_gyro_continuation_identity_test PRIVATE
  gufo_ext_core)
add_test(NAME flash_next_gyro_continuation_identity_test
  COMMAND flash_next_gyro_continuation_identity_test)
set_tests_properties(flash_next_gyro_continuation_identity_test PROPERTIES
  LABELS "cpu;flash_next_gyro;cache;correctness")

add_flash_next_gyro_test(runner_options gufo_flash_next_gyro "cpu")
add_flash_next_gyro_test(probe gufo_flash_next_gyro "cpu")
add_flash_next_gyro_test(config gufo_flash_next_gyro "cpu")
add_flash_next_gyro_test(continuation_layout gufo_flash_next_gyro "cpu")
add_flash_next_gyro_test(ngram gufo_flash_next_gyro "cpu")
add_flash_next_gyro_test(rope_scaling gufo_flash_next_gyro "cpu")
add_flash_next_gyro_test(rope_scaling_oracle gufo_flash_next_gyro_reference "cpu")
add_flash_next_gyro_test(mtp_sampling gufo_ext_core "cpu")
add_flash_next_gyro_test(prompt_lookup gufo_ext_core "cpu")

# Provider probe and trunk binding against a real Gyro-M artifact. It reads only
# GGUF headers, so it needs no GPU: AGENTION_GYRO_MODEL names the file and the
# test is skipped without it.
add_executable(flash_next_gyro_gyro_artifact_test
  "${GUFO_EXTENSION_BUILD_ROOT}/tests/gyro_artifact_test.cpp")
target_include_directories(flash_next_gyro_gyro_artifact_test BEFORE PRIVATE
  "${GUFO_EXTENSION_BUILD_ROOT}")
target_include_directories(flash_next_gyro_gyro_artifact_test SYSTEM PRIVATE "${GFN_PUBLIC_INCLUDE}")
target_link_libraries(flash_next_gyro_gyro_artifact_test PRIVATE gufo_flash_next_gyro)
add_dependencies(flash_next_gyro_tests flash_next_gyro_gyro_artifact_test)
add_test(NAME flash_next_gyro.gyro_artifact
  COMMAND flash_next_gyro_gyro_artifact_test)
set_tests_properties(flash_next_gyro.gyro_artifact PROPERTIES
  LABELS "cpu;flash_next_gyro;external-model"
  SKIP_RETURN_CODE 77
  TIMEOUT 900)

if(ENGINE_ENABLE_HIP)
  add_library(flash_next_gyro_test_hip INTERFACE)
  target_link_libraries(flash_next_gyro_test_hip INTERFACE
    gufo_flash_next_gyro_hip gufo_flash_next_gyro_mmq hip::host)
  target_compile_definitions(flash_next_gyro_test_hip INTERFACE ENGINE_ENABLE_HIP=1)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops hc_route_ops gdn_ops
      attention_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      select_ops trellis_ops verification_sampling wide_verification)
  # snapshot_copy is gufo core's download path and lives in core's own test
  # tree; the extension does not own it.
    add_flash_next_gyro_test(${name} flash_next_gyro_test_hip "hip;gpu;gfx1151")
  endforeach()
  # Reuse the synthetic model fixture, with short prompts around a reduced
  # sparse budget: exercises actual Executor capture/replay without GGUFs.
  add_test(NAME flash_next_gyro.decode_graph
    COMMAND flash_next_gyro_wide_verification_test --decode-graph-only)
  set_tests_properties(flash_next_gyro.decode_graph PROPERTIES
    LABELS "flash_next_gyro;hip;gpu;gfx1151")
  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(flash_next_gyro_attention_ops_test PRIVATE
    gufo_flash_next_gyro_reference)

  add_executable(flash_next_gyro_snapshot_test EXCLUDE_FROM_ALL
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/snapshot_test.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/device_headroom_fault.cpp")
  target_include_directories(flash_next_gyro_snapshot_test BEFORE PRIVATE
    "${GUFO_EXTENSION_BUILD_ROOT}")
  target_link_libraries(flash_next_gyro_snapshot_test PRIVATE
    flash_next_gyro_test_hip ${CMAKE_DL_LIBS})

  # Benches and diagnostic probes are built explicitly so routine iterations
  # stay focused.
  foreach(bench IN ITEMS dense_gemm_bench hc_mix_bench attn_bench
      decode_attn_bench routed_gemm_bench)
    add_executable(flash_next_gyro_${bench} EXCLUDE_FROM_ALL
      "${GUFO_EXTENSION_BUILD_ROOT}/tests/${bench}.cpp")
    target_include_directories(flash_next_gyro_${bench} BEFORE PRIVATE
      "${GUFO_EXTENSION_BUILD_ROOT}")
    target_include_directories(flash_next_gyro_${bench} SYSTEM PRIVATE
      "${GFN_PUBLIC_INCLUDE}")
    target_link_libraries(flash_next_gyro_${bench} PRIVATE flash_next_gyro_test_hip)
  endforeach()
  add_executable(flash_next_gyro_gpu_probe EXCLUDE_FROM_ALL
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/gpu_probe.cpp"
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/mtp_audit.cpp")
  target_include_directories(flash_next_gyro_gpu_probe BEFORE PRIVATE
    "${GUFO_EXTENSION_BUILD_ROOT}")
  target_link_libraries(flash_next_gyro_gpu_probe PRIVATE flash_next_gyro_test_hip)
  add_executable(flash_next_gyro_reference_probe EXCLUDE_FROM_ALL
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/reference_probe.cpp")
  target_include_directories(flash_next_gyro_reference_probe BEFORE PRIVATE
    "${GUFO_EXTENSION_BUILD_ROOT}")
  target_link_libraries(flash_next_gyro_reference_probe PRIVATE
    gufo_flash_next_gyro_reference)
  set_source_files_properties(
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/reference_probe.cpp" PROPERTIES
    COMPILE_OPTIONS "-O2")
endif()
