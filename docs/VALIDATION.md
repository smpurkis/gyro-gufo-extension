# Validation

Checked on a Strix Halo (`gfx1151`), ROCm 7.2.4, Gufo `main` at `029a1569`
(extension API 0.3.35) plus the `split.count = 0` reader tolerance from
[gufo-org/gufo#1313](https://github.com/gufo-org/gufo/pull/1313). The extension is compiled into the build with

```sh
cmake --preset gpu-test \
  -DCMAKE_PREFIX_PATH="/opt/rocm-7.2.4;/opt/rocm-10.0.0" \
  -DGUFO_EXTENSION_DIRS=/absolute/path/to/gyro-gufo-extension
cmake --build --preset gpu-test --target gufo flash_next_gyro_tests
```

`gufo diagnose --section extensions` then lists both model extensions side by
side, which is the point of the rename: the official `qwen38_flash_next`
provider and `flash_next_gyro`'s `flash-next-gyro-m` are registered together.

```
Built extensions:
  ...
  flash_next_gyro 0.1.0 (../gyro-gufo-extension)
  qwen38_flash_next 0.1.0 (src/models/qwen38_flash_next)
  provider flash-next-gyro-m: available
  provider flash-next-q4: available
```

## Tests

`ctest --test-dir build/gpu-test -R flash_next_gyro` — 26 tests, all pass; the
real-artifact check is skipped unless `QWEN38_FLASH_NEXT_GYRO_MODEL` points at
the GGUF.

| group | result |
| --- | --- |
| host contracts (`probe`, `config`, `continuation_layout`, `runner_options`, `ngram`, `rope_scaling`, `rope_scaling_oracle`, `mtp_sampling`, `prompt_lookup`, `continuation_identity`) | pass |
| provider probe against the artifact (`probe` covers both directions: `flash-next-q4` declines trellis experts, `flash-next-gyro-m` declines `Q4_K` experts and half-trellised trunks) | pass |
| `gyro_artifact` with `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` | pass — 48 trellis expert layers, 512 experts, 10 used, hidden 2560, ffn 640, rotation signs bound |
| GPU operators (`trellis_ops`, `routed_wmma_ops`, `mtp_ops`, `hc_mix_ops`, `hc_route_ops`, `gdn_ops`, `attention_ops`, `moe_epilogue_ops`, `moe_ids_ops`, `projection_ops`, `select_ops`, `verification_sampling`, `wide_verification`, `decode_graph`) | 15/15 pass |

Three of the model's in-tree tests were not ported: `session_test`,
`continuation_adapter_test` and `continuation_probe` reach Gufo internals
(`src/cli/serve/inference_backend.hpp`, `src/cache/slot.hpp`,
`src/core/hip/committed_backing.hpp`) that the audited extension surface does not
publish, and `snapshot_copy` exercises Gufo core's download path rather than
this model.

## Serving

`--sessions 1 --context 262144 --cache-ram-bytes 4294967296`, MTP draft
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` at `-d 3`, vision sidecar
`mmproj-F16.gguf`:

| measurement | value |
| --- | --- |
| load | 13.3 s, RSS 1.6 GiB, 80,788 MiB of the 128 GiB device pool |
| prefill | **1510.4 tok/s on an 11,789-token prompt** (cold); the earlier 28-token prompt showed only 165 tok/s because fixed per-request overhead dominates at that length |
| prompt cache | identical 11,789-token prompt reused all 11,789 tokens (PP not charged) |
| decode | 45.6 tok/s with MTP on a short prompt (37 drafts, 19 accepted); 29.5 tok/s after an 11,789-token prefill |
| image request | answered from a generated PNG; 86 prompt tokens, 53.3 tok/s |
| live continuation | identical prompt reused 399 cached tokens (validated on the prior API series) |
| disk continuation | after a server restart on the same cache directory, `event=disk_restore tokens=392 bytes=143468224 pieces=189`, 392 cached tokens (prior API series) |
| faults | 0 `device_lost`, 0 HIP errors, 0 OOM |

## The reader option this extension needs

Flash-Next's engine re-opens the artifact while binding weights, so both
`model/engine.cpp` and `model/runner.cpp` open it with
`{.allow_unsized_tensors = true}`. Without that the TQ2_T expert rows are
rejected during binding and serving fails with
`Invalid GGUF tensor shape or unsupported storage type: blk.0.ffn_down_exps.weight`.
It is only reachable in a serving run, so it was found here rather than by the
header-level tests.


## Module build

`gufo extension install <local checkout> --yes` on a gufo built from `029a1569`
configured, built and installed `flash_next_gyro 0.1.0` as a loadable module;
`gufo diagnose --section extensions` then lists
`flash_next_gyro 0.1.0 (module:.../gufo-ext-flash_next_gyro.so)` beside the
built-in `qwen38_flash_next`, with both `flash-next-gyro-m` and `flash-next-q4`
available. The public surface location is discovered from `gufo_ext_runtime`'s
include directories, so the same `extension.cmake` works for both compiled-in
and module builds.
