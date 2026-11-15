# Qwen3.8 Flash-Next Gyro-M extension

Agention **Gyro** trellis-quantized Flash-Next (`TQ2_T` routed experts) as a
Gufo community extension for AMD Strix Halo (`gfx1151`).

Maintainer: [`smpurkis`](https://github.com/smpurkis).

## What it serves

| artifact | provider | notes |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` | `flash-next-gyro-m` | 48 blocks, 512 routed experts (10 active), 262,144 native context, trellised experts + Q8_0 trunk |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | `flash-next-gyro-m` (`--mtp-model`) | MTP drafter, up to 7 draft tokens |
| `mmproj-BF16.gguf` | `flash-next-gyro-m` (`--mmproj`) | Qwen3VL vision projector |

The provider's probe accepts **only** artifacts whose routed experts are all
`TQ2_T` rows, so it installs beside the official `flash-next-q4` extension and
never claims a `UD-Q4_K_XL` file. It uses its own artifact key
(`gguf:qwen4exp-gyro`) and its own namespaces, so a module build cannot bind to
the official extension's kernels.

## Memory

Measured from the GGUF tensor lists with Gufo's own block sizes:

| part | bytes | resident on device |
| --- | ---: | --- |
| routed experts (TQ2_T, 128 weights per 34-byte block) | 29.88 GiB | yes |
| trunk (hyper-connections, QSA attention, GDN, indexer, routers) | 3.64 GiB | yes |
| embeddings + output head (Q8_0) | 1.45 GiB | yes |
| `per_layer_token_embd.weight` n-gram table (Q8_0, 320,001,536 rows x 160) | 50.66 GiB | **no** — read from disk with a 128 MiB row cache |
| **file total** | **85.65 GiB** | ~35 GiB resident |

For comparison, `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ2_XS` is 70.62 GiB on
disk (75.8 GB) but keeps ~43.8 GiB resident, because its n-gram table is
IQ4_NL (26.82 GiB) instead of Q8_0. Gyro-M has the larger file and the smaller
resident footprint.

## Install

Requires Gufo 0.32 or newer (`api_version >= 0.3.0`) built with the HIP backend;
validated against Gufo `main` at `800ce328`.

The vision sidecar can be Unsloth's `mmproj-BF16.gguf` (what Gufo's Qwen vision
stack expects) or Agention's `mmproj-F16.gguf`, which this extension also binds.

```sh
# as a module into an installed Gufo
gufo extension install https://github.com/smpurkis/gyro-gufo-extension@<sha>

# or compiled into your own build
cmake --preset release -DGUFO_EXTENSION_DIRS=/absolute/path/to/gyro-gufo-extension
cmake --build --preset release --parallel
```

Serve it:

```sh
gufo serve --port 5909 --sessions 2 --context 524288 --enable-yarn \
  --model Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf \
  --mmproj mmproj-BF16.gguf \
  --served-model-name Qwen3.8-Flash-Next-Gyro-M \
  --cache-ram-bytes 8589934592 \
  --speculative mtp --mtp-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -d 7
```

## Contents

```
model/          Flash-Next engine, executor, batched MTP, weights, n-gram table
model/gyro/     vendored Gyro TQ2_T decoder (Agention, Apache-2.0)
model/kernels/  ROCm kernels, including trellis.hip.cpp and the vendored mmq tier
family/         Qwen tokenizer, chat template, HIP kernels, vision encoder
tests/          operator and contract tests, run through ctest as flash_next_gyro.*
```

Everything is in `gufo::models::flash_next_gyro`, not `gufo::models::qwen38_flash_next`.

## Verification

```sh
ctest --test-dir build/gpu-test -R flash_next_gyro           # host + GPU operators
QWEN38_FLASH_NEXT_GYRO_MODEL=/path/to/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf \
  ctest --test-dir build/gpu-test -R flash_next_gyro.gyro_artifact
```

Measured serving numbers, the coexistence check and the reader option this
extension needs are in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Licenses

Gufo sources: MIT (`LICENSES/gufo.txt`). Gyro reference decoder: Apache-2.0,
AgentionAI (`licenses/gyro.txt`, `model/gyro/NOTICE`). Model weights have their
own licenses (Qwen Community License 1.0) and are not distributed here.
