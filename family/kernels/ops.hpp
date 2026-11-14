#ifndef GUFO_MODELS_QWEN_HIP_OPS_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_HPP_

// Compatibility umbrella. New call sites should include the narrow operation
// family they use.
#include "family/kernels/ops/attention.hpp"
#include "family/kernels/ops/gemm.hpp"
#include "family/kernels/ops/norm_residual.hpp"
#include "family/kernels/ops/ssm.hpp"
#include "family/kernels/ops/swiglu.hpp"
#include "family/kernels/ops/token.hpp"

#endif  // GUFO_MODELS_QWEN_HIP_OPS_HPP_
