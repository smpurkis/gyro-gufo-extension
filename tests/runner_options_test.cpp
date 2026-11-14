#include <cstddef>
#include <limits>

#include "model/runner.hpp"

namespace qfn = gufo::gyro;

static_assert(qfn::DecodeConcurrencyForSessions(0) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(1) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(2) == 2);
static_assert(qfn::DecodeConcurrencyForSessions(4) == 4);
static_assert(qfn::DecodeConcurrencyForSessions(8) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(9) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(
                  std::numeric_limits<std::size_t>::max()) == 8);

int main() {}
