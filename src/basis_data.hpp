// Internal: built-in basis-set table (see basis_data.cpp, generated).
#pragma once

#include <vector>

namespace qk::detail {

struct BuiltinBasisEntry {
  const char* basis;  ///< canonical lower-case name
  int Z;
  const char* g94;    ///< Gaussian94 block for this element
};

const std::vector<BuiltinBasisEntry>& builtin_basis_table();

}  // namespace qk::detail
