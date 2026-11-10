// cpp-quantum-kernel - thin OpenMP shim (everything degrades to serial).
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qk {

inline bool openmp_enabled() noexcept {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

inline int max_threads() noexcept {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int thread_id() noexcept {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline void set_num_threads(int n) noexcept {
#ifdef _OPENMP
  if (n > 0) omp_set_num_threads(n);
#else
  (void)n;
#endif
}

}  // namespace qk
