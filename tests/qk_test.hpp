// Minimal, dependency-free unit-test harness for cpp-quantum-kernel.
//
//   QK_TEST(name) { QK_CHECK(cond); QK_CHECK_NEAR(a, b, tol); QK_CHECK_THROWS(expr); }
//
// Each test binary links tests/test_main.cpp; pass a substring to run a subset.
#pragma once

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace qk_test {

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

inline int& failures() {
  static int f = 0;
  return f;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline void report(const char* file, int line, const std::string& msg) {
  ++failures();
  std::cerr << "    FAILED " << file << ":" << line << ": " << msg << "\n";
}

}  // namespace qk_test

#define QK_CAT_(a, b) a##b
#define QK_CAT(a, b) QK_CAT_(a, b)
#define QK_TEST(name)                                                              \
  static void QK_CAT(qk_test_fn_, __LINE__)();                                     \
  static const qk_test::Registrar QK_CAT(qk_test_reg_, __LINE__)(name, &QK_CAT(qk_test_fn_, __LINE__)); \
  static void QK_CAT(qk_test_fn_, __LINE__)()

#define QK_CHECK(cond)                                                    \
  do {                                                                    \
    if (!(cond)) qk_test::report(__FILE__, __LINE__, "QK_CHECK(" #cond ")"); \
  } while (0)

#define QK_CHECK_NEAR(actual, expected, tol)                                                          \
  do {                                                                                                \
    const double qk_a_ = (actual), qk_e_ = (expected), qk_t_ = (tol);                                 \
    if (!(std::abs(qk_a_ - qk_e_) <= qk_t_)) {                                                        \
      std::ostringstream qk_os_;                                                                      \
      qk_os_ << std::setprecision(15) << #actual << " = " << qk_a_ << ", expected " << qk_e_          \
             << " (|diff| = " << std::abs(qk_a_ - qk_e_) << " > " << qk_t_ << ")";                    \
      qk_test::report(__FILE__, __LINE__, qk_os_.str());                                              \
    }                                                                                                 \
  } while (0)

#define QK_CHECK_THROWS(expr)                                                         \
  do {                                                                                \
    bool qk_threw_ = false;                                                           \
    try {                                                                             \
      (void)(expr);                                                                   \
    } catch (...) {                                                                   \
      qk_threw_ = true;                                                               \
    }                                                                                 \
    if (!qk_threw_) qk_test::report(__FILE__, __LINE__, "expected exception: " #expr); \
  } while (0)
