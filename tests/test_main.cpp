#include <chrono>
#include <exception>
#include <iostream>
#include <string>

#include "qk_test.hpp"

int main(int argc, char** argv) {
  const std::string filter = argc > 1 ? argv[1] : "";
  int run = 0, failed_cases = 0;
  for (const auto& tc : qk_test::registry()) {
    if (!filter.empty() && tc.name.find(filter) == std::string::npos) continue;
    ++run;
    const int before = qk_test::failures();
    const auto t0 = std::chrono::steady_clock::now();
    try {
      tc.fn();
    } catch (const std::exception& e) {
      qk_test::report(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = qk_test::failures() == before;
    if (!ok) ++failed_cases;
    std::cout << (ok ? "[ PASS ] " : "[ FAIL ] ") << tc.name << "  (" << ms << " ms)\n";
  }
  std::cout << run - failed_cases << "/" << run << " test cases passed\n";
  return failed_cases == 0 && run > 0 ? 0 : 1;
}
