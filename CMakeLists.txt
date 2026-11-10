cmake_minimum_required(VERSION 3.16)
project(cpp_quantum_kernel
  VERSION 0.1.0
  DESCRIPTION "Lightweight many-electron integral evaluator and Hartree-Fock kernels"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QK_ENABLE_OPENMP "Parallelise integral and Fock builds with OpenMP" ON)
option(QK_NATIVE_ARCH   "Compile with -march=native (GCC/Clang)" OFF)
option(QK_BUILD_TESTS   "Build the test suite" ON)
option(QK_BUILD_APPS    "Build the command-line programs" ON)
option(QK_BUILD_BENCH   "Build the benchmark program" ON)
option(QK_SANITIZE      "Build with AddressSanitizer + UBSan (GCC/Clang)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(qk
  src/linalg.cpp
  src/boys.cpp
  src/hermite.cpp
  src/molecule.cpp
  src/basis.cpp
  src/basis_data.cpp
  src/one_electron.cpp
  src/eri.cpp
  src/fock.cpp
  src/scf.cpp
  src/hf1d.cpp)
add_library(qk::qk ALIAS qk)

target_include_directories(qk
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(qk PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(qk PRIVATE /W4 /permissive- /bigobj /utf-8)
  target_compile_definitions(qk PUBLIC _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(qk PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
  if(QK_NATIVE_ARCH)
    target_compile_options(qk PUBLIC -march=native)
  endif()
endif()

set(QK_HAVE_OPENMP OFF)
if(QK_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(qk PUBLIC OpenMP::OpenMP_CXX)
    set(QK_HAVE_OPENMP ON)
    message(STATUS "qk: OpenMP enabled (${OpenMP_CXX_VERSION})")
  else()
    message(STATUS "qk: OpenMP not found, building serial")
  endif()
endif()
if(NOT QK_HAVE_OPENMP)
  # Serial build: the '#pragma omp' lines are intentionally ignored.
  if(MSVC)
    target_compile_options(qk PRIVATE /wd4068)
  else()
    target_compile_options(qk PRIVATE -Wno-unknown-pragmas)
  endif()
endif()

if(QK_SANITIZE AND NOT MSVC)
  target_compile_options(qk PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(qk PUBLIC -fsanitize=address,undefined)
endif()

if(QK_BUILD_APPS)
  add_executable(qk-scf apps/qk_scf.cpp)
  target_link_libraries(qk-scf PRIVATE qk::qk)
  add_executable(qk-hf1d apps/qk_hf1d.cpp)
  target_link_libraries(qk-hf1d PRIVATE qk::qk)
  add_executable(qk-example examples/library_usage.cpp)
  target_link_libraries(qk-example PRIVATE qk::qk)
endif()

if(QK_BUILD_BENCH)
  add_executable(qk-bench bench/bench_eri.cpp)
  target_link_libraries(qk-bench PRIVATE qk::qk)
endif()

if(QK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
install(TARGETS qk EXPORT qkTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/qk DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT qkTargets NAMESPACE qk:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qk)
if(QK_BUILD_APPS)
  install(TARGETS qk-scf qk-hf1d RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
