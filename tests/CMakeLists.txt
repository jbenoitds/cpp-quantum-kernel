set(QK_TESTS linalg boys integrals scf hf1d)

foreach(name IN LISTS QK_TESTS)
  add_executable(test_${name} test_${name}.cpp test_main.cpp)
  target_link_libraries(test_${name} PRIVATE qk::qk)
  target_compile_definitions(test_${name} PRIVATE QK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME ${name} COMMAND test_${name})
endforeach()

# End-to-end check of the command-line program.
if(QK_BUILD_APPS)
  add_test(NAME cli_scf_water
    COMMAND qk-scf ${PROJECT_SOURCE_DIR}/examples/h2o_crawford.xyz --bohr --mp2 --quiet)
  set_tests_properties(cli_scf_water PROPERTIES
    PASS_REGULAR_EXPRESSION "RHF total energy +-74\\.942079928")
  add_test(NAME cli_hf1d_helium COMMAND qk-hf1d --quiet)
  set_tests_properties(cli_hf1d_helium PROPERTIES PASS_REGULAR_EXPRESSION "Total energy +-2\\.22420955")
endif()
