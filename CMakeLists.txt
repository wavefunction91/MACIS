cmake_minimum_required(VERSION 3.14)

# set the project name
project(CMZed VERSION 1.0 LANGUAGES CXX)

option( CMZED_ENABLE_OPENMP "Enable OpenMP Bindings" ON )

# find lapack
find_package(LAPACK REQUIRED)

# Find OpenMP
if( CMZED_ENABLE_OPENMP )
  find_package(OpenMP) 
endif()

find_package( Eigen3 CONFIG REQUIRED )

# option of building tests
option(BUILD_TESTS "Build test cases, checking different aspects of the library" OFF)

add_subdirectory(inc)
add_subdirectory(src)
add_subdirectory(main)
if(BUILD_TESTS)
  add_subdirectory(test)
endif()
add_subdirectory(docs)

# testing
enable_testing()

#Testing Slater determinants
add_test(NAME test_Slater COMMAND ${PROJECT_BINARY_DIR}/test/test_Slater.x 2 1 1)

#Testing Hubbard dimer Hamiltonian
add_test(NAME test_Hamil COMMAND ${PROJECT_BINARY_DIR}/test/test_Hamil.x ${PROJECT_BINARY_DIR}/test/test_Hamil/input.in)
set_tests_properties(test_Hamil
   PROPERTIES PASS_REGULAR_EXPRESSION "Ground state energy: -0.472136"
                    )

#Testing H2O 6-31g in (8e, 5o) active space
add_test(NAME test_h2o COMMAND ${PROJECT_BINARY_DIR}/test/test_h2o.x ${PROJECT_BINARY_DIR}/test/test_h2o/input.in)
set_tests_properties(test_h2o 
   PROPERTIES PASS_REGULAR_EXPRESSION "Ground state energy: -7.59583e"
                    )

#Testing H2O 6-31g in (8e, 5o) active space through ED routine
add_test(NAME test_ed COMMAND ${PROJECT_BINARY_DIR}/test/test_ed.x ${PROJECT_BINARY_DIR}/test/test_ed/input.in)
set_tests_properties(test_ed 
   PROPERTIES PASS_REGULAR_EXPRESSION "Ground state energy: -7.59583e"
                    )
