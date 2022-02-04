cmake_minimum_required(VERSION 3.14)

# set the project name
project(CMZed VERSION 1.0 LANGUAGES C CXX)

# Pull proper BLAS/LAPACK discovery modules
include( FetchContent )
FetchContent_Declare( linalg-cmake-modules
  #GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
  #GIT_TAG        89c82b78505f60657ef33b5bf38e5da059dff7f2
  GIT_REPOSITORY https://github.com/ajaypanyala/linalg-cmake-modules.git
  GIT_TAG        main
)
FetchContent_GetProperties( linalg-cmake-modules )
if( NOT linalg-cmake-modules_POPULATED )
  FetchContent_Populate( linalg-cmake-modules )
  list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
endif()

# IPS4O Sort
FetchContent_Declare( ips4o
  GIT_REPOSITORY https://github.com/SaschaWitt/ips4o.git 
)
FetchContent_GetProperties( ips4o )
if( NOT ips4o_POPULATED )
  FetchContent_Populate( ips4o )
endif()
add_library( ips4o INTERFACE )
target_include_directories( ips4o INTERFACE ${ips4o_SOURCE_DIR} )
target_link_libraries( ips4o INTERFACE atomic )

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

add_subdirectory(external/lobpcgxx)
add_subdirectory(external/sparsexx)

add_subdirectory(inc)
add_subdirectory(src)
add_subdirectory(main)
  add_subdirectory(test)
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
