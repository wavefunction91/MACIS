cmake_minimum_required(VERSION 3.14)
project(asci VERSION 0.1 LANGUAGES C CXX)


option( ASCI_ENABLE_OPENMP "Enable OpenMP Bindings" ON )

# CMake Modules
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include( asci-cmake-modules )

# Packaged Dependencies
add_subdirectory(external/lobpcgxx)
add_subdirectory(external/sparsexx)

# ASCI library
add_subdirectory(src)

# Tests
enable_testing()
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
