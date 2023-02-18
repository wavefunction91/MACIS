cmake_minimum_required(VERSION 3.14)
project(asci VERSION 0.1 LANGUAGES C CXX)


option( ASCI_ENABLE_OPENMP "Enable OpenMP Bindings" ON )
option( ASCI_ENABLE_BOOST  "Enable Boost" ON )

# CMake Modules
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include( asci-cmake-modules )

add_subdirectory(src)

# Tests
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
