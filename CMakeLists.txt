cmake_minimum_required(VERSION 3.14)
project(MACIS VERSION 0.1 LANGUAGES C CXX)


option( MACIS_ENABLE_OPENMP "Enable OpenMP Bindings" ON )
option( MACIS_ENABLE_BOOST  "Enable Boost" ON )

# CMake Modules
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include( macis-cmake-modules )
if( MACIS_ENABLE_OPENMP )
  find_package(OpenMP)
endif()

add_subdirectory(src)

# Tests
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
