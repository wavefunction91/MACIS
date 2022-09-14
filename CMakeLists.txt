cmake_minimum_required(VERSION 3.14)
project(asci VERSION 0.1 LANGUAGES C CXX)



option( ASCI_ENABLE_OPENMP "Enable OpenMP Bindings" ON )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include( asci-cmake-modules )


# Packaged Dependencies

# ASCI library
add_library( asci  src/fcidump.cxx src/fock_matrices.cxx src/transform.cxx src/orbital_gradient.cxx src/mcscf.cxx )
target_include_directories( asci PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    #$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(asci PUBLIC cxx_std_17)

# Parallelism
find_package( MPI REQUIRED )
target_link_libraries( asci PUBLIC MPI::MPI_CXX )
if(ASCI_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries( asci PUBLIC OpenMP::OpenMP_CXX )
endif()

# Linear Algebra
find_package(LAPACK REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(asci PUBLIC LAPACK::LAPACK Eigen3::Eigen blaspp lapackpp)

# Packaged Dependencies
include( asci-ips4o )
add_subdirectory(external/lobpcgxx)
add_subdirectory(external/sparsexx)
target_link_libraries(asci INTERFACE lobpcgxx sparsexx ips4o)


# Tests
enable_testing()
add_subdirectory(tests)
