add_library( asci
  fcidump.cxx
  fock_matrices.cxx
  transform.cxx
  orbital_gradient.cxx
  mcscf.cxx
)

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
target_link_libraries(asci PUBLIC lobpcgxx sparsexx)

# SPDLOG
include(asci-spdlog)
target_link_libraries(asci PUBLIC spdlog::spdlog)
