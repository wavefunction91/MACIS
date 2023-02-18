add_library( asci
  fcidump.cxx
  fock_matrices.cxx
  transform.cxx
  orbital_gradient.cxx
  casscf.cxx
  moller_plesset.cxx
  orbital_energies.cxx
  orbital_rotation_utilities.cxx
  orbital_hessian.cxx
  orbital_steps.cxx
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
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(asci PUBLIC Eigen3::Eigen)

# Boost
if(ASCI_ENABLE_BOOST)
  find_package(Boost CONFIG REQUIRED)
  target_link_libraries(asci PUBLIC Boost::boost)
endif()

# Packaged Dependencies
target_link_libraries(asci PUBLIC lobpcgxx sparsexx)

# SPDLOG
include(asci-spdlog)
target_link_libraries(asci PUBLIC spdlog::spdlog)

# MDSPAN
include(asci-mdspan)
target_link_libraries(asci PUBLIC mdspan)


# Installation
include( GNUInstallDirs )
add_library(asci::asci ALIAS asci)
install( TARGETS asci lobpcgxx sparsexx ${ASCI_SPDLOG_EXPORT}
  EXPORT asci-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)



set_target_properties( asci PROPERTIES EXPORT_NAME asci )

# Export build tree
export(EXPORT asci-targets
      NAMESPACE asci::
      FILE "${PROJECT_BINARY_DIR}/asci-targets.cmake")

# Install staic headers
install( 
  DIRECTORY   ${PROJECT_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING PATTERN "*.hpp"
)

   
# Export target to script
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/asci )
install( EXPORT asci-targets
  FILE         asci-targets.cmake
  NAMESPACE    asci::
  DESTINATION  ${INSTALL_CONFIGDIR}
)

# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/asci-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)


# Setup asci-config.cmake
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/asci-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/asci-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install CMake files
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/asci-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/asci-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

# Install Custom Find Modules
include( ${linalg-cmake-modules_SOURCE_DIR}/LinAlgModulesMacros.cmake )
install_linalg_modules( INSTALL_CONFIGDIR )
