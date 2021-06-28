cmake_minimum_required( VERSION 3.13 FATAL_ERROR )
project( lobpcgxx VERSION 0.0.1 LANGUAGES C CXX )


find_package( OpenMP )
# BLAS++/LAPACK++
include( FetchContent )
FetchContent_Declare( blaspp
  GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git 
  GIT_TAG 2021.04.01
)
FetchContent_Declare( lapackpp
  GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git 
)

FetchContent_MakeAvailable( blaspp )
FetchContent_MakeAvailable( lapackpp )
target_compile_definitions( lapackpp PUBLIC LAPACK_COMPLEX_CPP )

add_library( lobpcgxx INTERFACE )
target_link_libraries( lobpcgxx INTERFACE blaspp lapackpp )
target_compile_features( lobpcgxx INTERFACE cxx_std_17 )

target_include_directories( lobpcgxx INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  #$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)



add_executable( lobpcg_tester examples/tester.cpp )
target_link_libraries( lobpcg_tester PUBLIC lobpcgxx )

# INSTALL rules
add_library( lobpcgxx::lobpcgxx ALIAS lobpcgxx )

include( GNUInstallDirs )

# TARGETS
install( TARGETS lobpcgxx
  EXPORT lobpcgxx-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties( lobpcgxx PROPERTIES EXPORT_NAME lobpcgxx )

# Install Headers
install( 
  DIRECTORY   ${PROJECT_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING PATTERN "*.hpp"
)


# Export target to scripe
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/lobpcgxx )
install( EXPORT lobpcgxx-targets
  FILE         lobpcgxx-targets.cmake
  NAMESPACE    lobpcgxx::
  DESTINATION  ${INSTALL_CONFIGDIR}
)


# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)



# Setup lobpcgxx-config.cmake
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/lobpcgxx-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)
