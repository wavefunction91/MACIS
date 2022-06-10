# add to the library
list(APPEND SOURCES_CXX
     cmz_ed/eigsolver.c++
     cmz_ed/combins.c++
     cmz_ed/slaterdet.c++
     cmz_ed/hamil.c++
     cmz_ed/ed.c++
     cmz_ed/rdms.c++
     cmz_ed/utils.c++
    )

add_library(asci ${SOURCES_CXX})
set_target_properties(asci PROPERTIES VERSION ${PROJECT_VERSION})
target_include_directories( asci PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/inc>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(asci PUBLIC cxx_std_20)
target_link_libraries( asci PUBLIC "${LAPACK_LIBRARIES}" Eigen3::Eigen )
if( TARGET OpenMP::OpenMP_CXX )
  target_link_libraries( asci PUBLIC OpenMP::OpenMP_CXX )
endif()

add_library( ASCI ALIAS asci )

# install executable and header
include( GNUInstallDirs )

install( TARGETS asci
  EXPORT asci-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties( asci PROPERTIES EXPORT_NAME asci )
export(EXPORT asci-targets
      NAMESPACE cmz::
      FILE "${PROJECT_BINARY_DIR}/asci-targets.cmake")

# Export target to script
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/asci )
install( EXPORT asci-targets
  FILE         asci-targets.cmake
  NAMESPACE    cmz::
  DESTINATION  ${INSTALL_CONFIGDIR}
)

# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/asci-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)


# Setup x-config.cmake
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
