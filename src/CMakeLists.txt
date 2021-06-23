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

add_library(CMZed ${SOURCES_CXX})
set_target_properties(CMZed PROPERTIES VERSION ${PROJECT_VERSION})
target_include_directories( CMZed PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/inc>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(CMZed PUBLIC cxx_std_20)
target_link_libraries( CMZed PUBLIC "${LAPACK_LIBRARIES}" Eigen3::Eigen )
if( TARGET OpenMP::OpenMP_CXX )
  target_link_libraries( CMZed PUBLIC OpenMP::OpenMP_CXX )
endif()

add_library( cmz::ed ALIAS CMZed )

# install executable and header
include( GNUInstallDirs )

install( TARGETS CMZed
  EXPORT cmzed-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties( CMZed PROPERTIES EXPORT_NAME ed )
export(EXPORT cmzed-targets
      NAMESPACE cmz::
      FILE "${PROJECT_BINARY_DIR}/cmzed-targets.cmake")

# Export target to script
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cmzed )
install( EXPORT cmzed-targets
  FILE         cmzed-targets.cmake
  NAMESPACE    cmz::
  DESTINATION  ${INSTALL_CONFIGDIR}
)

# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cmzed-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)


# Setup x-config.cmake
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/cmzed-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmzed-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install CMake files
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cmzed-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmzed-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)
