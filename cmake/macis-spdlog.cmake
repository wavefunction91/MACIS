# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

find_package(spdlog CONFIG QUIET)
if( NOT spdlog_FOUND )
  include(FetchContent)

  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.x
  )

  set(SPDLOG_INSTALL "ON" CACHE BOOL "Install SPDLOG" FORCE)
  FetchContent_MakeAvailable( spdlog )
  set(MACIS_SPDLOG_EXPORT spdlog CACHE STRING "" FORCE)
endif()

