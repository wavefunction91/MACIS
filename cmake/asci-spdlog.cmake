find_package(spdlog)
if( NOT spdlog_FOUND )
  include(FetchContent)

  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.x
  )

  FetchContent_MakeAvailable( spdlog )
endif()
