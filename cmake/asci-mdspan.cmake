include(FetchContent)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
)
set( MDSPAN_CXX_STANDARD 17 CACHE STRING "" FORCE)
FetchContent_MakeAvailable( mdspan )
