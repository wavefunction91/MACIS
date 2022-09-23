cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( asci_cmake_subproject LANGUAGES C CXX )


if( NOT GIT_REVISION )
  # Get current Git Revision
  find_package( Git REQUIRED )
  execute_process( 
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    OUTPUT_VARIABLE GIT_REVISION
    ERROR_QUIET
  )
  string( STRIP "${GIT_REVISION}" GIT_REVISION )
endif()


if( NOT GITHUB_URL )
  set( GITHUB_URL "https://github.com" )
endif()

if( NOT GITHUB_REPOSITORY )
  set( GITHUB_REPOSITORY "wavefunction91/ASCI-CI.git" )
endif()

message( STATUS "Pulling GIT_URL        = ${GITHUB_URL}" )
message( STATUS "Pulling GIT_REPOSITORY = ${GITHUB_REPOSITORY}" )
message( STATUS "Pulling GIT_REVISION   = ${GIT_REVISION}" )



include( FetchContent )
FetchContent_Declare( asci
  GIT_REPOSITORY "${GITHUB_URL}/${GITHUB_REPOSITORY}"
  GIT_TAG        ${GIT_REVISION}
)
FetchContent_MakeAvailable( asci )

add_executable( asci_link_tester asci_link_tester.cxx )
target_link_libraries( asci_link_tester PUBLIC asci::asci )