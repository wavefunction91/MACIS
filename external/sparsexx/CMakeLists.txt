cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( sparsexx VERSION 0.1.0 LANGUAGES C CXX )

# Append local cmake directory to find CMAKE Modules
#if( CMAKE_MODULE_PATH )
#  list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
#else()
#  set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
#endif()
#list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
#list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/linalg-modules")


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include( FetchContent )
  FetchContent_Declare( linalg-cmake-modules
    #GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
    #GIT_TAG        89c82b78505f60657ef33b5bf38e5da059dff7f2
    GIT_REPOSITORY https://github.com/ajaypanyala/linalg-cmake-modules.git
    GIT_TAG        main
  )
  FetchContent_GetProperties( linalg-cmake-modules )
  if( NOT linalg-cmake-modules_POPULATED )
    FetchContent_Populate( linalg-cmake-modules )
    list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
  endif()
endif()


find_package( MPI      REQUIRED )
find_package( OpenMP   REQUIRED )

option( SPARSEXX_ENABLE_MKL       "Enable MKL Bindings"             ON )
option( SPARSEXX_ENABLE_CEREAL    "Enable serialization via cereal" ON )
option( SPARSEXX_ENABLE_RANGES_V3 "Enable Ranges-v3"                ON )

if( SPARSEXX_ENABLE_MKL )
  find_package( IntelMKL COMPONENTS lp64 REQUIRED )
  if( TARGET IntelMKL::IntelMKL )
    target_compile_definitions( IntelMKL::IntelMKL INTERFACE "SPARSEXX_ENABLE_MKL=1" )
    set( SPARSEXX_EXTERNAL_LIBRARIES IntelMKL::IntelMKL )
  endif()
endif()





#cmake_policy( SET CMP0079 NEW )
cmake_policy( VERSION 3.18 )
include( FetchContent )





# METIS 
FetchContent_Declare(
  metis_download
  URL http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
  #GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
)

FetchContent_GetProperties(metis_download)
if(NOT metis_download_POPULATED)
  FetchContent_Populate(metis_download)
  set( GKLIB_PATH ${metis_download_SOURCE_DIR}/GKlib CACHE PATH  "path to GKlib" )
  add_subdirectory(${metis_download_SOURCE_DIR} ${metis_download_BINARY_DIR})
  target_link_libraries( metis OpenMP::OpenMP_C )
  target_include_directories( metis PUBLIC ${metis_download_SOURCE_DIR}/include )
endif()
list(APPEND SPARSEXX_EXTERNAL_LIBRARIES metis)



if( SPARSEXX_ENABLE_RANGES_V3 )
  # RANGES-V3
  FetchContent_Declare(
    range_v3_download
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  )
  FetchContent_GetProperties(range_v3_download)
  if(NOT range_v3_download_POPULATED)
    FetchContent_Populate( range_v3_download )
    add_subdirectory(${range_v3_download_SOURCE_DIR} ${range_v3_download_BINARY_DIR})
    add_library( ranges INTERFACE IMPORTED )
    target_link_libraries( ranges INTERFACE range-v3 )
    target_compile_definitions( ranges INTERFACE "SPARSEXX_ENABLE_RANGES_V3=1" )
  endif()
  list(APPEND SPARSEXX_EXTERNAL_LIBRARIES ranges)
endif()




# CEREAL
if( SPARSEXX_ENABLE_CEREAL )
  FetchContent_Declare(
    cereal_download
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  )
  FetchContent_GetProperties(cereal_download)
  if(NOT cereal_download_POPULATED)
    FetchContent_Populate( cereal_download )
    add_library( cereal INTERFACE IMPORTED )
    set_target_properties( cereal PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${cereal_download_SOURCE_DIR}/include"
      INTERFACE_COMPILE_DEFINITIONS "CEREAL_THREAD_SAFE=1;SPARSEXX_ENABLE_CEREAL=1"
    )
  endif()
  list(APPEND SPARSEXX_EXTERNAL_LIBRARIES cereal)
endif()


add_library               ( sparsexx INTERFACE                               )
target_compile_features   ( sparsexx INTERFACE cxx_std_17                    )
target_link_libraries     ( sparsexx INTERFACE MPI::MPI_CXX OpenMP::OpenMP_CXX )
target_link_libraries     ( sparsexx INTERFACE ${SPARSEXX_EXTERNAL_LIBRARIES} )
target_include_directories( sparsexx INTERFACE ${PROJECT_SOURCE_DIR}/include )

target_compile_options( sparsexx INTERFACE "-Wall" "-Wextra" "-Wpedantic" "-Wnon-virtual-dtor")

#add_subdirectory( examples )
add_subdirectory( tests )

