include( FetchContent )
# IPS4O Sort
FetchContent_Declare( ips4o
  GIT_REPOSITORY https://github.com/SaschaWitt/ips4o.git 
)
FetchContent_MakeAvailable( ips4o )
add_library( ips4o INTERFACE )
target_include_directories( ips4o INTERFACE ${ips4o_SOURCE_DIR} )
target_link_libraries( ips4o INTERFACE atomic )

