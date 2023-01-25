# Packaged Dependencies
set( LOBPCGXX_ENABLE_TESTS OFF CACHE BOOL "" FORCE )
set( SPARSEXX_ENABLE_TESTS OFF CACHE BOOL "" FORCE )
add_subdirectory(lobpcgxx)
add_subdirectory(sparsexx)

add_subdirectory(asci)
