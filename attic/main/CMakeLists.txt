# include inc/ directory
include_directories(${CMAKE_SOURCE_DIR}/inc 
                    ${PROJECT_BINARY_DIR}/inc
                   )

# add to the library
add_executable(cmzed_main cmzed_main.c++)

# links
target_link_libraries(cmzed_main PUBLIC 
                      ASCI 
                      ${LAPACK_LIBRARIES}
                      Eigen3::Eigen
                     )

# install executable
install(TARGETS cmzed_main DESTINATION bin)
