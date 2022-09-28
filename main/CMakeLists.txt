# include inc/ directory
include_directories(${CMAKE_SOURCE_DIR}/inc 
                    ${PROJECT_BINARY_DIR}/inc
                   )

# add to the library
add_executable(cmzed_main cmzed_main.c++)
add_executable(run_asci_gf run_asci_gf.c++)
add_executable(run_asci run_asci.c++)

# links
target_link_libraries(cmzed_main PUBLIC 
                      ASCI 
                      ${LAPACK_LIBRARIES}
                      Eigen3::Eigen
                     )
target_link_libraries(run_asci_gf PUBLIC 
                      ASCI 
                      ${LAPACK_LIBRARIES}
                      Eigen3::Eigen
                     )
target_link_libraries(run_asci PUBLIC 
                      ASCI 
                      ${LAPACK_LIBRARIES}
                      Eigen3::Eigen
                     )

# install executable
install(TARGETS cmzed_main DESTINATION bin)
install(TARGETS run_asci_gf DESTINATION bin)
install(TARGETS run_asci DESTINATION bin)
