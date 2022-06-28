# include inc/ directory
include_directories(${CMAKE_SOURCE_DIR}/inc)
set(PATH_TO_TEST ${CMAKE_CURRENT_SOURCE_DIR})

#loop over tests
list( APPEND TEST_DIRECTORIES
      test_Slater
      test_Hamil
      test_h2o
      test_ed
      test_cr2_new
      test_cr2_grow
      test_c2
      test_parallel_new
      test_asci
      test_asci_throughFunc
      test_ed_func
      test_bandlan
      test_getGFbasis
      test_buildGFHamil
      test_HamilBuilds
      test_GF
    )
foreach( testdir ${TEST_DIRECTORIES} )
    # add executable
    add_executable( ${testdir}.x ${testdir}/${testdir}.c++ )
    # links
    target_link_libraries( ${testdir}.x asci lobpcgxx sparsexx ips4o )
    # if applicable, copy input file
    if( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${testdir}/input.in.in" )
      configure_file(${testdir}/input.in.in ${testdir}/input.in)
    endif()
endforeach( testdir ${TEST_DIRECTORIES} )

#add_executable( test_parallel.x test_parallel/test_parallel.c++ )
#target_link_libraries( test_parallel.x CMZed lobpcgxx sparsexx )
