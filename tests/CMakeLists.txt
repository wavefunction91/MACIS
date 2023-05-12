include(macis-catch2)

add_executable( macis_test 
  ut_main.cxx 
  bitset_operations.cxx 
  sd_operations.cxx 
  fcidump.cxx 
  read_wavefunction.cxx
  double_loop.cxx
  csr_hamiltonian.cxx
  davidson.cxx
  transform.cxx
  fock_matrices.cxx
  mcscf.cxx
  asci.cxx
  dist_quickselect.cxx
)
target_link_libraries( macis_test PUBLIC macis Catch2::Catch2 )
target_include_directories( macis_test PUBLIC ${PROJECT_BINARY_DIR}/tests )

add_executable( standalone_driver standalone_driver.cxx ini_input.cxx )
target_link_libraries( standalone_driver PUBLIC macis )

set(REF_DATA_PREFIX "${PROJECT_SOURCE_DIR}/tests/ref_data")
configure_file( ut_common.hpp.in ${PROJECT_BINARY_DIR}/tests/ut_common.hpp)

#include(FetchContent)
#FetchContent_Declare( pybind11
#  GIT_REPOSITORY https://github.com/pybind/pybind11.git
#)
#FetchContent_MakeAvailable(pybind11)
#pybind11_add_module(pymacis pymod.cxx)
#target_link_libraries(pymacis PUBLIC macis)

add_test( NAME ASCI_SERIAL_TEST COMMAND $<TARGET_FILE:macis_test> )
