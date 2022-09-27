include(asci-catch2)

add_executable( asci_test 
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
)
target_link_libraries( asci_test PUBLIC asci Catch2::Catch2 )
target_include_directories( asci_test PUBLIC ${PROJECT_BINARY_DIR}/tests )

add_executable( standalone_driver standalone_driver.cxx ini_input.cxx )
target_link_libraries( standalone_driver PUBLIC asci )

set(REF_DATA_PREFIX "${PROJECT_SOURCE_DIR}/tests/ref_data")
configure_file( ut_common.hpp.in ${PROJECT_BINARY_DIR}/tests/ut_common.hpp)

#include(FetchContent)
#FetchContent_Declare( pybind11
#  GIT_REPOSITORY https://github.com/pybind/pybind11.git
#)
#FetchContent_MakeAvailable(pybind11)
#pybind11_add_module(pyasci pymod.cxx)
#target_link_libraries(pyasci PUBLIC asci)

add_test( NAME ASCI_SERIAL_TEST COMMAND $<TARGET_FILE:asci_test> )
