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
)
target_link_libraries( asci_test PUBLIC asci Catch2::Catch2 )
