#pragma once

#include "csr_hamiltonian.hpp"
#include "davidson.hpp"
#include <chrono>

void serial_davidson_test( const SetSlaterDets& stts, const FermionHamil& Hop,
  const intgrls::integrals& ints, double H_tol, int max_m, double eig_tol ) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  auto H_st = clock_type::now();
  auto H  = make_csr_hamiltonian( stts, Hop, ints, H_tol );
  auto H_en = clock_type::now();

  auto eig_st = clock_type::now();
  auto E0 = davidson(max_m, H, eig_tol);
  auto eig_en = clock_type::now();

  std::cout << "E0  = " << E0 << std::endl;
  std::cout << "NNZ = " << H.nnz() << std::endl;
  std::cout << "H Construction = " << duration_type(H_en-H_st).count() << " ms\n";
  std::cout << "Davidson       = " << duration_type(eig_en-eig_st).count() 
            << " ms" << std::endl;

}
