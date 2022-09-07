#pragma once
#include <asci/types.hpp>
#include <asci/hamiltonian_generator.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/davidson.hpp>
#include <chrono>


namespace asci {


template <size_t N, typename index_t = int32_t>
double selected_ci_diag( 
  wavefunction_iterator_t<N> dets_begin,
  wavefunction_iterator_t<N> dets_end,
  HamiltonianGenerator<N>&   ham_gen,
  double                     h_el_tol,
  size_t                     davidson_max_m,
  double                     davidson_res_tol,
  std::vector<double>&       C_local,
  MPI_Comm                   comm,
  const bool                 quiet = false
) {

  if( !quiet )
  {
    std::cout << "* Diagonalizing CI Hamiltonian over " 
              << std::distance(dets_begin,dets_end)
              << " Determinants" << std::endl;

    std::cout << "  * Hamiltonian Knobs:" << std::endl
              << "    * Hamiltonian Element Tolerance = " << h_el_tol << std::endl;

    std::cout << "  * Davidson Knobs:" << std::endl
              << "    * Residual Tol = " << davidson_res_tol << std::endl
              << "    * Max M        = " << davidson_max_m << std::endl;
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  MPI_Barrier(comm);
  auto H_st = clock_type::now();
  // Generate Hamiltonian
  auto H = make_dist_csr_hamiltonian<index_t>( comm, dets_begin, dets_end,
    ham_gen, h_el_tol );

  MPI_Barrier(comm);
  auto H_en = clock_type::now();

  // Get total NNZ
  size_t local_nnz = H.nnz();
  size_t total_nnz;
  MPI_Allreduce( &local_nnz, &total_nnz, 1, MPI_UINT64_T, MPI_SUM, comm );
  if(!quiet)
  {
    std::cout << "  * Hamiltonian NNZ = " << total_nnz << std::endl;

    std::cout << "  * Timings:" << std::endl;
    std::cout << "    * Hamiltonian Construction = " 
      << duration_type(H_en-H_st).count() << std::endl;
  }

  // Resize eigenvector size
  C_local.resize( H.local_row_extent() );

  // Solve EVP
  MPI_Barrier(comm);
  auto dav_st = clock_type::now();
  #if 1
  double E = p_davidson( davidson_max_m, H, davidson_res_tol, C_local.data() );
  #else
  const size_t ndets = std::distance(dets_begin,dets_end);
  std::vector<double> H_dense(ndets*ndets);
  sparsexx::convert_to_dense( H.diagonal_tile(), H_dense.data(), ndets );

  std::vector<double> W(ndets);
  lapack::syevd( lapack::Job::NoVec, lapack::Uplo::Lower, ndets, 
    H_dense.data(), ndets, W.data() );
  auto E = W[0];
  #endif
  MPI_Barrier(comm);
  auto dav_en = clock_type::now();
  if( !quiet )
  {
    std::cout << "    * Davidson                 = " 
      << duration_type(dav_en-dav_st).count() << std::endl;
    std::cout << std::endl;
  } 

  return E;

}


} // namespace asci
