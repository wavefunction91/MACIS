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

  auto logger = spdlog::get("ci_solver");
  if(!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  logger->info("[Selected CI Solver]:");
  logger->info("  {} = {:6}, {} = {:.5e}, {} = {:.5e}, {} = {:4}",
    "NDETS", std::distance(dets_begin, dets_end),
    "MATEL_TOL", h_el_tol,
    "RES_TOL",   davidson_res_tol,
    "MAX_SUB",   davidson_max_m
  );

  //if( !quiet )
  //{
  //  std::cout << "* Diagonalizing CI Hamiltonian over " 
  //            << std::distance(dets_begin,dets_end)
  //            << " Determinants" << std::endl;

  //  std::cout << "  * Hamiltonian Knobs:" << std::endl
  //            << "    * Hamiltonian Element Tolerance = " << h_el_tol << std::endl;

  //  std::cout << "  * Davidson Knobs:" << std::endl
  //            << "    * Residual Tol = " << davidson_res_tol << std::endl
  //            << "    * Max M        = " << davidson_max_m << std::endl;
  //}

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Generate Hamiltonian
  MPI_Barrier(comm); auto H_st = clock_type::now();

  auto H = make_dist_csr_hamiltonian<index_t>( comm, dets_begin, dets_end,
    ham_gen, h_el_tol );

  MPI_Barrier(comm); auto H_en = clock_type::now();

  // Get total NNZ
  size_t local_nnz = H.nnz();
  size_t total_nnz;
  MPI_Allreduce( &local_nnz, &total_nnz, 1, MPI_UINT64_T, MPI_SUM, comm );
  logger->info("  {}   = {:6}, {}     = {:.5e} ms",
    "NNZ", total_nnz, "H_DUR", duration_type(H_en-H_st).count()
  );

  // Resize eigenvector size
  C_local.resize( H.local_row_extent() );

  // Setup guess
  auto D_local = extract_diagonal_elements( H.diagonal_tile() );
  p_diagonal_guess(C_local.size(), H, C_local.data());

  // Setup Davidson Functor
  SparseMatrixOperator op(H);

  // Solve EVP
  MPI_Barrier(comm); auto dav_st = clock_type::now();

  double E = p_davidson( H.local_row_extent(), davidson_max_m, op, 
    D_local.data(), davidson_res_tol, C_local.data(), H.comm() );

  MPI_Barrier(comm); auto dav_en = clock_type::now();

  if( !quiet )
  {
    std::cout << "    * Davidson                 = " 
      << duration_type(dav_en-dav_st).count() << std::endl;
    std::cout << std::endl;
  } 
  logger->info("  {} = {:.6e} Eh, {} = {:.5e} ms", 
    "E0", E,
    "DAVIDSON_DUR", duration_type(dav_en-dav_st).count());

  return E;

}


} // namespace asci
