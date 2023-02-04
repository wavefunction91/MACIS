#pragma once
#include <asci/types.hpp>
#include <asci/hamiltonian_generator.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/davidson.hpp>
#include <chrono>


namespace asci {



template <typename SpMatType>
double selected_ci_diag( 
  const SpMatType&     H,
  size_t               davidson_max_m,
  double               davidson_res_tol,
  std::vector<double>& C_local,
  MPI_Comm             comm
) {

  auto logger = spdlog::get("ci_solver");
  if(!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Resize eigenvector size
  C_local.resize( H.local_row_extent(), 0 );

  // Extract Diagonal
  auto D_local = extract_diagonal_elements( H.diagonal_tile() );

  // Setup guess
  auto max_c = *std::max_element(C_local.begin(), C_local.end(),
    [](auto a, auto b){ return std::abs(a) < std::abs(b); });
  max_c = std::abs(max_c);

  if(max_c > (1./C_local.size())) {
    logger->info("  * Will use passed vector as guess");
  } else {
    logger->info("  * Will generate identity guess");
    p_diagonal_guess(C_local.size(), H, C_local.data());
  }

  // Setup Davidson Functor
  SparseMatrixOperator op(H);

  // Solve EVP
  MPI_Barrier(comm); auto dav_st = clock_type::now();

  auto [niter, E] = p_davidson( H.local_row_extent(), davidson_max_m, op, 
    D_local.data(), davidson_res_tol, C_local.data(), H.comm() );

  MPI_Barrier(comm); auto dav_en = clock_type::now();

  logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", 
    "DAV_NITER", niter,
    "E0", E,
    "DAVIDSON_DUR", duration_type(dav_en-dav_st).count());

  return E;

}





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
  logger->info("  {} = {:.2e} GiB", "HMEM_LOC", H.mem_footprint()/1073741824.);

  // Solve EVP
  auto E = selected_ci_diag(H, davidson_max_m, 
    davidson_res_tol, C_local, comm);

  return E;

}


} // namespace asci
