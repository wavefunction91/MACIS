/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <chrono>
#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator.hpp>
#include <macis/solvers/davidson.hpp>
#include <macis/solvers/lobpcg.hpp>
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/util/submatrix.hpp>

namespace macis {

#ifdef MACIS_ENABLE_MPI
template <typename SpMatType>
double parallel_selected_ci_diag(const SpMatType& H, size_t davidson_max_m,
                                 double davidson_res_tol,
                                 std::vector<double>& C_local, MPI_Comm comm,
                                 const size_t nstates = 1) {
  auto logger = spdlog::get("ci_solver");
  if(!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Resize eigenvector size
  C_local.resize(H.local_row_extent(), 0);

  // Extract Diagonal
  auto D_local = extract_diagonal_elements(H.diagonal_tile());

  // Setup guess
  auto max_c = *std::max_element(
      C_local.begin(), C_local.end(),
      [](auto a, auto b) { return std::abs(a) < std::abs(b); });
  max_c = std::abs(max_c);

  if(max_c > (1. / C_local.size())) {
    logger->info("  * Will use passed vector as guess");
  } else {
    logger->info("  * Will generate identity guess");
    p_diagonal_guess(C_local.size(), H, C_local.data());
  }

  // Setup Davidson Functor
  SparseMatrixOperator op(H);

  // Solve EVP
  MPI_Barrier(comm);
  auto dav_st = clock_type::now();
  double E = 0.;
  size_t niter = 0;

  if(nstates == 1) {
    auto [niter0, E0] =
        p_davidson(H.local_row_extent(), davidson_max_m, op, D_local.data(),
                   davidson_res_tol, C_local.data() MACIS_MPI_CODE(, H.comm()));
    niter = niter0;
    E = E0;
  } else {
    std::vector<double> evals;
    std::vector<double> evecs;
    LobpcgGS(H.m(), nstates, op, evals, evecs, 10000, davidson_res_tol);
    logger->info("  Hamiltonian Eigenvalues:");
    for(int ii = 0; ii < nstates; ii++)
      logger->info("    --> Eval #{:4}: {:.6e}", ii, evals[ii]);
    E = evals[0];
    for(int ii = 0; ii < H.m(); ii++) C_local[ii] = evecs[ii];
  }

  MPI_Barrier(comm);
  auto dav_en = clock_type::now();

  logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
               niter, "E0", E, "DAVIDSON_DUR",
               duration_type(dav_en - dav_st).count());

  return E;
}
#endif

template <typename SpMatType>
double serial_selected_ci_diag(const SpMatType& H, size_t davidson_max_m,
                               double davidson_res_tol, std::vector<double>& C,
                               const size_t nstates = 1) {
  auto logger = spdlog::get("ci_solver");
  if(!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Resize eigenvector size
  C.resize(H.m(), 0);

  // Extract Diagonal
  auto D = extract_diagonal_elements(H);

  // Setup guess
  auto max_c = *std::max_element(C.begin(), C.end(), [](auto a, auto b) {
    return std::abs(a) < std::abs(b);
  });
  max_c = std::abs(max_c);

  if(max_c > (1. / C.size())) {
    logger->info("  * Will use passed vector as guess");
  } else {
    logger->info("  * Will generate identity guess");
    diagonal_guess(C.size(), H, C.data());
  }

  // Setup Davidson Functor
  SparseMatrixOperator op(H);

  // Solve EVP
  auto dav_st = clock_type::now();
  double E = 0.;
  size_t niter = 0;

  if(nstates == 1) {
    auto [niter0, E0] = davidson(H.m(), davidson_max_m, op, D.data(),
                                 davidson_res_tol, C.data());
    niter = niter0;
    E = E0;
  } else {
    std::vector<double> evals;
    std::vector<double> evecs;
    LobpcgGS(H.m(), nstates, op, evals, evecs, 10000, davidson_res_tol);
    logger->info("  Hamiltonian Eigenvalues:");
    for(int ii = 0; ii < nstates; ii++)
      logger->info("    --> Eval #{:4}: {:.6e}", ii, evals[ii]);
    E = evals[0];
    for(int ii = 0; ii < H.m(); ii++) C[ii] = evecs[ii];
  }

  auto dav_en = clock_type::now();

  logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
               niter, "E0", E, "DAVIDSON_DUR",
               duration_type(dav_en - dav_st).count());

  return E;
}

template <size_t N, typename index_t = int32_t>
double selected_ci_diag(wavefunction_iterator_t<N> dets_begin,
                        wavefunction_iterator_t<N> dets_end,
                        HamiltonianGenerator<N>& ham_gen, double h_el_tol,
                        size_t davidson_max_m, double davidson_res_tol,
                        std::vector<double>& C_local,
                        MACIS_MPI_CODE(MPI_Comm comm, )
                            const bool quiet = false,
                        const size_t nstates = 1) {
  auto logger = spdlog::get("ci_solver");
  if(!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  logger->info("[Selected CI Solver]:");
  logger->info("  {} = {:6}, {} = {:.5e}, {} = {:.5e}, {} = {:4}", "NDETS",
               std::distance(dets_begin, dets_end), "MATEL_TOL", h_el_tol,
               "RES_TOL", davidson_res_tol, "MAX_SUB", davidson_max_m);

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Generate Hamiltonian
  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto H_st = clock_type::now();

#ifdef MACIS_ENABLE_MPI
  auto H = make_dist_csr_hamiltonian<index_t>(comm, dets_begin, dets_end,
                                              ham_gen, h_el_tol);
#else
  auto H =
      make_csr_hamiltonian<index_t>(dets_begin, dets_end, ham_gen, h_el_tol);
#endif

  auto H_en = clock_type::now();
  MACIS_MPI_CODE(MPI_Barrier(comm);)

  // Get total NNZ
#ifdef MACIS_ENABLE_MPI
  size_t local_nnz = H.nnz();
  size_t total_nnz = allreduce(local_nnz, MPI_SUM, comm);
  size_t max_nnz = allreduce(local_nnz, MPI_MAX, comm);
  size_t min_nnz = allreduce(local_nnz, MPI_MIN, comm);
#else
  size_t total_nnz = H.nnz();
#endif

  logger->info("  {}   = {:6}, {}     = {:.5e} ms", "NNZ", total_nnz, "H_DUR",
               duration_type(H_en - H_st).count());

#ifdef MACIS_ENABLE_MPI
  auto world_size = comm_size(comm);
  if(world_size > 1) {
    double local_hdur = duration_type(H_en - H_st).count();
    double max_hdur = allreduce(local_hdur, MPI_MAX, comm);
    double min_hdur = allreduce(local_hdur, MPI_MIN, comm);
    double avg_hdur = allreduce(local_hdur, MPI_SUM, comm);
    avg_hdur /= world_size;
    logger->info(
        "  H_DUR_MAX = {:.2e} ms, H_DUR_MIN = {:.2e} ms, H_DUR_AVG = {:.2e} ms",
        max_hdur, min_hdur, avg_hdur);
  }
#endif
  logger->info("  {} = {:.2e} GiB", "HMEM_LOC",
               H.mem_footprint() / 1073741824.);
  logger->info("  {} = {:.2f}%", "H_SPARSE",
               total_nnz / double(H.n() * H.n()) * 100);
#ifdef MACIS_ENABLE_MPI
  if(world_size > 1) {
    logger->info("  NNZ_MAX = {}, NNZ_MIN = {}, NNZ_AVG = {}", max_nnz, min_nnz,
                 total_nnz / double(world_size));
  }
#endif

  // Solve EVP
#ifdef MACIS_ENABLE_MPI
  auto E = parallel_selected_ci_diag(H, davidson_max_m, davidson_res_tol,
                                     C_local, comm, nstates);
#else
  auto E = serial_selected_ci_diag(H, davidson_max_m, davidson_res_tol, C_local,
                                   nstates);
#endif

  return E;
}

}  // namespace macis
