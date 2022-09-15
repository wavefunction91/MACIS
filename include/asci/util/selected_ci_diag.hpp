#pragma once
#include <asci/types.hpp>
#include <asci/hamiltonian_generator.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/davidson.hpp>
#include <chrono>


namespace asci {

template <typename SpMatType>
void diagonal_guess( size_t N_local, const SpMatType& A, double* X ) {

  auto comm = A.comm();
  int world_rank, world_size;
  MPI_Comm_rank( comm, &world_rank );
  MPI_Comm_size( comm, &world_size );

  // Extract diagonal tile
  auto A_diagonal_tile = A.diagonal_tile_ptr();
  if( !A_diagonal_tile ) throw std::runtime_error("Diagonal Tile Not Populated");

  // Gather Diagonal
  auto D_local = extract_diagonal_elements( *A_diagonal_tile );

  std::vector<int> remote_counts(world_size), row_starts(world_size+1,0);
  for( auto i = 0; i < world_size; ++i ) {
    remote_counts[i] = A.row_extent(i);
    row_starts[i+1]  = row_starts[i] + A.row_extent(i);
  }

  std::vector<double> D(row_starts.back());

  MPI_Allgatherv( D_local.data(), D_local.size(), MPI_DOUBLE, D.data(),
    remote_counts.data(), row_starts.data(), MPI_DOUBLE, comm );

  // Determine min index
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance( D.begin(), D_min );

  // Zero out guess
  for(size_t i = 0; i < N_local; ++i ) X[i] = 0.;

  // Get owner rank
  int owner_rank = min_idx / remote_counts[0];
  if( world_rank == owner_rank ) {
    X[ min_idx - A.local_row_start() ] = 1.;
  }

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

  // Setup guess
  diagonal_guess(C_local.size(), H, C_local.data());

  // Setup Davidson Functor
  struct SpMatOp {
    const decltype(H)& matrix;
    sparsexx::spblas::spmv_info<index_t> spmv_info;

    SpMatOp() = delete;
    SpMatOp(decltype(matrix) m) : matrix(m), 
      spmv_info(sparsexx::spblas::generate_spmv_comm_info(m)) {}

    void operator_action( size_t m, double alpha, const double* V, size_t LDV,
      double beta, double* AV, size_t LDAV) const {
      sparsexx::spblas::pgespmv( alpha, matrix, V, beta, AV, spmv_info );
    }
  };
  SpMatOp op(H);
  auto D_local = extract_diagonal_elements( H.diagonal_tile() );

  // Solve EVP
  MPI_Barrier(comm);
  auto dav_st = clock_type::now();
  #if 1
  double E = p_davidson( H.local_row_extent(), davidson_max_m, op, 
    D_local.data(), davidson_res_tol, C_local.data(), H.comm() );
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
