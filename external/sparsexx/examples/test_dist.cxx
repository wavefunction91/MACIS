#include <sparsexx/util/mpi.hpp>
#include <sparsexx/util/graph.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/io/read_mm.hpp>

#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;


int main(int argc, char** argv) {
  MPI_Init( &argc, &argv );
  auto world_size = sparsexx::detail::get_mpi_size( MPI_COMM_WORLD );
  auto world_rank = sparsexx::detail::get_mpi_rank( MPI_COMM_WORLD );
  {
  assert( argc == 2 );
  using spmat_type = sparsexx::csr_matrix<double, int32_t>;
  #if 1
  auto A = sparsexx::read_mm<spmat_type>( std::string( argv[1] ) );
  #else
  // [ x x 0 0 x x 0 0 x ]
  // [ x x 0 0 x x 0 0 x ]
  // [ 0 0 x x 0 0 x x x ]
  // [ 0 0 x x 0 0 x x x ]
  // [ x x 0 0 x x 0 0 0 ]
  // [ x x 0 0 x x 0 0 0 ]
  // [ 0 0 x x 0 0 x x 0 ]
  // [ 0 0 x x 0 0 x x 0 ]
  // [ x x x x 0 0 0 0 x ]
  spmat_type A(9,9,41,0);
  A.rowptr() = { 0, 5, 10, 15, 20, 24, 28, 32, 36, 41 };
  A.colind() = {
    0, 1, 4, 5, 8,
    0, 1, 4, 5, 8,
    2, 3, 6, 7, 8,
    2, 3, 6, 7, 8,
    0, 1, 4, 5,
    0, 1, 4, 5,
    2, 3, 6, 7,
    2, 3, 6, 7,
    1, 2, 3, 4, 8 
  };

  //std::fill(A.nzval().begin(), A.nzval().end(), 1. );
  std::iota(A.nzval().begin(), A.nzval().end(), 1. );
  /*
  A.nzval() = {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9, 9 
  };
  */
  #endif
  const int N = A.m();

  const bool do_reorder = true;
  std::vector<int32_t> mat_perm;
  if( do_reorder ) {

    int nparts = std::max(2l,world_size);
    // Reorder on root rank
    if( world_rank == 0 ) {
      // Partition the graph of A (assumes symmetric)
      auto kway_part_begin = clock_type::now(); 
      auto part = sparsexx::kway_partition( nparts, A );
      auto kway_part_end = clock_type::now(); 

      // Form permutation from partition
      std::tie( mat_perm, std::ignore)  = sparsexx::perm_from_part( nparts, part );

      // Permute rows/cols of A 
      // A(I,P(J)) = A(P(I),J)
      auto permute_begin = clock_type::now(); 
      A = sparsexx::permute_rows_cols( A, mat_perm, mat_perm );
      auto permute_end = clock_type::now(); 

      std::chrono::duration<double, std::milli> kway_part_dur = 
        kway_part_end - kway_part_begin;
      std::chrono::duration<double, std::milli> permute_dur = 
        permute_end - permute_begin;

      std::cout << "KWAY PART DUR = " << kway_part_dur.count() << std::endl;
      std::cout << "PERMUTE DUR   = " << permute_dur.count() << std::endl;
    } else {
      mat_perm.resize(N);
    }

    // Broadcast reordered data
    if( world_size > 1 ) {
      using index_type = sparsexx::detail::index_type_t<spmat_type>;
      using value_type = sparsexx::detail::value_type_t<spmat_type>;
      MPI_Bcast( A.rowptr().data(), A.rowptr().size() * sizeof(index_type),
                 MPI_BYTE, 0, MPI_COMM_WORLD );
      MPI_Bcast( A.colind().data(), A.colind().size() * sizeof(index_type),
                 MPI_BYTE, 0, MPI_COMM_WORLD );
      MPI_Bcast( A.nzval().data(), A.nzval().size() * sizeof(value_type),
                 MPI_BYTE, 0, MPI_COMM_WORLD );

      MPI_Bcast( mat_perm.data(), mat_perm.size() * sizeof(int32_t),
                 MPI_BYTE, 0, MPI_COMM_WORLD );
    }
  }

  // Get distributed matrix 
  sparsexx::dist_sparse_matrix<spmat_type> A_dist( MPI_COMM_WORLD, A );
  auto spmv_info = sparsexx::spblas::generate_spmv_comm_info( A_dist );

  {
  std::stringstream ss;
  ss << "MAT DIST RANK " << world_rank 
     << ": DIAGONAL NNZ = " << (A_dist.diagonal_tile().nnz()) 
     << ": OFF-DIAGONAL NNZ = " << (A_dist.off_diagonal_tile().nnz())
     << ": SEND COUNT = " << (spmv_info.send_indices.size()) 
     << ": RECV COUNT = " << (spmv_info.recv_indices.size()) 
     << std::endl;

  //std::cout << ss.str();
  }

  {
  size_t local_comm_volume = spmv_info.send_indices.size() + 
                             spmv_info.recv_indices.size();
  size_t comm_volume = 0;
  MPI_Reduce( &local_comm_volume, &comm_volume, 1, MPI_UINT64_T, MPI_SUM, 
    0, MPI_COMM_WORLD );
  if( world_rank == 0 ) std::cout << "COMM VOLUME = " << comm_volume << std::endl;
  }

  


  // Serial SPMV
  std::vector<double> V(N), AV(N);
  std::iota( V.begin(), V.end(), 0 );
  for( auto& x : V ) x *= 0.01;
  if( mat_perm.size() ) {
    sparsexx::permute_vector( N, V.data(), mat_perm.data(),
      sparsexx::PermuteDirection::Backward );
  }

  sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N );
  if( mat_perm.size() ) {
    sparsexx::permute_vector( N, AV.data(), mat_perm.data(), 
      sparsexx::PermuteDirection::Forward );
  }

  // Parallel SPMV
  std::vector<double> V_dist ( A_dist.local_row_extent() ),
                      AV_dist( A_dist.local_row_extent() );
  auto [dist_row_st, dist_row_en] = A_dist.row_bounds( world_rank );
  for( auto i = dist_row_st; i < dist_row_en; ++i ) {
    V_dist[i-dist_row_st] = V[i];
  }


  sparsexx::spblas::pgespmv( 1., A_dist, V_dist.data(), 0., AV_dist.data(), 
    spmv_info );




  // Compare results
  std::vector<double> AV_dist_combine(N);
  size_t n_per_rank = N / world_size;
  MPI_Allgather( AV_dist.data(), n_per_rank, MPI_DOUBLE, AV_dist_combine.data(),
    n_per_rank, MPI_DOUBLE, MPI_COMM_WORLD );
  if( N % world_size and world_size > 1 ) {
    if( world_rank == (world_size-1) ) {
      std::copy_n( AV_dist.data() + n_per_rank, N % world_size, 
        AV_dist_combine.data() + world_size*n_per_rank );
    }
    MPI_Bcast( AV_dist_combine.data() + world_size*n_per_rank, N%world_size,
      MPI_DOUBLE, world_size-1, MPI_COMM_WORLD );
  }
  if( mat_perm.size() ) {
    sparsexx::permute_vector( N, AV_dist_combine.data(), mat_perm.data(), 
      sparsexx::PermuteDirection::Forward );
  }

  double max_diff = 0.;
  for( auto i = 0; i < N; ++i ) {
    max_diff = std::max( max_diff, std::abs(AV_dist_combine[i] - AV[i]) );
  }
  //std::cout << "MAX DIFF = " << max_diff << std::endl;

  if( !world_rank ) {
  //for( auto i = 0; i < N; ++i ) {
  //  std::cout << AV[i] << std::endl;
  //}
  }
  

  }
  MPI_Finalize();
}
