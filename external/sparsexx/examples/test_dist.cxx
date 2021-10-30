#include <sparsexx/util/mpi.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/io/read_mm.hpp>

#include <sparsexx/spblas/spmbv.hpp>

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
  #endif
  const int N = A.m();

  sparsexx::dist_sparse_matrix<spmat_type> A_dist( MPI_COMM_WORLD, A );
  auto spmv_info = A_dist.get_block_row_dist_info();

  // Serial SPMV
  std::vector<double> V(N), AV(N);
  std::iota( V.begin(), V.end(), 0 );
  for( auto& x : V ) x *= 0.01;

  sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N );

  // Parallel SPMV
  std::vector<double> V_dist ( A_dist.local_row_extent() ),
                      AV_dist( A_dist.local_row_extent() );
  std::iota( V_dist.begin(), V_dist.end(), A_dist.local_row_start() );
  for( auto& x : V_dist ) x *= 0.01;

  // **** Diagonal part ****
  sparsexx::spblas::gespmbv(1, 1., A_dist.diagonal_tile(), V_dist.data(), N,
    0., AV_dist.data(), N );

  // **** Off diagonal part ****

  // Post recv for data needed
  std::vector<std::vector<double>> V_recv( world_size );
  std::vector<MPI_Request> recv_reqs;
  for( auto i = 0; i < world_size; ++i ) 
  if( i != world_rank ) {
    const auto& indices = spmv_info.recv_indices[i];
    auto pack_sz = indices.size();
    if(pack_sz) {
      recv_reqs.emplace_back();
      V_recv[i].resize( pack_sz );
      MPI_Irecv( V_recv[i].data(), pack_sz, MPI_DOUBLE, i, 0,
        MPI_COMM_WORLD, &recv_reqs.back() );
    }
  }

  // Pack data to send
  std::vector<std::vector<double>> V_pack( world_size );
  for( auto i = 0; i < world_size; ++i )
  if( i != world_rank ) {
    const auto& indices = spmv_info.send_indices[i];
    auto pack_sz = indices.size();
    if( pack_sz ) {
      V_pack[i].resize( pack_sz );
      for( auto j = 0; j < pack_sz; ++j ) {
        V_pack[i][j] = V_dist[ indices[j] - A_dist.local_row_start() ];
      }
      MPI_Request req;
      MPI_Isend( V_pack[i].data(), pack_sz, MPI_DOUBLE, i, 0,
        MPI_COMM_WORLD, &req );
      MPI_Request_free(&req);
    }
  }

  // Wait for recvs to be satisfied
  MPI_Waitall( recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE );

  // Unpack into long vector
  std::vector<double> V_offdiag(N);
  for( auto i = 0; i < world_size; ++i )
  if( i != world_rank ) {
    const auto& indices = spmv_info.recv_indices[i];
    auto pack_sz = indices.size();
    for( auto j = 0; j < pack_sz; ++j ) {
      V_offdiag[ indices[j] ] = V_recv[i][j];
    }
  }

  sparsexx::spblas::gespmbv(1, 1., A_dist.off_diagonal_tile(), V_offdiag.data(), N,
    1., AV_dist.data(), N );



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

  double max_diff = 0.;
  for( auto i = 0; i < N; ++i ) {
    max_diff = std::max( max_diff, std::abs(AV_dist_combine[i] - AV[i]) );
  }
  std::cout << "MAX DIFF = " << max_diff << std::endl;
  

  }
  MPI_Finalize();
}
