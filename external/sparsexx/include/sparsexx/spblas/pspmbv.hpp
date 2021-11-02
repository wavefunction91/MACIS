#pragma once

#include <sparsexx/spblas/type_traits.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>

namespace sparsexx::spblas {

template <typename IndexType>
struct spmv_info {

  using index_type = IndexType;

  MPI_Comm comm;

  //std::vector< std::vector<index_type> > send_indices;
  //std::vector< std::vector<index_type> > recv_indices;

  std::vector< index_type > send_indices;
  std::vector< index_type > recv_indices;
  std::vector< size_t > send_offsets;
  std::vector< size_t > recv_offsets;
  std::vector< size_t > send_counts;
  std::vector< size_t > recv_counts;

};

template <typename DistSpMatrixType>
auto generate_spmv_comm_info( const DistSpMatrixType& A ) {

  using index_type = detail::index_type_t<DistSpMatrixType>;
  auto comm = A.comm();
  auto comm_size = detail::get_mpi_size(comm);
  auto comm_rank = detail::get_mpi_rank(comm);

  // Get unique column indices for local rows of A
  // excluding locally owned elements (i.e. off-diagonal col indices)
  auto off_diagonal_tile = A.off_diagonal_tile_ptr();
  std::set<index_type> unique_elements_set;
  if( off_diagonal_tile ) {
    assert( off_diagonal_tile->indexing() == 0 );
    unique_elements_set.insert( off_diagonal_tile->colind().begin(),
                                off_diagonal_tile->colind().end() );
  }

  // Place unique col indices into contiguous memory
  std::vector<index_type> unique_elements( 
    unique_elements_set.begin(), unique_elements_set.end()
  );

  // Generate a list of elements that need to be sent by remote
  // MPI ranks to the current processs
  std::vector< std::vector<index_type> > recv_indices_by_rank( comm_size );
  auto uniq_it = unique_elements.begin();
  for( int i = 0; i < comm_size; ++i ) {
    // Get row extent for rank-i
    auto [row_st, row_en] = A.row_bounds(i);

    // Find upper bound for row end
    auto next_uniq_it = std::lower_bound( uniq_it, unique_elements.end(), 
      row_en );

    // Copy indices into local memory
    recv_indices_by_rank[i] = std::vector<index_type>( uniq_it, next_uniq_it );
    uniq_it = next_uniq_it; // Update iterators
  }

  // Calculate element counts that will be received from remote processes
  std::vector<size_t> recv_counts(comm_size);
  std::transform( recv_indices_by_rank.begin(), recv_indices_by_rank.end(),
    recv_counts.begin(), [](const auto& idx) {return idx.size();} );

  // Ensure that local recv counts are zero
  assert( recv_counts[comm_rank] == 0 );

  // Gather recv counts to remote ranks
  // This tells each rank the number of elements each remote process
  // expects to receive from the local process
  std::vector<size_t> recv_counts_gathered( comm_size * comm_size );
  MPI_Allgather( recv_counts.data(), sizeof(size_t)*comm_size, 
    MPI_BYTE, recv_counts_gathered.data(), sizeof(size_t)*comm_size,
    MPI_BYTE, comm );

  
  // Allocate memory to store the remote indices each remote process
  // expects to receive from the current process and post async
  // receives to receive those indices
  std::vector< std::vector<index_type> > send_indices_by_rank( comm_size );
  std::vector< MPI_Request > recv_reqs;
  for( int i = 0; i < comm_size; ++i ) 
  if( i != comm_rank ) {
    auto nremote = recv_counts_gathered[ comm_rank + i*comm_size ];
    if( nremote ) {
      send_indices_by_rank[i].resize( nremote );
      auto& req = recv_reqs.emplace_back();
      MPI_Irecv( send_indices_by_rank[i].data(), nremote * sizeof(index_type),
        MPI_BYTE, i, 0, comm, &req );
    }
  }

  // Send to each remote process the indices it needs to send to the
  // current process
  for( auto i = 0; i < comm_size; ++i ) 
  if( recv_counts[i] ) {
    MPI_Request req;
    MPI_Isend( recv_indices_by_rank[i].data(), recv_counts[i]*sizeof(index_type),
      MPI_BYTE, i, 0, comm, &req );
    MPI_Request_free(&req);
  }

  // Wait on receives to complete
  MPI_Waitall( recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE );


  // Calculate element counts that will be sent to each remote processes
  std::vector<size_t> send_counts(comm_size);
  std::transform( send_indices_by_rank.begin(), send_indices_by_rank.end(),
    send_counts.begin(), [](const auto& idx) {return idx.size();} );



  // Compute offsets for send/recv indices in contiguous memory
  std::vector<size_t> recv_offsets(comm_size), send_offsets(comm_size);
  std::exclusive_scan( send_counts.begin(), send_counts.end(), 
    send_offsets.begin(), 0 );
  std::exclusive_scan( recv_counts.begin(), recv_counts.end(), 
    recv_offsets.begin(), 0 );

  // Linearize the send/recv data structures
  size_t nrecv_indices = 
    std::accumulate( recv_counts.begin(), recv_counts.end(), 0ul );
  size_t nsend_indices = 
    std::accumulate( send_counts.begin(), send_counts.end(), 0ul );
  std::vector<index_type> send_indices, recv_indices;
  send_indices.reserve( nsend_indices );
  recv_indices.reserve( nrecv_indices );
  for( auto i = 0; i < comm_size; ++i ) {
    send_indices.insert( send_indices.end(), 
      send_indices_by_rank[i].begin(),
      send_indices_by_rank[i].end() );
    recv_indices.insert( recv_indices.end(), 
      recv_indices_by_rank[i].begin(),
      recv_indices_by_rank[i].end() );
  }

    spmv_info<index_type> info;
    info.comm = comm;
    info.send_indices = std::move(send_indices);
    info.recv_indices = std::move(recv_indices);
    info.recv_offsets = std::move(recv_offsets);
    info.recv_counts  = std::move(recv_counts);
    info.send_offsets = std::move(send_offsets);
    info.send_counts  = std::move(send_counts);

    return info;


}

namespace detail {
  using namespace sparsexx::detail;
}

template <typename DistSpMatType, 
          typename ScalarType = detail::value_type_t<DistSpMatType>,
          typename IndexType  = detail::index_type_t<DistSpMatType>>
void pgespmv( detail::type_identity_t<ScalarType> ALPHA, const DistSpMatType& A,
              const detail::type_identity_t<ScalarType>* V,
              detail::type_identity_t<ScalarType> BETA,
              detail::type_identity_t<ScalarType>* AV,
              const spmv_info<detail::type_identity_t<IndexType>>& spmv_info) {

  using value_type = ScalarType;
  using index_type = IndexType;

  const auto M = A.m();
  const auto N = A.n();

  auto comm = A.comm();
  auto comm_size = detail::get_mpi_size(comm);
  auto comm_rank = detail::get_mpi_rank(comm);

  const auto& recv_counts  = spmv_info.recv_counts;
  const auto& recv_offsets = spmv_info.recv_offsets;
  const auto& recv_indices = spmv_info.recv_indices;

  const auto& send_counts  = spmv_info.send_counts;
  const auto& send_offsets = spmv_info.send_offsets;
  const auto& send_indices = spmv_info.send_indices;



  /***** Initial Communication Part *****/

  // Allocate buffer to receive packed data from remote processes
  size_t nrecv_pack = recv_indices.size();
  std::vector<value_type> V_recv_pack(nrecv_pack);

  // Post async receives
  std::vector<MPI_Request> recv_reqs;
  for( auto i = 0; i < comm_size; ++i ) 
  if( recv_counts[i] ) {
    recv_reqs.emplace_back();
    MPI_Irecv( V_recv_pack.data() + recv_offsets[i],
               recv_counts[i] * sizeof(value_type), MPI_BYTE, i, 0,
               comm, &recv_reqs.back() );
  }


  // Pack data to send to remote processes
  size_t nsend_pack = send_indices.size();
  std::vector<value_type> V_send_pack(nsend_pack);
  for( size_t i = 0; i < nsend_pack; ++i ) {
    V_send_pack[i] = V[ send_indices[i] - A.local_row_start() ];
  }

  // Send data to remote processes (fire and forget)
  std::vector<MPI_Request> send_reqs;
  for( auto i = 0; i < comm_size; ++i ) 
  if( send_counts[i] ) {
    send_reqs.emplace_back();
    MPI_Isend( V_send_pack.data() + send_offsets[i],
               send_counts[i] * sizeof(value_type), MPI_BYTE, i, 0,
               comm, &send_reqs.back() );
  }



  /***** Diagonal Matvec *****/
  gespmbv( 1, ALPHA, A.diagonal_tile(), V, N, BETA, AV, N );

  // Allocate memory for unpacked remote data
  std::vector<value_type> V_remote( N );

  // Wait for receives to complete 
  MPI_Waitall( recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE );

  // Unpack data into contiguous buffer 
  for( size_t i = 0; i < nrecv_pack; ++i ) {
    V_remote[recv_indices[i]] = V_recv_pack[i];
  }


  /***** Off-diagonal Matvec *****/
  gespmbv( 1, ALPHA, A.off_diagonal_tile(), V_remote.data(), N, 1., AV, N );

  // Wait for all sends to complete to keep packed buffer in scope
  MPI_Waitall( send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE );

}

}
