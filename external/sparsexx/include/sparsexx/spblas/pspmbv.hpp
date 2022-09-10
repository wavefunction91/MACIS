#pragma once

#include <sparsexx/spblas/type_traits.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>

#include <sparsexx/util/permute.hpp>

namespace sparsexx::spblas {

namespace detail {
  using namespace sparsexx::detail;
}


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


  inline size_t communication_volume() {
    size_t local_comm_vol = (send_indices.size() + recv_indices.size())/2;
    return detail::mpi_allreduce( local_comm_vol, MPI_SUM, comm );
  }


  template <typename T>
  std::vector<MPI_Request> post_remote_recv( T* X ) const {
    std::vector<MPI_Request> reqs;
    int comm_size = recv_offsets.size();
    for( int i = 0; i < comm_size; ++i ) 
    if( recv_counts[i] ) {
      reqs.emplace_back( 
        detail::mpi_irecv( X + recv_offsets[i], recv_counts[i], i, 0, comm )
      );
    }
    return reqs;
  }

  template <typename T>
  std::vector<MPI_Request> post_remote_send( const T* X ) const {
    std::vector<MPI_Request> reqs;
    int comm_size = send_offsets.size();
    for( int i = 0; i < comm_size; ++i ) 
    if( send_counts[i] ) {
      reqs.emplace_back( 
        detail::mpi_isend( X + send_offsets[i], send_counts[i], i, 0, comm )
      );
    }
    return reqs;
  }

};

template <typename DistSpMatrixType>
auto generate_spmv_comm_info( const DistSpMatrixType& A ) {

  using index_type = sparsexx::detail::index_type_t<DistSpMatrixType>;
  auto comm = A.comm();
  auto comm_size = sparsexx::detail::get_mpi_size(comm);
  auto comm_rank = sparsexx::detail::get_mpi_rank(comm);

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
  auto recv_counts_gathered = detail::mpi_allgather( recv_counts, comm );

  
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
      recv_reqs.emplace_back( 
        detail::mpi_irecv( send_indices_by_rank[i], i, 0, comm )
      );
    }
  }

  // Send to each remote process the indices it needs to send to the
  // current process
  std::vector< MPI_Request > send_reqs;
  for( auto i = 0; i < comm_size; ++i ) 
  if( recv_counts[i] ) {
    send_reqs.emplace_back( 
      detail::mpi_isend( recv_indices_by_rank[i], i, 0, comm )
    );
  }

  // Wait on receives to complete
  detail::mpi_waitall_ignore_status( recv_reqs );


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

  // Correct send indices to be relative to local row start
  const auto lrs = A.local_row_start();
  for( auto& i : send_indices ) i -= lrs;

  // Wait for sends to complete to avoid race conditions
  detail::mpi_waitall_ignore_status( send_reqs );

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

template <typename DistSpMatType, 
          typename ScalarType = detail::value_type_t<DistSpMatType>,
          typename IndexType  = detail::index_type_t<DistSpMatType>>
void pgespmv( detail::type_identity_t<ScalarType> ALPHA, const DistSpMatType& A,
              const detail::type_identity_t<ScalarType>* V,
              detail::type_identity_t<ScalarType> BETA,
              detail::type_identity_t<ScalarType>* AV,
              const spmv_info<detail::type_identity_t<IndexType>>& spmv_info) {

  using value_type = ScalarType;
  //using index_type = IndexType;

  //const auto M = A.m();
  const auto N = A.n();

  //auto comm = A.comm();
  //auto comm_size = detail::get_mpi_size(comm);
  //auto comm_rank = detail::get_mpi_rank(comm);

  const auto& recv_indices = spmv_info.recv_indices;
  const auto& send_indices = spmv_info.send_indices;

  /***** Initial Communication Part *****/

  // Allocated packed buffers
  size_t nrecv_pack = recv_indices.size();
  size_t nsend_pack = send_indices.size();
  std::vector<value_type> V_recv_pack(nrecv_pack);
  std::vector<value_type> V_send_pack(nsend_pack);

  // Buffer for offdiagonal matvec
  std::vector<value_type> V_remote( N );

  // Post async recv's for remote data required for offdiagonal
  // matvec
  auto recv_reqs = spmv_info.post_remote_recv( V_recv_pack.data() );

  // Pack data to send to remote processes
  sparsexx::permute_vector( nsend_pack, V, send_indices.data(), 
    V_send_pack.data(), sparsexx::PermuteDirection::Backward );

  // Send data (async) to remote processes
  auto send_reqs = spmv_info.post_remote_send( V_send_pack.data() );



  /***** Diagonal Matvec *****/
  gespmbv( 1, ALPHA, A.diagonal_tile(), V, N, BETA, AV, N );

  // Wait for receives to complete 
  detail::mpi_waitall_ignore_status( recv_reqs );

  // Unpack data into contiguous buffer 
  sparsexx::permute_vector( nrecv_pack, V_recv_pack.data(), recv_indices.data(),
    V_remote.data(), sparsexx::PermuteDirection::Forward );


  /***** Off-diagonal Matvec *****/
  if( A.off_diagonal_tile_ptr() )
  gespmbv( 1, ALPHA, A.off_diagonal_tile(), V_remote.data(), N, 1., AV, N );

  // Wait for all sends to complete to keep packed buffer in scope
  detail::mpi_waitall_ignore_status( send_reqs );

}

}
