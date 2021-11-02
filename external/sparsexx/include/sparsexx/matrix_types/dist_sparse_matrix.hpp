#pragma once
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/mpi.hpp>
#include <sparsexx/util/submatrix.hpp>

#include <iostream>
#include <sstream>
#include <set>
#include <algorithm>

namespace sparsexx {

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


template <typename SpMatType>
class dist_sparse_matrix {

public:

  using value_type = detail::value_type_t<SpMatType>;
  using index_type = detail::index_type_t<SpMatType>;
  using size_type  = detail::size_type_t<SpMatType>;
  using tile_type  = SpMatType;

protected:

  MPI_Comm comm_;
  int comm_size_;
  int comm_rank_;

  size_type global_m_;
  size_type global_n_;

  std::shared_ptr<tile_type> diagonal_tile_     = nullptr;
  std::shared_ptr<tile_type> off_diagonal_tile_ = nullptr;

  std::vector< std::pair<index_type,index_type> > dist_row_extents_;

public:

  constexpr dist_sparse_matrix() noexcept = default;

  dist_sparse_matrix( MPI_Comm c, size_type M, size_type N ) :
    comm_(c), global_m_(M), global_n_(N) {

    comm_size_ = detail::get_mpi_size(comm_);
    comm_rank_ = detail::get_mpi_rank(comm_);
    const auto nrow_per_rank = M / comm_size_;

    dist_row_extents_.resize(comm_size_);
    dist_row_extents_[0] = { 0, nrow_per_rank };
    for( int i = 1; i < comm_size_; ++i ) {
      dist_row_extents_[i] = {
        dist_row_extents_[i-1].second,
        dist_row_extents_[i-1].second + nrow_per_rank
      };
    }
    dist_row_extents_.back().second += M % comm_size_; // Last rank gets carry-over
  
  }

  dist_sparse_matrix( MPI_Comm c, const SpMatType& A ) :
    dist_sparse_matrix( c, A.m(), A.n() ) {

    auto [local_row_st, local_row_en] = dist_row_extents_[comm_rank_];
    auto local_lo = std::make_pair<int64_t,int64_t>(local_row_st, local_row_st);
    auto local_up = std::make_pair<int64_t,int64_t>(local_row_en, local_row_en);
    diagonal_tile_ = std::make_shared<tile_type>(
      extract_submatrix( A, local_lo, local_up )
    );
    off_diagonal_tile_ = std::make_shared<tile_type>(
      extract_submatrix_inclrow_exclcol( A, local_lo, local_up )
    );
    diagonal_tile_->set_indexing(0);
    off_diagonal_tile_->set_indexing(0);
      

  }

  inline auto m() const { return global_m_; }
  inline auto n() const { return global_n_; }

  inline MPI_Comm comm() const { return comm_; }

  inline auto row_bounds(int rank) const {return dist_row_extents_[rank]; }

  inline size_type row_extent(int rank) const {
    return dist_row_extents_[rank].second -
           dist_row_extents_[rank].first; 
  }

  inline size_type local_row_extent() const { 
    return row_extent(comm_rank_);
  }

  inline size_type local_row_start() const {
    return dist_row_extents_[comm_rank_].first;
  }


  spmv_info<index_type> get_spmv_info() {
  
    // Get unique elements
    std::set<index_type> unique_elements_set;
    //if( diagonal_tile_ ) {
    //  assert( diagonal_tile_->indexing() == 0 );
    //  for( auto i : diagonal_tile_->colind() )
    //    unique_elements_set.insert( i + dist_row_extents_[comm_rank_].first );
    //}
    if( off_diagonal_tile_ ) {
      assert( off_diagonal_tile_->indexing() == 0 );
      unique_elements_set.insert( off_diagonal_tile_->colind().begin(),
                                  off_diagonal_tile_->colind().end() );
    }

    // Place in contiguous memory
    std::vector<index_type> unique_elements( 
      unique_elements_set.begin(), unique_elements_set.end()
    );

    //std::vector<size_type> recv_counts( comm_size_ );
    std::vector< std::vector<index_type> > recv_indices( comm_size_ );
    auto uniq_it = unique_elements.begin();
    for( int i = 0; i < comm_size_; ++i ) {
      // Get row extent for this rank
      auto [row_st, row_en] = dist_row_extents_[i];

      // Find upper bound for row end
      auto next_uniq_it = std::lower_bound( uniq_it, unique_elements.end(), 
        row_en );

      //recv_counts[i] = std::distance( uniq_it, next_uniq_it );
      recv_indices[i] = std::vector<index_type>( uniq_it, next_uniq_it );
      uniq_it = next_uniq_it;
    }


    std::vector<size_t> recv_counts(comm_size_);
    std::transform( recv_indices.begin(), recv_indices.end(),
      recv_counts.begin(), [](const auto& idx) {return idx.size();} );

    std::vector<size_t> recv_counts_gathered( comm_size_ * comm_size_ );
    MPI_Allgather( recv_counts.data(), sizeof(size_t)*comm_size_, 
      MPI_BYTE, recv_counts_gathered.data(), sizeof(size_t)*comm_size_,
      MPI_BYTE, comm_ );


    // Allocate and post recvs
    std::vector< std::vector<index_type> > send_indices( comm_size_ );
    std::vector< MPI_Request > recv_send_indices_request;
    for( auto i = 0; i < comm_size_; ++i )
    if( i != comm_rank_ ) {
      auto nremote = recv_counts_gathered[ comm_rank_ + i*comm_size_ ];
      if( nremote ) {
        send_indices[i].resize( nremote );
        auto& req = recv_send_indices_request.emplace_back();
        MPI_Irecv( send_indices[i].data(), nremote * sizeof(index_type),
          MPI_BYTE, i, 0, comm_, &req );
      }
    }

    // Post sends
    for( auto i = 0; i < comm_size_; ++i ) 
    if( i != comm_rank_ and recv_counts[i] ) {
      MPI_Request req;
      MPI_Isend( recv_indices[i].data(), recv_counts[i]*sizeof(index_type),
        MPI_BYTE, i, 0, comm_, &req );
      MPI_Request_free(&req);
    }

    // Compute recv offsets
    std::vector< size_t > recv_offsets( comm_size_, 0 );
    for( size_t i = 0, ioff = 0; i < comm_size_; ++i )
    if( i != comm_rank_ ) {
      recv_offsets[i] = ioff;
      ioff += recv_counts[i];
    }

    MPI_Waitall( recv_send_indices_request.size(),
      recv_send_indices_request.data(), MPI_STATUSES_IGNORE );

    // Compute send offsets / counts
    std::vector<size_t> send_counts( comm_size_ );
    std::transform( send_indices.begin(), send_indices.end(),
      send_counts.begin(), [](const auto& idx) {return idx.size();} );

    std::vector< size_t > send_offsets( comm_size_, 0 );
    for( size_t i = 0, ioff = 0; i < comm_size_; ++i )
    if( i != comm_rank_ ) {
      send_offsets[i] = ioff;
      ioff += send_counts[i];
    }

    std::vector<index_type> recv_indices_linear, send_indices_linear;
    for( size_t i = 0, ioff = 0; i < comm_size_; ++i )
    if( i != comm_rank_ ) {
      recv_indices_linear.insert( recv_indices_linear.end(),
                                  recv_indices[i].begin(),
                                  recv_indices[i].end() );
      send_indices_linear.insert( send_indices_linear.end(),
                                  send_indices[i].begin(),
                                  send_indices[i].end() );
    }

    spmv_info<index_type> info;
    info.comm = comm_;
    info.send_indices = std::move(send_indices_linear);
    info.recv_indices = std::move(recv_indices_linear);
    info.recv_offsets = std::move(recv_offsets);
    info.recv_counts  = std::move(recv_counts);
    info.send_offsets = std::move(send_offsets);
    info.send_counts  = std::move(send_counts);

    return info;
  }

  auto       diagonal_tile_ptr()       { return diagonal_tile_; }
  const auto diagonal_tile_ptr() const { return diagonal_tile_; }
  auto       off_diagonal_tile_ptr()       { return off_diagonal_tile_; }
  const auto off_diagonal_tile_ptr() const { return off_diagonal_tile_; }

  const auto& diagonal_tile() const { return *diagonal_tile_; }
  const auto& off_diagonal_tile() const { return *off_diagonal_tile_; }
}; // class dist_sparse_matrix


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


}
