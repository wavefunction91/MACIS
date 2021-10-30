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
struct block_row_dist_info {

  using index_type = IndexType;

  MPI_Comm comm;

  std::vector< std::vector<index_type> > send_indices;
  std::vector< std::vector<index_type> > recv_indices;

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


  block_row_dist_info<index_type> get_block_row_dist_info() {
  
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


    std::vector<size_type> recv_counts(comm_size_);
    std::transform( recv_indices.begin(), recv_indices.end(),
      recv_counts.begin(), [](const auto& idx) {return idx.size();} );

    std::vector<size_type> recv_counts_gathered( comm_size_ * comm_size_ );
    MPI_Allgather( recv_counts.data(), sizeof(size_type)*comm_size_, 
      MPI_BYTE, recv_counts_gathered.data(), sizeof(size_type)*comm_size_,
      MPI_BYTE, comm_ );

    if(0){
    std::stringstream ss;
    ss << "RANK " << comm_rank_ << " will send:" << std::endl;
    for( auto i = 0; i < comm_size_; ++i ) 
    if( i != comm_rank_ ) {
      auto nremote = recv_counts_gathered[ comm_rank_ + i*comm_size_ ];
      if(nremote) ss << "  " << nremote << " to RANK " << i << std::endl;
    }
    std::cout << ss.str() << std::endl;
    }

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

    MPI_Waitall( recv_send_indices_request.size(),
      recv_send_indices_request.data(), MPI_STATUSES_IGNORE );


    return block_row_dist_info<index_type>{comm_,send_indices,recv_indices};
  }


  const auto& diagonal_tile() const { return *diagonal_tile_; }
  const auto& off_diagonal_tile() const { return *off_diagonal_tile_; }
}; // class dist_sparse_matrix


}
