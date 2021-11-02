#pragma once

#include <sparsexx/spblas/type_traits.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>

namespace sparsexx::spblas {

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
