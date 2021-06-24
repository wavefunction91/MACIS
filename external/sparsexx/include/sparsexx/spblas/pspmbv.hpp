#pragma once

#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>

namespace sparsexx::spblas {

template <typename SpMatType, typename ALPHAT, typename BETAT>
//std::enable_if_t< detail::spmbv_uses_generic_csr_v<SpMatType, ALPHAT, BETAT> >
void
  pgespmbv_grv( int64_t K, ALPHAT ALPHA, const dist_sparse_matrix<SpMatType>& A,
    const typename SpMatType::value_type* V,  int64_t LDV,  BETAT BETA,
          typename SpMatType::value_type* AV, int64_t LDAV ) {

  if( LDAV != A.global_m() ) throw std::runtime_error("AV cannot be a submatrix");

  // Scale AV
  #pragma omp parallel for collapse(2)
  for( int64_t k = 0; k < K;     ++k )
  for( int64_t i = 0; i < A.global_m(); ++i )
    AV[ i + k*LDAV ] *= BETA;

  // Loop over local tiles and compute local pieces of spbmv
  for( const auto& [tile_index, local_tile] : A ) {

    const auto& [row_st, row_en] = local_tile.global_row_extent;
    const auto& [col_st, col_en] = local_tile.global_col_extent;

    const auto spmv_m = row_en - row_st;
    const auto spmv_n = K;
    const auto spmv_k = col_en - col_st;

    const auto* V_local  = V  + col_st;
          auto* AV_local = AV + row_st;

    gespmbv( spmv_n, ALPHA, local_tile.local_matrix, V_local, LDV, 
      1., AV_local, LDAV );
      
  }



  auto comm = A.comm();
  MPI_Allreduce( MPI_IN_PLACE, AV, K*LDAV, 
    sparsexx::detail::mpi_data_t< sparsexx::detail::value_type_t<SpMatType> >,
    MPI_SUM, comm );

}

}
