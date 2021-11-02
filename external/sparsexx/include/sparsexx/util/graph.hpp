#pragma once

#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/submatrix.hpp>
#include <sparsexx/util/permute.hpp>

#include <metis.h>
#include <algorithm>
#include <numeric>
#include <omp.h>

namespace sparsexx {

template <typename SpMatType>
detail::enable_if_csr_matrix_t< SpMatType,
  std::tuple<
    std::vector< detail::index_type_t<SpMatType> >,
    std::vector< detail::index_type_t<SpMatType> >
  >> extract_adjacency( const SpMatType& A, int ret_indexing = 0 ) {

  const auto M = A.m();
  //const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  using index_t = detail::index_type_t<SpMatType>;
  std::vector< index_t > rowptr(M+1), colind( A.nnz() );

  rowptr[0] = ret_indexing;
  auto* ci_st = colind.data();

  // Get number of diagonal elements
  int64_t ndiag = 0;
  for( index_t i = 0; i < M; ++i ) {

    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
  
    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;

    const auto diag_it = std::find( Aci_st, Aci_en, i+indexing );

    // Copy values to left of diagonal
    ci_st = std::copy( Aci_st, diag_it, ci_st );


    auto nrow = std::distance( Aci_st, Aci_en );
    if( diag_it != Aci_en ) {
      ndiag++;
      rowptr[i+1] = rowptr[i] + nrow - 1;

      // Copy values right of diagonal
      ci_st = std::copy(diag_it+1, Aci_en, ci_st );
    } else rowptr[i+1] = rowptr[i] + nrow;

  }

  colind.resize( A.nnz() - ndiag );
  for( auto& i : colind ) i -= (indexing - ret_indexing);

  return std::tuple( rowptr, colind );


}

template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, std::vector<idx_t>>
  kway_partition( int64_t npart, const SpMatType& A ) {

  auto [adj_rp, adj_ci] = extract_adjacency( A, 0 );

  if( npart < 2 )
    throw std::runtime_error("KWayPart only works for K > 1");

  std::vector<idx_t> part(A.m());

  idx_t nweights = 1;
  idx_t nvert    = A.m();
  idx_t nparts   = npart;
  idx_t obj;

  METIS_PartGraphKway( &nvert, &nweights, adj_rp.data(), 
    adj_ci.data(), NULL, NULL, NULL, &nparts, NULL, NULL,
    NULL, &obj, part.data() );

  return part;

}






}
