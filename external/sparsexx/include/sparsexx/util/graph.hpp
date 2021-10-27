#pragma once

#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/submatrix.hpp>

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

inline auto perm_from_part( int64_t npart, const std::vector<idx_t>& part ) {

  std::vector<idx_t> perm( part.size() );
  std::vector<idx_t> partptr( npart+1 );
  std::iota( perm.begin(), perm.end(), 0 );

  partptr.at(0) = 0;
  auto cur_part = perm.begin();
  for( int64_t p = 0; p < npart-1; ++p ) {
    cur_part = std::stable_partition( cur_part, perm.end(),
      [&](auto i){ return part.at(i) == p; } );
    partptr.at(p+1) = std::distance( perm.begin(), cur_part );
  }
  partptr.at(npart) = perm.size();

  // Sanity check
  for( int64_t p = 0; p < npart; ++p ) {
    if( std::any_of( perm.begin() + partptr.at(p),
                     perm.begin() + partptr.at(p+1),
                     [&](auto i){ return part.at(i) != p; } ) )
      throw std::runtime_error("PERM NOT PARTITIONED");
  }

  return std::tuple( perm, partptr );

}


template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, SpMatType>
  permute_rows( const SpMatType& A, 
    const std::vector<idx_t>& perm ) {

  SpMatType Ap( A.m(), A.n(), A.nnz(), A.indexing() );

  const auto  m   = A.m();
  const auto indexing = A.indexing();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto* Anz = A.nzval().data();

  auto* Aprp = Ap.rowptr().data();
  auto* Apci = Ap.colind().data();
  auto* Apnz = Ap.nzval().data();

  Aprp[0] = Ap.indexing();

  for( int64_t i = 0; i < m; ++i ) {
    const auto A_j_st = Arp[perm[i]]   - indexing;
    const auto A_j_en = Arp[perm[i]+1] - indexing;
    const auto j_ext  = A_j_en - A_j_st;

    Aprp[i+1] = Aprp[i] + j_ext;
  }

  #pragma omp parallel for
  for( int64_t i = 0; i < m; ++i ) {
    const auto A_j_st  = Arp[perm[i]]   - indexing;
    const auto A_j_en  = Arp[perm[i]+1] - indexing;
    const auto Ap_j_st = Aprp[i]        - indexing;
    const auto Ap_j_en = Aprp[i+1]      - indexing;

    const auto j_ext  = A_j_en - A_j_st;

    const auto* Aci_st = Aci + A_j_st;
    const auto* Aci_en = Aci + A_j_en;
    const auto* Anz_st = Anz + A_j_st;
    const auto* Anz_en = Anz + A_j_en;

    auto* Apci_st = Apci + Ap_j_st;
    auto* Apnz_st = Apnz + Ap_j_st;
    std::copy( Aci_st, Aci_en, Apci_st );
    std::copy( Anz_st, Anz_en, Apnz_st );

  }

  return Ap;
}


template <typename IntT>
std::vector<IntT> invert_perm( const std::vector<IntT>& perm ) {

  const auto n = perm.size();
  std::vector<IntT> iperm( n );

  #pragma omp parallel for
  for( int64_t i = 0; i < n; ++i ) {
    const auto it = std::find( perm.begin(), perm.end(), i );
    if( it == perm.end() ) throw std::runtime_error("Something terrible happened");
    iperm[i] = std::distance( perm.begin(), it );
  }
    
  return iperm;

}



template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, SpMatType>
  permute_cols( const SpMatType& A, 
    const std::vector<idx_t>& perm ) {

  SpMatType Ap( A.m(), A.n(), A.nnz(), A.indexing() );

  const auto  m   = A.m();
  const auto indexing = A.indexing();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto* Anz = A.nzval().data();

  auto* Aprp = Ap.rowptr().data();
  auto* Apci = Ap.colind().data();
  auto* Apnz = Ap.nzval().data();

  auto iperm = invert_perm( perm );
  std::vector<std::vector<int64_t>> ind_th( omp_get_max_threads() );

  // Copy rowptr
  std::copy( A.rowptr().begin(), A.rowptr().end(), Ap.rowptr().begin() );

  #pragma omp parallel for
  for( int64_t i = 0; i < m; ++i ) {
    const auto A_j_st = Arp[i]   - indexing;
    const auto A_j_en = Arp[i+1] - indexing;
    const auto j_ext  = A_j_en - A_j_st;

    const auto* Aci_st = Aci + A_j_st;
    const auto* Anz_st = Anz + A_j_st;

    auto* Apci_st = Apci + A_j_st;
    auto* Apnz_st = Apnz + A_j_st;

    auto& ind = ind_th[omp_get_thread_num()];
    if( ind.size() != j_ext ) {
      ind.resize( j_ext );
      std::iota( ind.begin(), ind.end(), 0 );
    }

    std::sort( ind.begin(), ind.end(), 
      [&]( auto ii, auto jj ) {
        return iperm[Aci_st[ii]-indexing] < iperm[Aci_st[jj]-indexing];
      } );

    for( auto j = 0; j < j_ext; ++j ) {
      Apci_st[j] = iperm[ Aci_st[ind[j]] - indexing ] + indexing;
      Apnz_st[j] = Anz_st[ind[j]];
    }

  }

  return Ap;
}









template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, SpMatType>
  permute_rows_cols( const SpMatType& A, 
    const std::vector<idx_t>& rperm, 
    const std::vector<idx_t>& cperm ) {

  SpMatType Ap( A.m(), A.n(), A.nnz(), A.indexing() );

  const auto  m   = A.m();
  const auto indexing = A.indexing();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto* Anz = A.nzval().data();

  auto* Aprp = Ap.rowptr().data();
  auto* Apci = Ap.colind().data();
  auto* Apnz = Ap.nzval().data();

  auto icperm = invert_perm( cperm );
  std::vector<std::vector<int64_t>> ind_th( omp_get_max_threads() );

  Aprp[0] = Ap.indexing();
  for( int64_t i = 0; i < m; ++i ) {
    const auto A_j_st = Arp[rperm[i]]   - indexing;
    const auto A_j_en = Arp[rperm[i]+1] - indexing;
    const auto j_ext  = A_j_en - A_j_st;

    Aprp[i+1] = Aprp[i] + j_ext;

  }

  #pragma omp parallel for
  for( int64_t i = 0; i < m; ++i ) {

    const auto A_j_st = Arp[rperm[i]]   - indexing;
    const auto A_j_en = Arp[rperm[i]+1] - indexing;
    const auto Ap_j_st = Aprp[i]        - indexing;
    const auto Ap_j_en = Aprp[i+1]      - indexing;
    const auto j_ext  = A_j_en - A_j_st;



    const auto* Aci_st = Aci + A_j_st;
    const auto* Anz_st = Anz + A_j_st;

    auto* Apci_st = Apci + Ap_j_st;
    auto* Apnz_st = Apnz + Ap_j_st;

    auto& ind = ind_th[omp_get_thread_num()];
    if( ind.size() != j_ext ) {
      ind.resize( j_ext );
      std::iota( ind.begin(), ind.end(), 0 );
    }

    std::sort( ind.begin(), ind.end(), 
      [&]( auto ii, auto jj ) {
        return icperm[Aci_st[ii]-indexing] < icperm[Aci_st[jj]-indexing];
      } );

    for( auto j = 0; j < j_ext; ++j ) {
      Apci_st[j] = icperm[ Aci_st[ind[j]] - indexing ] + indexing;
      Apnz_st[j] = Anz_st[ind[j]];
    }

  }

  return Ap;
}

}
