#pragma once
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>

namespace sparsexx {

template <typename SpMatType, 
  typename = detail::enable_if_csr_matrix_t<SpMatType>
> SpMatType extract_submatrix( const SpMatType& A, 
  std::pair<int64_t,int64_t> lo, std::pair<int64_t,int64_t> up) {

  const auto M = A.m();
  const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  const auto row_lo = lo.first;
  const auto row_up = up.first;
  const auto col_lo = lo.second;
  const auto col_up = up.second;
  const auto M_sub = row_up - row_lo;
  const auto N_sub = col_up - col_lo;

  // Determine NNZ in submatrix
  size_t nnz_sub = 0;
  std::vector<typename SpMatType::index_type> rowptr_sub( M_sub + 1 );
  rowptr_sub[0] = indexing;
  for( int64_t i = 0; i < M_sub; ++i ) {
    const auto j_st  = Arp[row_lo + i]   - indexing;
    const auto j_en  = Arp[row_lo + i+1] - indexing;

    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;

    auto nnz_row = std::count_if( Aci_st, Aci_en, 
      [&](const auto j){ 
        return (j-indexing) >= col_lo and (j-indexing) < col_up; 
      } );

    nnz_sub += nnz_row;
    rowptr_sub[i+1] = rowptr_sub[i] + nnz_row;
  }


  // Extract submatrix
  SpMatType sub( M_sub, N_sub, nnz_sub, indexing );
  sub.rowptr() = std::move( rowptr_sub );

  auto* sub_nz = sub.nzval().data();
  auto* sub_rp = sub.rowptr().data();
  auto* sub_ci = sub.colind().data();
  for( int64_t i = 0; i < M_sub; ++i ) {
    const auto Aj_st    = Arp[row_lo + i]   - indexing;
    const auto Aj_en    = Arp[row_lo + i+1] - indexing;
    const auto sub_j_st = sub_rp[i]         - indexing;

    const auto* Aci_st    = Aci    + Aj_st;
    const auto* Aci_en    = Aci    + Aj_en;
          auto* sub_ci_st = sub_ci + sub_j_st;

    const auto* Anz_st    = Anz    + Aj_st;
    const auto* Anz_en    = Anz    + Aj_en;
          auto* sub_nz_st = sub_nz + sub_j_st;

    const auto  Aci_sub_st_i = std::find_if( Aci_st, Aci_en,
      [&](const auto j){ return (j-indexing) >= col_lo; } );
    const auto  Aci_sub_en_r = std::find_if( 
      std::make_reverse_iterator(Aci_en), 
      std::make_reverse_iterator(Aci_sub_st_i),
      [&](const auto j){ return (j-indexing) < col_up; } );

    const auto ioff_st = std::distance( Aci_st, Aci_sub_st_i );
    const auto ioff_en = std::distance( Aci_sub_en_r, 
      std::make_reverse_iterator(Aci_st) );

    assert( ioff_en >= ioff_st );
    const auto* Aci_sub_st = Aci_st + ioff_st;
    const auto* Aci_sub_en = Aci_st + ioff_en;
    const auto* Anz_sub_st = Anz_st + ioff_st;
    const auto* Anz_sub_en = Anz_st + ioff_en;

    std::copy( Aci_sub_st, Aci_sub_en, sub_ci_st );
    std::copy( Anz_sub_st, Anz_sub_en, sub_nz_st );
  }

  for( auto& i : sub.colind() ) i -= col_lo; // Offset the columns bounds

  return sub;
}


template <typename SpMatType, 
  typename = detail::enable_if_csr_matrix_t<SpMatType>
> SpMatType extract_upper_triangle( const SpMatType& A ) {

  const auto M = A.m();
  const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  // Determine NNZ and row counts in the upper triangle
  size_t nnz_ut = 0;
  std::vector<typename SpMatType::index_type> rowptr_ut( M + 1 );
  rowptr_ut[0] = indexing;
  for( int64_t i = 0; i < M; ++i ) {
    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
  
    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;

    auto nnz_row = std::count_if( Aci_st, Aci_en, 
      [&](const auto j){ return (j-indexing) >= i; } );

    nnz_ut += nnz_row;
    rowptr_ut[i+1] = rowptr_ut[i] + nnz_row;
  }


  // Extract the Upper triangle
  SpMatType U( M, N, nnz_ut, indexing );
  //std::copy( rowptr_ut.begin(), rowptr_ut.end(), U.rowptr().begin() );
  U.rowptr() = std::move( rowptr_ut );
  auto* Unz = U.nzval().data();
  auto* Urp = U.rowptr().data();
  auto* Uci = U.colind().data();
  for( int64_t i = 0; i < M; ++i ) {
    const auto Aj_st  = Arp[i]   - indexing;
    const auto Aj_en  = Arp[i+1] - indexing;
    const auto Uj_st  = Urp[i]   - indexing;
  
    const auto* Aci_st = Aci + Aj_st;
    const auto* Aci_en = Aci + Aj_en;
          auto* Uci_st = Uci + Uj_st;

    const auto* Anz_st = Anz + Aj_st;
    const auto* Anz_en = Anz + Aj_en;
          auto* Unz_st = Unz + Uj_st;
  
    const auto* Aci_ut_st = std::find_if( Aci_st, Aci_en,
      [&](const auto x){ return (x-indexing) >= i; } );
    const auto ioff = std::distance( Aci_st, Aci_ut_st );
    const auto* Anz_ut_st = Anz_st + ioff;

    std::copy( Aci_ut_st, Aci_en, Uci_st );
    std::copy( Anz_ut_st, Anz_en, Unz_st );
  }

  return U;
}




template <typename SpMatType, 
  typename = detail::enable_if_csr_matrix_t<SpMatType>
> std::vector<typename SpMatType::value_type> 
  extract_diagonal_elements( const SpMatType& A ) {

  const auto M = A.m();
  //const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  std::vector<typename SpMatType::value_type> D( M );

  using index_t = typename SpMatType::index_type;
  for( index_t i = 0; i < M; ++i ) {

    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
  
    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;
    const auto* Anz_st = Anz + j_st;

    const auto diag_it = std::find( Aci_st, Aci_en, i+indexing );
    const auto diag_off = std::distance( Aci_st, diag_it );
    D[i] = Anz_st[diag_off];

  }

  return D;
}





template <typename SpMatType, 
  typename = detail::enable_if_csr_matrix_t<SpMatType>
> typename SpMatType::value_type trace( const SpMatType& A ) {

  const auto M = A.m();
  //const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  typename SpMatType::value_type tr = 0.;

  using index_t = typename SpMatType::index_type;
  for( index_t i = 0; i < M; ++i ) {

    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
  
    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;
    const auto* Anz_st = Anz + j_st;

    const auto diag_it = std::find( Aci_st, Aci_en, i+indexing );
    if( diag_it != Aci_en ) {
      const auto diag_off = std::distance( Aci_st, diag_it );
      tr += Anz_st[ diag_off ];
    }

  }

  return tr;
}






template <typename SpMatType>
detail::enable_if_csr_matrix_t< SpMatType,
  std::tuple<
    std::vector< detail::index_type_t<SpMatType> >,
    std::vector< detail::index_type_t<SpMatType> >
  >> extract_adjacency_base0( const SpMatType& A ) {

  const auto M = A.m();
  //const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  using index_t = detail::index_type_t<SpMatType>;
  std::vector< index_t > rowptr(M+1), colind( A.nnz() );

  rowptr[0] = 0;
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
  for( auto& i : colind ) i -= indexing;

  return std::tuple( rowptr, colind );


}





}
