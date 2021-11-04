
#pragma once
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include <sparsexx/matrix_types/csr_matrix.hpp>

using namespace std;
using namespace cmz::ed;

template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const SetSlaterDets& stts,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh
) {


  // Form CSR adjacency
  std::vector<index_t> colind, rowptr;
  std::vector<double>  nzval;

  const auto ndets = stts.size();

  std::vector<slater_det> stts_vec( stts.begin(), stts.end() );
  std::vector<uint64_t> stts_states(ndets);
  std::transform( stts_vec.begin(), stts_vec.end(), stts_states.begin(),
    [](const auto& s){ return s.GetState(); } );

  std::vector< std::vector<index_t> > colind_by_row( ndets );
  std::vector< std::vector<double>  > nzval_by_row ( ndets );

  const double res_fraction = 0.20;
  for( auto& v : colind_by_row ) v.reserve( ndets * res_fraction );
  for( auto& v : nzval_by_row )  v.reserve( ndets * res_fraction );

  #pragma omp parallel for
  for( index_t i = 0; i < ndets; ++i ) {
    for( index_t j = 0; j < ndets; ++j ) 
    if( std::popcount( stts_states[i] ^ stts_states[j] ) <= 4 ) {
      const auto h_el = Hop.GetHmatel( stts_vec[i], stts_vec[j] );
      if( std::abs(h_el) > H_thresh ) {
        colind_by_row[i].emplace_back(j);
        nzval_by_row[i].emplace_back( h_el );
      }
    }
  }

  std::vector<size_t> row_counts( ndets );
  std::transform( colind_by_row.begin(), colind_by_row.end(), row_counts.begin(),
    [](const auto& v){ return v.size(); } );
  const size_t _nnz = std::accumulate( row_counts.begin(), row_counts.end(), 0ul );

  rowptr.resize( ndets + 1 );
  std::exclusive_scan( row_counts.begin(), row_counts.end(), rowptr.begin(), 0);
  rowptr[ndets] = rowptr[ndets-1] + row_counts[ndets-1];

  colind.reserve( _nnz );
  nzval .reserve( _nnz );
  for( auto& v : colind_by_row ) colind.insert(colind.end(), v.begin(), v.end());
  for( auto& v : nzval_by_row )  nzval .insert(nzval.end(),  v.begin(), v.end());

  // Move resources into CSR matrix
  const auto nnz = colind.size();
  sparsexx::csr_matrix<double, index_t> H( ndets, ndets, nnz, 0 );
  H.colind() = std::move(colind);
  H.rowptr() = std::move(rowptr);
  H.nzval()  = std::move(nzval);

  return H;

}

