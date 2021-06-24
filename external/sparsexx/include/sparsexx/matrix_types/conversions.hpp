#pragma once

#include "csr_matrix.hpp"
#include "coo_matrix.hpp"
#include <stdexcept>

namespace sparsexx {

template <typename T, typename index_t, typename Alloc>
csr_matrix<T,index_t,Alloc>::csr_matrix( const coo_matrix<T,index_t,Alloc>& other ) :
  csr_matrix( other.m(), other.n(), other.nnz(), other.indexing() ) {

  if( not other.is_sorted_by_row_index() ) {
    throw 
      std::runtime_error("COO -> CSR Conversion Requires COO To Be Row Sorted");
  }

  const auto& rowind_coo = other.rowind();
  const auto& colind_coo = other.colind();
  const auto& nzval_coo  = other.nzval();

  // Compute rowptr
  rowptr_.at(0) = other.indexing();
  auto cur_row = 0;
  for( size_t i = 0; i < nnz_; ++i ) 
  if( rowind_coo[i] != (cur_row + indexing_) ) {
    cur_row++;
    rowptr_.at(cur_row) = i + indexing_;
  }
  rowptr_.at(m_) = nnz_ + indexing_;

  std::copy( colind_coo.begin(), colind_coo.end(), colind_.begin() );
  std::copy( nzval_coo.begin(), nzval_coo.end(), nzval_.begin() );
}


}
