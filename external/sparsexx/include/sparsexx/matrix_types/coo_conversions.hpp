#pragma once

#include <cassert>
#include "csr_matrix.hpp"
#include "coo_matrix.hpp"

namespace sparsexx {

template <typename T, typename index_t, typename Alloc>
coo_matrix<T,index_t,Alloc>::coo_matrix( const csr_matrix<T,index_t,Alloc>& other ) :
  coo_matrix( other.m(), other.n(), other.nnz(), other.indexing() ) {

  auto rowind_it = rowind_.begin();
  for( size_t i = 0; i < m_; ++i ) {
    const auto row_count = other.rowptr()[i+1] - other.rowptr()[i];
    rowind_it = std::fill_n( rowind_it, row_count, i + indexing_ );
  }
  assert( rowind_it == rowind_.end() );

  std::copy( other.colind().begin(), other.colind().end(), colind_.begin() );
  std::copy( other.nzval().begin(),  other.nzval().end(),  nzval_.begin()  );

}

}
