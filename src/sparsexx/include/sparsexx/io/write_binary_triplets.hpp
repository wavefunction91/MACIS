#pragma once

#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <fstream>
#include <cassert>
#include <string>
#include <iostream>
#include <stdexcept>

namespace sparsexx {

template <typename SpMatType>
detail::enable_if_coo_matrix_t<SpMatType>
  write_binary_triplet( const SpMatType& A, std::string fname ) {

  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ofstream f_out( fname, std::ios::binary );
  size_t nnz = A.nnz();;
  index_t m = A.m(), n = A.n();
  f_out.write( (char*)&m,   sizeof(index_t) );
  f_out.write( (char*)&n,   sizeof(index_t) );
  f_out.write( (char*)&nnz, sizeof(size_t)  ); 

  f_out.write( (char*) A.rowind().data(), nnz * sizeof(index_t) );
  f_out.write( (char*) A.colind().data(), nnz * sizeof(index_t) );
  f_out.write( (char*) A.nzval().data(),  nnz * sizeof(value_t) );

}







template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType>
  write_binary_triplet( const SpMatType& A, std::string fname ) {

  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ofstream f_out( fname, std::ios::binary );
  size_t nnz = A.nnz();;
  index_t m = A.m(), n = A.n();
  f_out.write( (char*)&m,   sizeof(index_t) );
  f_out.write( (char*)&n,   sizeof(index_t) );
  f_out.write( (char*)&nnz, sizeof(size_t)  ); 

  // Construct rowind
  std::vector<index_t> rowind(nnz);
  auto rowind_it = rowind.begin();
  for( size_t i = 0; i < m; ++i ) {
    const auto row_count = A.rowptr()[i+1] - A.rowptr()[i];
    rowind_it = std::fill_n( rowind_it, row_count, i + A.indexing() );
  }
  assert( rowind_it == rowind.end() );

  f_out.write( (char*) rowind.data(),     nnz * sizeof(index_t) );
  f_out.write( (char*) A.colind().data(), nnz * sizeof(index_t) );
  f_out.write( (char*) A.nzval().data(),  nnz * sizeof(value_t) );

}

}
