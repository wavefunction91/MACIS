#pragma once

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/util/string.hpp>
#include <fstream>
#include <cassert>
#include <string>


namespace sparsexx {


template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
csr_matrix<T,index_t,Alloc> read_rb( std::string fname ) {

  std::ifstream f_in(fname);

  std::string line;

  // Skip the first two lines
  std::getline( f_in, line ); // comments
  std::getline( f_in, line ); // some strange metadata

  // Get useful meta data
  int64_t m, n, nnz;
  bool is_sym = false;
  {
    std::getline( f_in, line );
    auto tokens = tokenize( line );
    assert( tokens.size() == 5 );

    auto type = tokens[0];
    m   = std::stoll( tokens[1] );
    n   = std::stoll( tokens[2] );
    nnz = std::stoll( tokens[3] );
    // Whats the last one for??

    is_sym = type[1] == 's' or type[1] == 'S';
    assert( nnz <= m*n );
  }

  // Skip format line
  std::getline( f_in, line );

  csr_matrix<T,index_t,Alloc> A( m, n, nnz );

  int64_t curcount = 0;
  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    for( const auto& t : tokens )
      A.rowptr()[curcount++] = std::stoi(t);

    if( curcount == (n+1) ) break;

  }

  curcount = 0;
  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    for( const auto& t : tokens )
      A.colind()[curcount++] = std::stoi(t);

    if( curcount == nnz ) break;

  }

  curcount = 0;
  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    for( const auto& t : tokens )
      A.nzval()[curcount++] = std::stod(t);

    if( curcount == nnz ) break;

  }

  assert( !std::getline(f_in,line) );
  return A;

}

}
