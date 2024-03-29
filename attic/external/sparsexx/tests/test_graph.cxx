#include "catch2/catch.hpp"
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/io/read_mm.hpp>

#include <sparsexx/util/graph.hpp>

TEST_CASE("Adjacency", "[graph]") {

  using mat_type = sparsexx::csr_matrix<double,int32_t>;
  std::string ref_fname = SPARSEXX_DATA_DIR "/mycielskian3/mycielskian3.mtx";
  auto A = sparsexx::read_mm<mat_type>( ref_fname );

  auto [adj_rowptr, adj_colind] = sparsexx::extract_adjacency( A, 1 );
  REQUIRE( adj_rowptr == A.rowptr() );
  REQUIRE( adj_colind == A.colind() );
  
}

