
#pragma once
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>

using namespace std;
using namespace cmz::ed;



// This creates a block of the hamiltonian
// H( bra_begin:bra_end, ket_begin:ket_end )
// Currently double loops, but should delegate if the bra/ket coencide in the future
template <typename index_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  std::vector<slater_det>::iterator bra_begin,
  std::vector<slater_det>::iterator bra_end,
  std::vector<slater_det>::iterator ket_begin,
  std::vector<slater_det>::iterator ket_end,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh
) {

  // Extract states for superior memory access
  const size_t nbra_dets = std::distance( bra_begin, bra_end );
  const size_t nket_dets = std::distance( ket_begin, ket_end );
  std::vector<uint64_t> bra_states( nbra_dets );
  std::vector<uint64_t> ket_states( nket_dets );

  auto get_state = [](const auto& s){ return s.GetState(); };
  std::transform( bra_begin, bra_end, bra_states.begin(), get_state );
  std::transform( ket_begin, ket_end, ket_states.begin(), get_state );

  // Preallocate colind / nzval indirection
  std::vector< std::vector<index_t> > colind_by_row( nbra_dets );
  std::vector< std::vector<double> >  nzval_by_row ( nbra_dets );

  const size_t res_count = 0.07 * nket_dets;
  for( auto& v : colind_by_row ) v.reserve( res_count );
  for( auto& v : nzval_by_row )  v.reserve( res_count );

  // Construct adjacencey
  #pragma omp parallel for
  for( index_t i = 0; i < nbra_dets; ++i ) {
  for( index_t j = 0; j < nket_dets; ++j ) 
  if( std::popcount( bra_states[i] ^ ket_states[j] ) <= 4 ) {
    const auto h_el = Hop.GetHmatel( *(bra_begin+i), *(ket_begin+j) );
    if( std::abs(h_el) > H_thresh ) {
      colind_by_row[i].emplace_back( j );
      nzval_by_row [i].emplace_back( h_el );
    }
  }
  }

  // Compute row counts
  std::vector< size_t > row_counts( nbra_dets );
  std::transform( colind_by_row.begin(), colind_by_row.end(), row_counts.begin(),
    [](const auto& v){ return v.size(); } );

  // Compute NNZ 
  const size_t nnz = std::accumulate( row_counts.begin(), row_counts.end(), 0ul );

  sparsexx::csr_matrix<double, index_t> H( nbra_dets, nket_dets, nnz, 0 );
  auto& rowptr = H.rowptr();
  auto& colind = H.colind();
  auto& nzval  = H.nzval();

  // Compute rowptr
  std::exclusive_scan( row_counts.begin(), row_counts.end(), rowptr.begin(), 0 );
  rowptr[nbra_dets] = rowptr[nbra_dets - 1] + row_counts.back();

  // Linearize colind/nzval
  auto linearize_vov = []( const auto& vov, auto& lin ) {
    auto it = lin.begin();
    for( const auto& v : vov ) {
      it = std::copy( v.begin(), v.end(), it );
    }
  };

  linearize_vov( colind_by_row, colind );
  linearize_vov( nzval_by_row,  nzval  );

  return H;

}



// Dispatch make_csr_hamiltonian for non-vector SD containers (saves a copy)
template <typename index_t, typename BraIterator, typename KetIterator>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  BraIterator bra_begin,
  BraIterator bra_end,
  KetIterator ket_begin,
  KetIterator ket_end,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh
) {


  // Put SD's into contiguous, random-access memory
  std::vector< slater_det > bra_vec( bra_begin, bra_end );
  std::vector< slater_det > ket_vec( ket_begin, ket_end );

  return make_csr_hamiltonian_block<index_t>( bra_vec.begin(), bra_vec.end(),
    ket_vec.begin(), ket_vec.end(), Hop, ints, H_thresh );

}


// Syntactic sugar for SetSlaterDets containers
template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const SetSlaterDets& stts,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh
) {
  return make_csr_hamiltonian_block<index_t>( stts.begin(), stts.end(), 
    stts.begin(), stts.end(), Hop, ints, H_thresh );
}


template <typename index_t>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian( MPI_Comm comm, 
                             std::vector<slater_det>::iterator sd_begin,
                             std::vector<slater_det>::iterator sd_end,
                             const FermionHamil& Hop,
                             const intgrls::integrals& ints,
                             const double H_thresh
                           ) {

  using namespace sparsexx;
  using namespace sparsexx::detail;

  size_t ndets = std::distance( sd_begin, sd_end );
  dist_sparse_matrix< csr_matrix<double,index_t> > H_dist( comm, ndets, ndets );

  // Get local row bounds
  auto [bra_st, bra_en] = H_dist.row_bounds( get_mpi_rank(comm) );

  // Build diagonal part
  H_dist.set_diagonal_tile(
    make_csr_hamiltonian_block<index_t>( 
      sd_begin + bra_st, sd_begin + bra_en,
      sd_begin + bra_st, sd_begin + bra_en,
      Hop, ints, H_thresh
    )
  );

  // Create a copy of SD's with local bra dets zero'd out
  std::vector<slater_det> sds_offdiag( sd_begin, sd_end );
  for( auto i = bra_st; i < bra_en; ++i ) sds_offdiag[i] = slater_det();

  // Build off-diagonal part
  H_dist.set_off_diagonal_tile(
    make_csr_hamiltonian_block<index_t>( 
      sd_begin + bra_st, sd_begin + bra_en,
      sds_offdiag.begin(), sds_offdiag.end(),
      Hop, ints, H_thresh
    )
  );

  return H_dist;
    
}


template <typename index_t, typename SlaterDetIterator>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian( MPI_Comm comm, 
                             SlaterDetIterator   sd_begin,
                             SlaterDetIterator   sd_end,
                             const FermionHamil& Hop,
                             const intgrls::integrals& ints,
                             const double H_thresh
                           ) {

  std::vector<slater_det> sd_vec( sd_begin, sd_end );
  return make_dist_csr_hamiltonian<index_t>( comm, sd_vec.begin(), sd_vec.end(),
    Hop, ints, H_thresh );

}



template <typename index_t>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian_bcast( MPI_Comm comm,
                             std::vector<slater_det>::iterator sd_begin,
                             std::vector<slater_det>::iterator sd_end,
                             const FermionHamil& Hop,
                             const intgrls::integrals& ints,
                             const double H_thresh
                           ) {

  using namespace sparsexx;
  using namespace sparsexx::detail;
   
  csr_matrix<double, index_t> H_replicated;

  const auto ndets     = std::distance( sd_begin, sd_end );
  const auto comm_rank = get_mpi_rank( comm );

  // Form Hamiltonian explicitly on the root rank
  if( !comm_rank ) { 
    H_replicated = make_csr_hamiltonian_block<index_t>( sd_begin, sd_end, 
      sd_begin, sd_end, Hop, ints, H_thresh );
  }


  // Broadcast NNZ to allow for non-root ranks to allocate memory 
  size_t nnz = H_replicated.nnz();
  mpi_bcast( &nnz, 1, 0, comm );

  // Allocate H_replicated on non-root ranks
  if( comm_rank ) {
    H_replicated = csr_matrix<double, index_t>( ndets, ndets, nnz, 0 );
  }

  // Broadcast the matrix data
  mpi_bcast( H_replicated.colind(), 0, comm );
  mpi_bcast( H_replicated.rowptr(), 0, comm );
  mpi_bcast( H_replicated.nzval(),  0, comm );
     
  // Distribute Hamiltonian from replicated data
  return dist_sparse_matrix< csr_matrix<double,index_t> >( comm, H_replicated );
}

template <typename index_t, typename SlaterDetIterator>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian_bcast( MPI_Comm comm,
                             SlaterDetIterator sd_begin,
                             SlaterDetIterator sd_end,
                             const FermionHamil& Hop,
                             const intgrls::integrals& ints,
                             const double H_thresh
                           ) {
  std::vector<slater_det> sd_vec( sd_begin, sd_end );
  return make_dist_csr_hamiltonian_bcast<index_t>( comm, 
    sd_vec.begin(), sd_vec.end(), Hop, ints, H_thresh );
}
