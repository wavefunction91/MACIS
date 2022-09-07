
#pragma once
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <asci/hamiltonian_generator.hpp>
#include <asci/types.hpp>

namespace asci {



// Base implementation of bitset CSR generation
template <typename index_t,size_t N>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  wavefunction_iterator_t<N> bra_begin,
  wavefunction_iterator_t<N> bra_end,
  wavefunction_iterator_t<N> ket_begin,
  wavefunction_iterator_t<N> ket_end,
  HamiltonianGenerator<N>&   ham_gen,
  double                     H_thresh
) {

  size_t nbra = std::distance( bra_begin, bra_end );
  size_t nket = std::distance( ket_begin, ket_end );
  
  if( nbra and nket ) {
    return ham_gen.template make_csr_hamiltonian_block<index_t>(
      bra_begin, bra_end, ket_begin, ket_end, H_thresh
    );
  } else {
    return sparsexx::csr_matrix<double,index_t>(nbra,nket,0,0);
  }

}


// Base implementation of dist-CSR H construction for bitsets
template <typename index_t, size_t N>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian( MPI_Comm comm, 
                             wavefunction_iterator_t<N> sd_begin,
                             wavefunction_iterator_t<N> sd_end,
                             HamiltonianGenerator<N>&   ham_gen,
                             const double               H_thresh
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
      ham_gen, H_thresh
    )
  );

  auto world_size = get_mpi_size(comm);

  if( world_size > 1 ) {

    // Create a copy of SD's with local bra dets zero'd out
    std::vector<std::bitset<N>> sds_offdiag( sd_begin, sd_end );
    for( auto i = bra_st; i < bra_en; ++i ) sds_offdiag[i] = 0ul;

    // Build off-diagonal part
    H_dist.set_off_diagonal_tile(
      make_csr_hamiltonian_block<index_t>( 
        sd_begin + bra_st, sd_begin + bra_en,
        sds_offdiag.begin(), sds_offdiag.end(),
        ham_gen, H_thresh
      )
    );
  }

  return H_dist;
    
}


}
