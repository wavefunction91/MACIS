
#pragma once
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;

#include <bitset>
#include "dbwy/hamiltonian_generator.hpp"
#include "dbwy/double_loop.hpp"
#include "dbwy/residue_arrays.hpp"

using namespace std;
using namespace cmz::ed;

namespace detail {

template <typename Iterator, typename Type>
struct value_type_equiv {
  inline static constexpr bool value =
    std::is_same_v< std::remove_cvref_t<typename std::iterator_traits<Iterator>::value_type>, Type >;
};

template <typename Iterator, typename Type>
inline constexpr bool value_type_equiv_v = value_type_equiv<Iterator,Type>::value;
  
template <size_t N, typename Iterator>
std::enable_if_t< value_type_equiv_v<Iterator, slater_det>, std::vector<std::bitset<N>> >
to_bitset( Iterator states_begin, Iterator states_end ) {

  const size_t nstates = std::distance( states_begin, states_end );
  std::vector< std::bitset<N> > bits( nstates );

  if( nstates ) {
    const size_t norb = states_begin->GetNorbs();
    const auto alpha_mask = (1ul << norb) - 1ul;
    const auto beta_mask  = alpha_mask << norb;

    auto conv = [=]( slater_det _state ) {
      auto state = _state.GetState();
      std::bitset<N> state_alpha = state & alpha_mask;
      std::bitset<N> state_beta  = (state & beta_mask) >> norb;
      return state_alpha | (state_beta << (N/2));
    };

    std::transform( states_begin, states_end, bits.begin(), conv );
  }

  return bits;
}

}






// Base implementation of bitset CSR generation
template <typename index_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  std::vector< std::bitset<64> >::iterator bra_begin,
  std::vector< std::bitset<64> >::iterator bra_end,
  std::vector< std::bitset<64> >::iterator ket_begin,
  std::vector< std::bitset<64> >::iterator ket_end,
  dbwy::HamiltonianGenerator<64>&          ham_gen,
  double                                   H_thresh
) {

  size_t nbra = std::distance( bra_begin, bra_end );
  size_t nket = std::distance( ket_begin, ket_end );
  
  if( nbra and nket ) {
    return ham_gen.make_csr_hamiltonian_block<index_t>(
      bra_begin, bra_end, ket_begin, ket_end, H_thresh
    );
  } else {
    return sparsexx::csr_matrix<double,index_t>(nbra,nket,0,0);
  }

}








// Dispatch make_csr_hamiltonian_block for non-bitset containers
template <typename index_t, typename BraIterator, typename KetIterator>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  BraIterator bra_begin,
  BraIterator bra_end,
  KetIterator ket_begin,
  KetIterator ket_end,
  dbwy::HamiltonianGenerator<64>& ham_gen,
  const double H_thresh
) {

  size_t nbra = std::distance( bra_begin, bra_end );
  size_t nket = std::distance( ket_begin, ket_end );

  if(nbra and nket) {

    // Convert cmz slater_det -> bitset
    auto bra_vec = detail::to_bitset<64>(bra_begin, bra_end);
    auto ket_vec = detail::to_bitset<64>(ket_begin, ket_end);

    return make_csr_hamiltonian_block<index_t>(bra_vec.begin(), bra_vec.end(),
      ket_vec.begin(), ket_vec.end(), ham_gen, H_thresh );

  } else {

    return sparsexx::csr_matrix<double,index_t>(nbra,nket,0,0);

  }

}


// Syntactic sugar for SetSlaterDets containers
template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const SetSlaterDets& stts,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh
) {

  // Generate intermediates
  dbwy::DoubleLoopHamiltonianGenerator<64> ham_gen( stts.begin()->GetNorbs(), ints );

  return make_csr_hamiltonian_block<index_t>( stts.begin(), stts.end(), 
    stts.begin(), stts.end(), ham_gen, H_thresh );

}
















// Base implementation of dist-CSR H construction for bitsets
template <typename index_t>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian( MPI_Comm comm, 
                             std::vector<std::bitset<64>>::iterator sd_begin,
                             std::vector<std::bitset<64>>::iterator sd_end,
                             dbwy::HamiltonianGenerator<64>&        ham_gen,
                             const double                           H_thresh
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
    std::vector<std::bitset<64>> sds_offdiag( sd_begin, sd_end );
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

template <typename index_t, typename Iterator>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian( MPI_Comm                         comm, 
                             Iterator                         sd_begin,
                             Iterator                         sd_end,
                             dbwy::HamiltonianGenerator<64>&  ham_gen,
                             const double                     H_thresh
                           ) {

  auto sd_vec = detail::to_bitset<64>(sd_begin, sd_end);
  return make_dist_csr_hamiltonian<index_t>( comm, sd_vec.begin(), sd_vec.end(),
    ham_gen, H_thresh);

}


template <typename index_t, typename Iterator>
sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
  make_dist_csr_hamiltonian( MPI_Comm comm, 
                             Iterator   sd_begin,
                             Iterator   sd_end,
                             const FermionHamil& Hop,
                             const intgrls::integrals& ints,
                             const double H_thresh
                           ) {

  dbwy::DoubleLoopHamiltonianGenerator<64> ham_gen( sd_begin->GetNorbs(), ints );
  return make_dist_csr_hamiltonian<index_t>( comm, sd_begin, sd_end, ham_gen, H_thresh );

}








#if 0
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
    auto t_st = clock_type::now();
    H_replicated = make_csr_hamiltonian_block<index_t>( sd_begin, sd_end, 
      sd_begin, sd_end, Hop, ints, H_thresh );

    duration_type dur = clock_type::now() - t_st;
    std::cout << "Serial H Construction took " << dur.count() << " ms" << std::endl;
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
#endif
