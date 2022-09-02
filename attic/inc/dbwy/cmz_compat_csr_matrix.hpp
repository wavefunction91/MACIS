#pragma once
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"

#include "csr_hamiltonian.hpp"

namespace dbwy {

namespace detail {

template <typename Iterator, typename Type>
struct value_type_equiv {
  inline static constexpr bool value =
    std::is_same_v< 
      std::remove_cvref_t<typename std::iterator_traits<Iterator>::value_type>, 
      Type >;
};

template <typename Iterator, typename Type>
inline constexpr bool value_type_equiv_v = value_type_equiv<Iterator,Type>::value;
  
template <size_t N, typename Iterator>
std::enable_if_t< value_type_equiv_v<Iterator, cmz::ed::slater_det>, 
                  std::vector<std::bitset<N>> >
to_bitset( Iterator states_begin, Iterator states_end ) {

  const size_t nstates = std::distance( states_begin, states_end );
  std::vector< std::bitset<N> > bits( nstates );

  if( nstates ) {
    const size_t norb = states_begin->GetNorbs();
    const auto alpha_mask = (1ul << norb) - 1ul;
    const auto beta_mask  = alpha_mask << norb;

    auto conv = [=]( cmz::ed::slater_det _state ) {
      auto state = _state.GetState();
      std::bitset<N> state_alpha = state & alpha_mask;
      std::bitset<N> state_beta  = (state & beta_mask) >> norb;
      return state_alpha | (state_beta << (N/2));
    };

    std::transform( states_begin, states_end, bits.begin(), conv );
  }

  return bits;
}

} // namespace detail




// Dispatch make_csr_hamiltonian_block for non-bitset containers
template <typename index_t, typename BraIterator, typename KetIterator>
std::enable_if_t< 
  detail::value_type_equiv_v<BraIterator, cmz::ed::slater_det> and
  detail::value_type_equiv_v<KetIterator, cmz::ed::slater_det>,
  sparsexx::csr_matrix<double,index_t> 
>
  make_csr_hamiltonian_block(
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





template <typename index_t, typename Iterator>
std::enable_if_t< detail::value_type_equiv_v<Iterator, cmz::ed::slater_det>,
  sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
>
  make_dist_csr_hamiltonian( MPI_Comm                         comm, 
                             Iterator                         sd_begin,
                             Iterator                         sd_end,
                             HamiltonianGenerator<64>&  ham_gen,
                             const double                     H_thresh
                           ) {

  auto sd_vec = detail::to_bitset<64>(sd_begin, sd_end);
  return make_dist_csr_hamiltonian<index_t>( comm, sd_vec.begin(), sd_vec.end(),
    ham_gen, H_thresh);

}


template <typename index_t, typename Iterator>
std::enable_if_t< detail::value_type_equiv_v<Iterator, cmz::ed::slater_det>,
  sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double,index_t> >
>
  make_dist_csr_hamiltonian( MPI_Comm comm, 
                             Iterator   sd_begin,
                             Iterator   sd_end,
                             const cmz::ed::FermionHamil& Hop,
                             const cmz::ed::intgrls::integrals& ints,
                             const double H_thresh
                           ) {

  DoubleLoopHamiltonianGenerator<64> ham_gen( sd_begin->GetNorbs(), 
    ints.u.data(), ints.t.data() );
  return make_dist_csr_hamiltonian<index_t>( comm, sd_begin, sd_end, ham_gen, H_thresh );

}



// Syntactic sugar for SetSlaterDets containers
template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const cmz::ed::SetSlaterDets& stts,
  const cmz::ed::FermionHamil&  Hop,
  const cmz::ed::intgrls::integrals& ints,
  const double H_thresh
) {

  // Generate intermediates
  dbwy::DoubleLoopHamiltonianGenerator<64> ham_gen( stts.begin()->GetNorbs(), 
    ints.u.data(), ints.t.data() );

  return make_csr_hamiltonian_block<index_t>( stts.begin(), stts.end(), 
    stts.begin(), stts.end(), ham_gen, H_thresh );

}

}
