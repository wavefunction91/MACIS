#pragma once
#include "sd_operations.hpp"

namespace dbwy {

template <size_t N, size_t NShift>
void append_singles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_same,
  const std::vector<uint32_t>& occ_same,
  const std::vector<uint32_t>& vir_same,
  const std::vector<uint32_t>& occ_othr,
  const double*                T_pq,
  const double*                G_kpq,
  const double*                V_kpq,
  size_t                       norb,
  double                       h_el_tol,
  std::vector< std::pair<std::bitset<2*N>,double> >& asci_contributions
) {

  const std::bitset<2*N> one = 1;
  const size_t norb2 = norb*norb;
  for( auto i : occ_same )
  for( auto a : vir_same ) {

    double h_el = T_pq[a + i*norb];

    const double* G_ov = G_kpq + a*norb + i*norb2;
    const double* V_ov = V_kpq + a*norb + i*norb2;
    for( auto p : occ_same ) h_el += G_ov[p];
    for( auto p : occ_othr ) h_el += V_ov[p];

    // Early Exit
    if( std::abs(h_el) < h_el_tol ) continue;

    // Calculate Excited Determinant
    auto ex_det = state_full ^ (one << (i+NShift)) ^ (one << (a+NShift));

    // Calculate Excitation Sign in a Canonical Way
    auto sign = single_excitation_sign( state_same, a, i );
    h_el *= sign;

    // Append to return values
    asci_contributions.push_back( {ex_det, coeff*h_el} );

  } // Loop over single extitations

}


template <size_t N, size_t NShift>
void append_ss_doubles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_spin,
  const std::vector<uint32_t>& occ,
  const std::vector<uint32_t>& vir,
  const double*                G,
  size_t                       norb,
  double                       h_el_tol,
  std::vector< std::pair<std::bitset<2*N>,double> >& asci_contributions
) {

  const size_t nocc = occ.size();
  const size_t nvir = vir.size();

  const std::bitset<N> one = 1;
  for( auto ii = 0; ii < nocc; ++ii )
  for( auto aa = 0; aa < nvir; ++aa ) {

    const auto i = occ[ii];
    const auto a = vir[aa];
    const auto G_ai = G + a + i*norb;

    for( auto jj = ii + 1; jj < nocc; ++jj )
    for( auto bb = aa + 1; bb < nvir; ++bb ) {
      const auto j = occ[jj];
      const auto b = vir[bb];
      const auto jb = b + j*norb;
      const auto G_aibj = G_ai[jb*norb*norb];

      if( std::abs(G_aibj) < h_el_tol ) continue;

      // Calculate excited determinant string (spin)
      const auto full_ex_spin = (one << i) ^ (one << j) ^ (one << a) ^ (one << b);
      auto ex_det_spin = state_spin ^ full_ex_spin;

      // Calculate the sign in a canonical way
      double sign = 1.;
      {
        auto ket = state_spin;
        auto bra = ex_det_spin;
        const auto _o1 = first_occupied_flipped( ket, full_ex_spin );
        const auto _v1 = first_occupied_flipped( bra, full_ex_spin );
        sign = single_excitation_sign( ket, _v1, _o1 );

        ket ^= (one << _o1) ^ (one << _v1);
        const auto fx = bra ^ ket;
        const auto _o2 = first_occupied_flipped( ket, fx );
        const auto _v2 = first_occupied_flipped( bra, fx );
        sign *= single_excitation_sign( ket, _v2, _o2 );
      }

      // Calculate full excited determinant
      const auto full_ex = expand_bitset<2*N>(full_ex_spin) << NShift;
      auto ex_det = state_full ^ full_ex;

      // Update sign of matrix element
      auto h_el = sign * G_aibj;

      // Append {det, c*h_el}
      asci_contributions.push_back( {ex_det, coeff*h_el} );

    } // Restricted BJ loop
  } // AI Loop


}


template <size_t N>
void append_os_doubles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_alpha,
  std::bitset<N>               state_beta,
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta,
  const std::vector<uint32_t>& vir_alpha,
  const std::vector<uint32_t>& vir_beta,
  const double*                V,
  size_t                       norb,
  double                       h_el_tol,
  std::vector< std::pair<std::bitset<2*N>,double> >& asci_contributions
) {

  const std::bitset<2*N> one = 1;
  for( auto i : occ_alpha )
  for( auto a : vir_alpha ) {
    const auto V_ai = V + a + i*norb;

    double sign_alpha = single_excitation_sign( state_alpha, a, i );
    for( auto j : occ_beta )
    for( auto b : vir_beta ) {
      const auto jb = b + j*norb;
      const auto V_aibj = V_ai[jb*norb*norb];

      if( std::abs(V_aibj) < h_el_tol ) continue;

      double sign_beta = single_excitation_sign( state_beta,  b, j );
      double sign = sign_alpha * sign_beta;
      auto ex_det = state_full ^ (one << i) ^ (one << a) ^
                               (((one << j) ^ (one << b)) << N);
      auto h_el = sign * V_aibj;

      asci_contributions.push_back( {ex_det, coeff*h_el} );
    }
  }

}

}
