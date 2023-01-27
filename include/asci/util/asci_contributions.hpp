#pragma once
#include <asci/types.hpp>
#include <asci/sd_operations.hpp>
#include <asci/hamiltonian_generator.hpp>

namespace asci {

template <typename WfnT>
struct asci_contrib {
  WfnT   state;
  double rv;
};

template <typename WfnT>
using asci_contrib_container = std::vector<asci_contrib<WfnT>>;

template <size_t N, size_t NShift>
void append_singles_asci_contributions(
  double                              coeff,
  wfn_t<2*N>                          state_full,
  wfn_t<N>                            state_same,
  const std::vector<uint32_t>&        occ_same,
  const std::vector<uint32_t>&        vir_same,
  const std::vector<uint32_t>&        occ_othr,
  const double*                       T_pq,
  const size_t                        LDT,
  const double*                       G_kpq,
  const size_t                        LDG,
  const double*                       V_kpq,
  const size_t                        LDV,
  double                              h_el_tol,
  double                              root_diag,
  double                              E0,
  HamiltonianGenerator<2*N>&          ham_gen,
  asci_contrib_container<wfn_t<2*N>>& asci_contributions
) {

  const auto LDG2 = LDG * LDG;
  const auto LDV2 = LDV * LDV;
  for( auto i : occ_same )
  for( auto a : vir_same ) {
   
    // Compute single excitation matrix element
    double h_el = T_pq[a + i*LDT];
    const double* G_ov = G_kpq + a*LDG + i*LDG2;
    const double* V_ov = V_kpq + a*LDV + i*LDV2;
    for( auto p : occ_same ) h_el += G_ov[p];
    for( auto p : occ_othr ) h_el += V_ov[p];

    // Early Exit
    if( std::abs(h_el) < h_el_tol ) continue;

    // Calculate Excited Determinant
    auto ex_det = state_full;
    ex_det.flip(i+NShift).flip(a+NShift);

    // Calculate Excitation Sign in a Canonical Way
    auto sign = single_excitation_sign( state_same, a, i );
    h_el *= sign;

    // Calculate fast diagonal matrix element
    auto h_diag = 
      ham_gen.fast_diag_single( occ_same, occ_othr, i, a, root_diag );
    h_el /= (E0 - h_diag);

    // Append to return values
    asci_contributions.push_back( {ex_det, coeff*h_el} );

  } // Loop over single extitations

}




template <size_t N, size_t NShift>
void append_ss_doubles_asci_contributions( 
  double                              coeff,
  wfn_t<2*N>                          state_full,
  wfn_t<N>                            state_spin,
  const std::vector<uint32_t>&        ss_occ,
  const std::vector<uint32_t>&        vir,
  const std::vector<uint32_t>&        os_occ,
  const double*                       G,
  size_t                              LDG,
  double                              h_el_tol,
  double                              root_diag,
  double                              E0,
  HamiltonianGenerator<2*N>&          ham_gen,
  asci_contrib_container<wfn_t<2*N>>& asci_contributions
) {

  const size_t nocc = ss_occ.size();
  const size_t nvir = vir.size();

  const size_t LDG2 = LDG*LDG;
  for( auto ii = 0; ii < nocc; ++ii )
  for( auto aa = 0; aa < nvir; ++aa ) {

    const auto i = ss_occ[ii];
    const auto a = vir[aa];
    const auto G_ai = G + (a + i*LDG)*LDG2;

    for( auto jj = ii + 1; jj < nocc; ++jj )
    for( auto bb = aa + 1; bb < nvir; ++bb ) {
      const auto j = ss_occ[jj];
      const auto b = vir[bb];
      const auto jb = b + j*LDG;
      const auto G_aibj = G_ai[jb];

      if( std::abs(G_aibj) < h_el_tol ) continue;

      // Calculate excited determinant string (spin)
      #if 0
      const auto full_ex_spin = (one << i) ^ (one << j) ^ (one << a) ^ (one << b);
      #else
      const auto full_ex_spin = wfn_t<N>(0).flip(i).flip(j).flip(a).flip(b);
      #endif
      auto ex_det_spin = state_spin ^ full_ex_spin;

      // Calculate the sign in a canonical way
      double sign = doubles_sign( state_spin, ex_det_spin, full_ex_spin );

      // Calculate full excited determinant
      const auto full_ex = expand_bitset<2*N>(full_ex_spin) << NShift;
      auto ex_det = state_full ^ full_ex;

      // Update sign of matrix element
      auto h_el = sign * G_aibj;

      // Evaluate fast diagonal matrix element
      auto h_diag = 
        ham_gen.fast_diag_ss_double( ss_occ, os_occ, i, j, a, b, root_diag );
      h_el /= (E0 - h_diag);

      // Append {det, c*h_el}
      asci_contributions.push_back( {ex_det, coeff*h_el} );

    } // Restricted BJ loop
  } // AI Loop

}







template <size_t N>
void append_os_doubles_asci_contributions( 
  double                              coeff,
  wfn_t<2*N>                          state_full,
  wfn_t<N>                            state_alpha,
  wfn_t<N>                            state_beta,
  const std::vector<uint32_t>&        occ_alpha,
  const std::vector<uint32_t>&        occ_beta,
  const std::vector<uint32_t>&        vir_alpha,
  const std::vector<uint32_t>&        vir_beta,
  const double*                       V,
  size_t                              LDV,
  double                              h_el_tol,
  double                              root_diag,
  double                              E0,
  HamiltonianGenerator<2*N>&          ham_gen,
  asci_contrib_container<wfn_t<2*N>>& asci_contributions
) {

  const size_t LDV2 = LDV * LDV;
  for( auto i : occ_alpha )
  for( auto a : vir_alpha ) {
    const auto V_ai = V + a + i*LDV;

    double sign_alpha = single_excitation_sign( state_alpha, a, i );
    for( auto j : occ_beta )
    for( auto b : vir_beta ) {
      const auto jb = b + j*LDV;
      const auto V_aibj = V_ai[jb*LDV2];

      if( std::abs(V_aibj) < h_el_tol ) continue;

      double sign_beta = single_excitation_sign( state_beta,  b, j );
      double sign = sign_alpha * sign_beta;
#if 0
      auto ex_det = state_full ^ (one << i) ^ (one << a) ^
                               (((one << j) ^ (one << b)) << N);
#else
      auto ex_det = state_full;
      ex_det.flip(a).flip(i).flip(j+N).flip(b+N);
#endif
      auto h_el = sign * V_aibj;

      // Evaluate fast diagonal element
      auto h_diag = 
        ham_gen.fast_diag_os_double( occ_alpha, occ_beta, i, j, a, b, root_diag );
      h_el /= ( E0 - h_diag );

      asci_contributions.push_back( {ex_det, coeff*h_el} );
    } // BJ loop
  } // AI loop

}












template <size_t N, typename IndContainer>
void generate_pairs( const IndContainer& inds, std::vector<wfn_t<N>>& w ) {
  const size_t nind = inds.size();
  w.resize((nind * (nind-1))/2,0);
  for(int i = 0, ij = 0; i < nind; ++i      )
  for(int j = i+1;       j < nind; ++j, ++ij) {
    w[ij].flip(inds[i]).flip(inds[j]);
  }
}



/**
 *  @param[in]  det       Input root determinant
 *  @param[in]  T         Triplet constraint mask
 *  @param[in]  O         Overfill mask (full mask 0 -> norb)
 *  @param[in]  B         B mask (?)
 *  @param[out] t_doubles 
 */
template <size_t N>
void generate_triplet_doubles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  std::vector<wfn_t<N>>& t_doubles
) {

  if( (det & T) == 0 ) return;

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;

  // Occ/Vir pairs to generate excitations
  std::vector<wfn_t<N>> O,V; 

  // Generate Virtual Pairs
  if( (o & T).count() >= 2 ) {
    v = o & T;
    o ^= v;
  }

  const auto virt_ind = bits_to_indices(v);
  const auto o_and_t = o & T;
  switch( (o & T).count() ) {
    case 1:
      for( auto a : virt_ind ) {
        V.emplace_back(o_and_t).flip(a);
      }
      o ^= o_and_t;
      break;
    default:
      generate_pairs(virt_ind, V);
      break;
  }

  // Generate Occupied Pairs
  const auto o_and_not_b = o & ~B;
  if( o_and_not_b.count() > 2 ) return;

  switch(o_and_not_b.count()) {
    case 1 :
      for( auto i : bits_to_indices( o & B ) ) {
        O.emplace_back(o_and_not_b).flip(i);
      }
      break;
    default:
      if( o_and_not_b.count() == 2 ) o = o_and_not_b;
      generate_pairs( bits_to_indices(o), O );
      break;
  }

  t_doubles.clear();
  for(auto ij : O) {
    const auto temp = det ^ ij;
    for( auto ab : V ) {
      t_doubles.emplace_back(temp | ab);
    }
  }
}

/**
 *  @param[in]  det       Input root determinant
 *  @param[in]  T         Triplet constraint mask
 *  @param[in]  O         Overfill mask (full mask 0 -> norb)
 *  @param[in]  B         B mask (?)
 *  @param[out] t_singles 
 */
template <size_t N>
void generate_triplet_singles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  std::vector<wfn_t<N>>& t_singles
) {

  if( (det & T).count() < 2 ) return;

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;

  if( (o & T).count() >= 1 ) {
    v = o & T;
    o ^= v;
  }

  if( (o & ~B).count() >  1 ) return;
  if( (o & ~B).count() == 1 ) o &= ~B;

  const auto occ = bits_to_indices(o);
  const auto vir = bits_to_indices(v);
  t_singles.clear();
  t_singles.reserve(occ.size() * vir.size());
  for( auto i : occ ) {
    auto temp = det; temp.flip(i);
    for( auto a : vir ) t_singles.emplace_back(temp).flip(a);
  }
}

}
