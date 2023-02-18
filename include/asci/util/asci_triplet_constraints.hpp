#pragma once
#include <asci/types.hpp>
#include <asci/sd_operations.hpp>

namespace asci {
#if 1

template <size_t N>
auto make_triplet_masks(size_t norb,
  unsigned i, unsigned j, unsigned k) {

  wfn_t<N> T(0); T.flip(i).flip(j).flip(k);
  auto overfill = full_mask<N>(norb);
  wfn_t<N> B(1); B <<= k; B = B.to_ullong() - 1;

  return std::make_tuple(T,overfill,B);
}

template <size_t N>
auto generate_triplet_single_excitations( wfn_t<N> det,
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B ) {

  if( (det & T).count() < 2 ) 
    return std::make_pair(wfn_t<N>(0), wfn_t<N>(0));

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;

  if( (o & T).count() >= 1 ) {
    v = o & T;
    o ^= v;
  }

  if( (o & ~B).count() >  1 )
    return std::make_pair(wfn_t<N>(0), wfn_t<N>(0));

  if( (o & ~B).count() == 1 ) o &= ~B;

  return std::make_pair(o,v);
}

template <size_t N>
void generate_triplet_singles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  std::vector<wfn_t<N>>& t_singles
) {

  auto [o,v] = generate_triplet_single_excitations( det, T, O_mask, B );
  const auto oc = o.count();
  const auto vc = v.count();
  if( !oc or !vc ) return;

  t_singles.clear();
  t_singles.reserve(oc*vc);
  const auto occ = bits_to_indices(o);
  const auto vir = bits_to_indices(v);
  for( auto i : occ ) {
    auto temp = det; temp.flip(i);
    for( auto a : vir ) t_singles.emplace_back(temp).flip(a);
  }
  
}

template <typename... Args>
unsigned count_triplet_singles(Args&&... args) {
  auto [o,v] = generate_triplet_single_excitations( std::forward<Args>(args)... );
  return o.count() * v.count();
}


/**
 *  @param[in]  det       Input root determinant
 *  @param[in]  T         Triplet constraint mask
 *  @param[in]  O         Overfill mask (full mask 0 -> norb)
 *  @param[in]  B         B mask (?)
 *  @param[out] t_doubles 
 */
template <size_t N>
auto generate_triplet_double_excitations( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B
) {
  // Occ/Vir pairs to generate excitations
  std::vector<wfn_t<N>> O,V; 

  if( (det & T) == 0 ) return std::make_tuple(O,V);

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;


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
  if( o_and_not_b.count() > 2 ) return std::make_tuple(O,V);

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

  return std::make_tuple(O,V);
}

template <size_t N>
void generate_triplet_doubles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  std::vector<wfn_t<N>>& t_doubles
) {

  auto [O,V] = generate_triplet_double_excitations(det,T,O_mask,B);

  t_doubles.clear();
  for(auto ij : O) {
    const auto temp = det ^ ij;
    for( auto ab : V ) {
      t_doubles.emplace_back(temp | ab);
    }
  }
}

/**
 *  @param[in]  det Input root determinant
 *  @param[in]  T   Triplet constraint mask
 *  @param[in]  O   Overfill mask (full mask 0 -> norb)
 *  @param[in]  B   B mask (?)
 */
template <size_t N>
unsigned count_triplet_doubles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B
) {

  if( (det & T) == 0 ) return 0;

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;

  // Generate Virtual Pairs
  if( (o & T).count() >= 2 ) {
    v = o & T;
    o ^= v;
  }

  unsigned nv_pairs = v.count();
  const auto o_and_t = o & T;
  switch( (o & T).count() ) {
    case 1:
      o ^= o_and_t;
      break;
    default:
      nv_pairs = (nv_pairs * (nv_pairs-1))/2;
      break;
  }

  // Generate Occupied Pairs
  const auto o_and_not_b = o & ~B;
  if( o_and_not_b.count() > 2 ) return 0;

  unsigned no_pairs = 0;
  switch(o_and_not_b.count()) {
    case 1 :
      no_pairs = (o & B).count();
      break;
    default:
      if( o_and_not_b.count() == 2 ) o = o_and_not_b;
      no_pairs = o.count();
      no_pairs = (no_pairs * (no_pairs-1))/2;
      break;
  }

  return no_pairs * nv_pairs;
}

template <size_t N, typename... Args>
size_t triplet_histogram( wfn_t<N> det, size_t n_os_singles, size_t n_os_doubles, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B ) {

  auto ns = count_triplet_singles( det, T, O_mask, B );
  auto nd = count_triplet_doubles( det, T, O_mask, B );

  size_t ndet = 0;
  ndet += ns;                // AA
  ndet += nd;                // AAAA
  ndet += ns * n_os_singles; // AABB
  auto T_min = ffs(T) - 1;
  if( (det & T).count() == 3 and ((det ^ T) >> T_min).count() == 0 ) {
    ndet += n_os_singles + n_os_doubles + 1; // BB + BBBB + No Excitations
  }

  return ndet;
}





template <size_t N>
void generate_triplet_singles_contributions_ss(
  double coeff,
  wfn_t<N> det, wfn_t<N> T, wfn_t<N> O, wfn_t<N> B,
  wfn_t<N> os_det, 
  const std::vector<uint32_t>&        occ_same,
  const std::vector<uint32_t>&        occ_othr,
  const double*                       eps,
  const double*                       T_pq,
  const size_t                        LDT,
  const double*                       G_kpq,
  const size_t                        LDG,
  const double*                       V_kpq,
  const size_t                        LDV,
  double                              h_el_tol,
  double                              root_diag,
  double                              E0,
  HamiltonianGenerator<N>&            ham_gen,
  asci_contrib_container<wfn_t<N>>& asci_contributions) {

  auto [o,v] = generate_triplet_single_excitations(det, T, O, B);
  const auto no = o.count();
  const auto nv = v.count();
  if(!no or !nv) return;

  const size_t LDG2 = LDG * LDG;
  const size_t LDV2 = LDV * LDV;
  for(int ii = 0; ii < no; ++ii) {
    const auto i = fls(o);
    o.flip(i);
    auto v_cpy = v;
  for(int aa = 0; aa < nv; ++aa) {
    const auto a = fls(v_cpy);
    v_cpy.flip(a);

    double h_el = T_pq[a + i*LDT];
    const double* G_ov = G_kpq + a*LDG + i*LDG2;
    const double* V_ov = V_kpq + a*LDV + i*LDV2;
    for( auto p : occ_same ) h_el += G_ov[p];
    for( auto p : occ_othr ) h_el += V_ov[p];

    // Early Exit
    if( std::abs(coeff * h_el) < h_el_tol ) continue;

    // Calculate Excited Determinant
    auto ex_det = det | os_det; ex_det.flip(i).flip(a);

    // Compute Sign in a Canonical Way
    auto sign = single_excitation_sign(det, a, i);
    h_el *= sign;

    // Compute Fast Diagonal Matrix Element
    auto h_diag =
      //ham_gen.fast_diag_single(occ_same, occ_othr, i, a, root_diag);
      ham_gen.fast_diag_single(eps[i], eps[a], i, a, 
        root_diag);
    h_el /= (E0 - h_diag);

    asci_contributions.push_back( {ex_det, coeff * h_el} );
  }
  }

}



template <size_t N>
void generate_triplet_doubles_contributions_ss(
  double coeff,
  wfn_t<N> det, wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  wfn_t<N> os_det, 
  const std::vector<uint32_t>&        occ_same,
  const std::vector<uint32_t>&        occ_othr,
  const double*                       eps,
  const double*                       G,
  const size_t                        LDG,
  double                              h_el_tol,
  double                              root_diag,
  double                              E0,
  HamiltonianGenerator<N>&            ham_gen,
  asci_contrib_container<wfn_t<N>>& asci_contributions) {

  auto [O,V] = generate_triplet_double_excitations(det, T, O_mask, B);
  const auto no_pairs = O.size();
  const auto nv_pairs = V.size();
  if( !no_pairs or !nv_pairs ) return;

  const size_t LDG2 = LDG * LDG;
  for(int _ij = 0; _ij < no_pairs; ++_ij) {
    const auto ij = O[_ij];
    const auto i  = ffs(ij) - 1;
    const auto j  = fls(ij);
    const auto G_ij = G + (j + i*LDG2)*LDG;
    const auto ex_ij = det ^ ij;
  for(int _ab = 0; _ab < nv_pairs; ++_ab) {
    const auto ab = V[_ab];
    const auto a  = ffs(ab) - 1;
    const auto b  = fls(ab);
    
    const auto G_aibj = G_ij[b + a*LDG2];
    //printf(" %d %d %d %d %.6e\n", i,j,a,b, G_aibj);

    // Early Exit
    if( std::abs(coeff * G_aibj) < h_el_tol ) continue;

    // Calculate Excited Determinant (spin)
    const auto full_ex_spin = ij | ab;
    const auto ex_det_spin  = ex_ij | ab;
    

    // Compute Sign in a Canonical Way
    auto sign = doubles_sign( det, ex_det_spin, full_ex_spin );
    
    // Calculate Full Excited Determinant
    const auto full_ex = ex_det_spin | os_det;

    // Update Sign of Matrix Element
    auto h_el = sign * G_aibj;

    // Evaluate fast diagonal matrix element
    auto h_diag =
      //ham_gen.fast_diag_ss_double( occ_same, occ_othr, i, j, a, b, root_diag);
      ham_gen.fast_diag_ss_double( eps[i], eps[j],
        eps[a], eps[b], i, j, a, b, root_diag);
    h_el /= (E0 - h_diag);

    asci_contributions.push_back( {full_ex, coeff * h_el} );

  }
  }
}




template <size_t N>
void generate_triplet_doubles_contributions_os(
  double coeff,
  wfn_t<N> det, wfn_t<N> T, wfn_t<N> O, wfn_t<N> B,
  wfn_t<N> os_det, 
  const std::vector<uint32_t>&        occ_same,
  const std::vector<uint32_t>&        occ_othr,
  const std::vector<uint32_t>&        vir_othr,
  const double*                       eps_same,
  const double*                       eps_othr,
  const double*                       V,
  const size_t                        LDV,
  double                              h_el_tol,
  double                              root_diag,
  double                              E0,
  HamiltonianGenerator<N>&            ham_gen,
  asci_contrib_container<wfn_t<N>>& asci_contributions) {

  
  // Generate Single Excitations that Satisfy the Constraint
  auto [o,v] = generate_triplet_single_excitations(det, T, O, B);
  const auto no = o.count();
  const auto nv = v.count();
  if(!no or !nv) return;

  const size_t LDV2 = LDV * LDV;
  for(int ii = 0; ii < no; ++ii) {
    const auto i = fls(o);
    o.flip(i);
    auto v_cpy = v;
  for(int aa = 0; aa < nv; ++aa) {
    const auto a = fls(v_cpy);
    v_cpy.flip(a);

    const auto* V_ai = V + a + i*LDV;
    double sign_same = single_excitation_sign( det, a, i );

    for( auto j : occ_othr )
    for( auto b : vir_othr ) {
      const auto jb = b + j*LDV;
      const auto V_aibj = V_ai[jb*LDV2];

      // Early Exist
      if( std::abs(coeff * V_aibj) < h_el_tol ) continue;

      //double sign_othr = single_excitation_sign( os_det >> (N/2),  b, j );
      double sign_othr = single_excitation_sign(bitset_hi_word(os_det) ,  b, j );
      double sign = sign_same * sign_othr;

      // Compute Excited Determinant
      auto ex_det = det | os_det; ex_det.flip(i).flip(a).flip(j+N/2).flip(b+N/2);

      // Finalize Matrix Element
      auto h_el = sign * V_aibj;

      auto h_diag = 
        //ham_gen.fast_diag_os_double( occ_same, occ_othr, i, j, a, b, root_diag );
        ham_gen.fast_diag_os_double( eps_same[i], eps_othr[j],
          eps_same[a], eps_othr[b], i, j, a, b, root_diag );
      h_el /= ( E0 - h_diag );

      asci_contributions.push_back( {ex_det, coeff*h_el} );
    } // BJ

  } // A
  } // I
}

#endif
}
