#pragma once
#include <asci/hamiltonian_generator.hpp>

namespace asci {


template <size_t N>
double HamiltonianGenerator<N>::matrix_element( full_det_t bra, full_det_t ket ) 
  const {

  auto bra_alpha = truncate_bitset<N/2>(bra);
  auto ket_alpha = truncate_bitset<N/2>(ket);
  auto bra_beta  = truncate_bitset<N/2>(bra >> (N/2));
  auto ket_beta  = truncate_bitset<N/2>(ket >> (N/2));

  auto ex_alpha = bra_alpha ^ ket_alpha;
  auto ex_beta  = bra_beta  ^ ket_beta;

  auto bra_occ_alpha = bits_to_indices( bra_alpha );
  auto bra_occ_beta  = bits_to_indices( bra_beta  );

  return matrix_element( bra_alpha, ket_alpha, ex_alpha,
    bra_beta, ket_beta, ex_beta, bra_occ_alpha, bra_occ_beta );

}

template <size_t N>
double HamiltonianGenerator<N>::matrix_element( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta, 
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta ) const {

  const uint32_t ex_alpha_count = ex_alpha.count();
  const uint32_t ex_beta_count  = ex_beta.count();

  if( (ex_alpha_count + ex_beta_count) > 4 ) return 0.;

  if( ex_alpha_count == 4 ) 
    return matrix_element_4( bra_alpha, ket_alpha, ex_alpha );

  else if( ex_beta_count == 4 )
    return matrix_element_4( bra_beta, ket_beta, ex_beta );

  else if( ex_alpha_count == 2 and ex_beta_count == 2 )
    return matrix_element_22( bra_alpha, ket_alpha, ex_alpha,
      bra_beta, ket_beta, ex_beta );

  else if( ex_alpha_count == 2 )
    return matrix_element_2( bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
      bra_occ_beta );

  else if( ex_beta_count == 2 )
    return matrix_element_2( bra_beta, ket_beta, ex_beta, bra_occ_beta,
      bra_occ_alpha );

  else return matrix_element_diag( bra_occ_alpha, bra_occ_beta );

}



template <size_t N>
double HamiltonianGenerator<N>::matrix_element_4( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex ) const {

  auto [o1,v1,o2,v2,sign] = doubles_sign_indices( bra, ket, ex );

  return sign * G_pqrs_[v1 + o1*norb_ + v2*norb2_ + o2*norb3_];

}


template <size_t N>
double HamiltonianGenerator<N>::matrix_element_22( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta ) const {

  auto [o1,v1,sign_a] = 
    single_excitation_sign_indices( bra_alpha, ket_alpha, ex_alpha );
  auto [o2,v2,sign_b] = 
    single_excitation_sign_indices( bra_beta,  ket_beta,  ex_beta  );
  auto sign = sign_a*sign_b;

  return sign * V_pqrs_[v1 + o1*norb_ + v2*norb2_ + o2*norb3_];

}



template <size_t N>
double HamiltonianGenerator<N>::matrix_element_2( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex,
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta ) const {

  auto [o1,v1,sign] = single_excitation_sign_indices(bra,ket,ex);
  
  double h_el = T_pq_[v1 + o1*norb_];

  const double* G_red_ov = G_red_.data() + v1*norb_ + o1*norb2_;
  for( auto p : bra_occ_alpha ) {
    h_el += G_red_ov[p];
  }

  const double* V_red_ov = V_red_.data() + v1*norb_ + o1*norb2_;
  for( auto p : bra_occ_beta ) {
    h_el += V_red_ov[p];
  }

  return sign * h_el;
}




template <size_t N>
double HamiltonianGenerator<N>::matrix_element_diag( 
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta ) const {

  double h_el = 0;

  // One-electron piece
  for( auto p : occ_alpha ) h_el += T_pq_[p*(norb_+1)];
  for( auto p : occ_beta  ) h_el += T_pq_[p*(norb_+1)];


  // Same-spin two-body term
  for( auto q : occ_alpha )
  for( auto p : occ_alpha ) {
    h_el += G2_red_[p + q*norb_];
  }
  for( auto q : occ_beta )
  for( auto p : occ_beta ) {
    h_el += G2_red_[p + q*norb_];
  }

  // Opposite-spin two-body term
  for( auto q : occ_beta  )
  for( auto p : occ_alpha ) {
    h_el += V2_red_[p + q*norb_];
  }

  return h_el;
}
}
