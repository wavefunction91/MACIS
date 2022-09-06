#include <asci/hamiltonian_generator.hpp>
#include <blas.hh>
#include <lapack.hh>

namespace asci {

template <size_t N>
HamiltonianGenerator<N>::HamiltonianGenerator( size_t no, double* V, double* T ) :
  norb_(no), norb2_(no*no), norb3_(no*no*no),
  V_pqrs_(V), T_pq_(T) {

  generate_integral_intermediates(no, V_pqrs_);

}


////////////////////////////////////////////////
//    Routines for fast diagonal elements     //
////////////////////////////////////////////////
template <size_t N>
double HamiltonianGenerator<N>::single_orbital_en( uint32_t orb,
  const std::vector<uint32_t>& ss_occ,
  const std::vector<uint32_t>& os_occ ) const {

  // One electron component
  double orb_en = T_pq_[ orb + orb*norb_ ];

  // Same-spin two-body term
  for( auto q : ss_occ )
    orb_en += G2_red_[orb + q*norb_] + G2_red_[q + orb*norb_];
  orb_en -= G2_red_[orb + orb*norb_];

  // Opposite-spin two-body term
  for( auto q : os_occ  )
    orb_en += V2_red_[orb + q*norb_];

  return orb_en;
}

template <size_t N>
double HamiltonianGenerator<N>::fast_diag_single( 
  // These refer to original determinant
  const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ, 
  uint32_t orb_hol, uint32_t orb_par, double orig_det_Hii ) const {

  return orig_det_Hii
       + single_orbital_en( orb_par, ss_occ, os_occ )  
       - single_orbital_en( orb_hol, ss_occ, os_occ )
       - G2_red_[ orb_par + norb_*orb_hol ] 
       - G2_red_[ orb_hol + norb_*orb_par ];
}

template <size_t N>
double HamiltonianGenerator<N>::fast_diag_ss_double( 
  // These refer to original determinant
  const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ, 
  uint32_t orb_hol1, uint32_t orb_hol2, uint32_t orb_par1, uint32_t orb_par2,
  double orig_det_Hii ) const {

  return orig_det_Hii
       + single_orbital_en( orb_par1, ss_occ, os_occ ) 
       + single_orbital_en( orb_par2, ss_occ, os_occ )
       - single_orbital_en( orb_hol1, ss_occ, os_occ ) 
       - single_orbital_en( orb_hol2, ss_occ, os_occ )
       + G2_red_[ orb_hol1 + norb_*orb_hol2 ] 
       + G2_red_[ orb_hol2 + norb_*orb_hol1 ]
       + G2_red_[ orb_par1 + norb_*orb_par2 ] 
       + G2_red_[ orb_par2 + norb_*orb_par1 ]
       - G2_red_[ orb_par1 + norb_*orb_hol1 ] 
       - G2_red_[ orb_hol1 + norb_*orb_par1 ]
       - G2_red_[ orb_par2 + norb_*orb_hol1 ] 
       - G2_red_[ orb_hol1 + norb_*orb_par2 ]
       - G2_red_[ orb_par1 + norb_*orb_hol2 ] 
       - G2_red_[ orb_hol2 + norb_*orb_par1 ]
       - G2_red_[ orb_par2 + norb_*orb_hol2 ] 
       - G2_red_[ orb_hol2 + norb_*orb_par2 ];
}

template <size_t N>
double HamiltonianGenerator<N>::fast_diag_os_double( 
  // These refer to original determinant
  const std::vector<uint32_t>& up_occ, const std::vector<uint32_t>& do_occ,
  uint32_t orb_holu, uint32_t orb_hold, uint32_t orb_paru, uint32_t orb_pard,
  double orig_det_Hii ) const {

  return orig_det_Hii
       + single_orbital_en( orb_paru, up_occ, do_occ ) 
       + single_orbital_en( orb_pard, do_occ, up_occ )
       - single_orbital_en( orb_holu, up_occ, do_occ ) 
       - single_orbital_en( orb_hold, do_occ, up_occ )
       + V2_red_[ orb_holu + norb_*orb_hold ] 
       + V2_red_[ orb_paru + norb_*orb_pard ]
       - G2_red_[ orb_paru + norb_*orb_holu ] 
       - G2_red_[ orb_holu + norb_*orb_paru ]
       - G2_red_[ orb_pard + norb_*orb_hold ] 
       - G2_red_[ orb_hold + norb_*orb_pard ]
       - V2_red_[ orb_paru + norb_*orb_hold ] 
       - V2_red_[ orb_holu + norb_*orb_pard ];
}
/////////////////////////////////////////////////////
//   END - Routines for fast diagonal elements     //
/////////////////////////////////////////////////////




template <size_t N>
void HamiltonianGenerator<N>::generate_integral_intermediates(
  size_t no, const double* V) {

  size_t no2 = no  * no;
  size_t no3 = no2 * no;
  size_t no4 = no3 * no;

  // G(i,j,k,l) = V(i,j,k,l) - V(i,l,k,j)
  G_pqrs_ = std::vector<double>( V, V + no4 );
  for( auto i = 0ul; i < no; ++i )
  for( auto j = 0ul; j < no; ++j )
  for( auto k = 0ul; k < no; ++k )
  for( auto l = 0ul; l < no; ++l ) {
    G_pqrs_[i + j*no + k*no2 + l*no3] -= V[i + l*no + k*no2 + j*no3];
  }

  // G_red(i,j,k) = G(i,j,k,k) = G(k,k,i,j)
  // V_red(i,j,k) = V(i,j,k,k) = V(k,k,i,j)
  G_red_.resize(no3);
  V_red_.resize(no3);
  for( auto j = 0ul; j < no; ++j ) 
  for( auto i = 0ul; i < no; ++i )
  for( auto k = 0ul; k < no; ++k ) {
    G_red_[k + i*no + j*no2 ] = G_pqrs_[k*(no+1) + i*no2 + j*no3];
    V_red_[k + i*no + j*no2 ] = V      [k*(no+1) + i*no2 + j*no3];
  }

  // G2_red(i,j) = 0.5 * G(i,i,j,j)
  // V2_red(i,j) = V(i,i,j,j)
  G2_red_.resize(no2);
  V2_red_.resize(no2);
  for( auto j = 0ul; j < no; ++j ) 
  for( auto i = 0ul; i < no; ++i ) {
    G2_red_[i + j*no] = 0.5 * G_pqrs_[i*(no+1) + j*(no2+no3)];
    V2_red_[i + j*no] = V[i*(no+1) + j*(no2+no3)];
  }


}






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
double HamiltonianGenerator<N>::matrix_element_4( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex ) const {

  auto [o1,v1,o2,v2,sign] = doubles_sign_indices( bra, ket, ex );

  return sign * G_pqrs_[v1 + o1*norb_ + v2*norb2_ + o2*norb3_];

}

template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_4( spin_det_t bra, 
  spin_det_t ket, spin_det_t ex, double val, double* trdm ) {

  auto [o1,v1,o2,v2,sign] = doubles_sign_indices( bra, ket, ex );

  val *= sign * 0.5;
  trdm[ v1 + o1*norb_ + v2*norb2_ + o2*norb3_ ] += val;
  trdm[ v2 + o1*norb_ + v1*norb2_ + o2*norb3_ ] -= val;
  trdm[ v1 + o2*norb_ + v2*norb2_ + o1*norb3_ ] -= val;
  trdm[ v2 + o2*norb_ + v1*norb2_ + o1*norb3_ ] += val;

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
void HamiltonianGenerator<N>::rdm_contributions_22( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta,
  double val, double* trdm ) {

  auto [o1,v1,sign_a] = 
    single_excitation_sign_indices( bra_alpha, ket_alpha, ex_alpha );
  auto [o2,v2,sign_b] = 
    single_excitation_sign_indices( bra_beta,  ket_beta,  ex_beta  );
  auto sign = sign_a*sign_b;

  val *= sign * 0.5;
  trdm[ v1 + o1*norb_ + v2*norb2_ + o2*norb3_ ] += val;
  trdm[ v2 + o2*norb_ + v1*norb2_ + o1*norb3_ ] += val;

}







template <size_t N>
double HamiltonianGenerator<N>::matrix_element_2( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex,
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta ) const{

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
void HamiltonianGenerator<N>::rdm_contributions_2( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex,
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta,
  double val, double* ordm, double* trdm) {

  auto [o1,v1,sign] = single_excitation_sign_indices(bra,ket,ex);

  ordm[v1 + o1*norb_] += sign * val;
  
  val *= sign * 0.5;
  for( auto p : bra_occ_alpha ) {
    trdm[ v1 + o1*norb_ + p *norb2_ +  p*norb3_ ] += val;
    trdm[ p  + p *norb_ + v1*norb2_ + o1*norb3_ ] += val;
    trdm[ v1 + p *norb_ + p *norb2_ + o1*norb3_ ] -= val;
    trdm[ p  + o1*norb_ + v1*norb2_ +  p*norb3_ ] -= val;
  }

  for( auto p : bra_occ_beta ) {
    trdm[ v1 + o1*norb_ + p *norb2_ +  p*norb3_ ] += val;
    trdm[ p  + p *norb_ + v1*norb2_ + o1*norb3_ ] += val;
  }

}

template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_2( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex,
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta,
  double val, double* ordm) {

  auto [o1,v1,sign] = single_excitation_sign_indices(bra,ket,ex);

  ordm[v1 + o1*norb_] += sign * val;

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

template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_diag( 
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta,
  double val, double* ordm, double* trdm ) {

  // One-electron piece
  for( auto p : occ_alpha ) ordm[p*(norb_+1)] += val;
  for( auto p : occ_beta  ) ordm[p*(norb_+1)] += val;

  val *= 0.5;
  // Same-spin two-body term
  for( auto q : occ_alpha )
  for( auto p : occ_alpha ) {
    trdm[p + p*norb_ + q*norb2_ + q*norb3_] += val;
    trdm[p + q*norb_ + p*norb2_ + q*norb3_] -= val;
  }
  for( auto q : occ_beta )
  for( auto p : occ_beta ) {
    trdm[p + p*norb_ + q*norb2_ + q*norb3_] += val;
    trdm[p + q*norb_ + p*norb2_ + q*norb3_] -= val;
  }

  // Opposite-spin two-body term
  for( auto q : occ_beta  )
  for( auto p : occ_alpha ) {
    trdm[p + p*norb_ + q*norb2_ + q*norb3_] += val;
    trdm[q + q*norb_ + p*norb2_ + p*norb3_] += val;
  }

}

template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_diag( 
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta,
  double val, double* ordm ) {

  // Just one-electron piece
  for( auto p : occ_alpha ) ordm[p*(norb_+1)] += val;
  for( auto p : occ_beta  ) ordm[p*(norb_+1)] += val;

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
void HamiltonianGenerator<N>::rdm_contributions( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta, 
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta,
  double val, double* ordm, double* trdm) {

  const uint32_t ex_alpha_count = ex_alpha.count();
  const uint32_t ex_beta_count  = ex_beta.count();

  if( (ex_alpha_count + ex_beta_count) > 4 ) return;

  if( ex_alpha_count == 4 ) 
    rdm_contributions_4( bra_alpha, ket_alpha, ex_alpha, val, trdm );

  else if( ex_beta_count == 4 )
    rdm_contributions_4( bra_beta, ket_beta, ex_beta, val, trdm );

  else if( ex_alpha_count == 2 and ex_beta_count == 2 )
    rdm_contributions_22( bra_alpha, ket_alpha, ex_alpha,
      bra_beta, ket_beta, ex_beta, val, trdm );

  else if( ex_alpha_count == 2 )
    rdm_contributions_2( bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
      bra_occ_beta, val, ordm, trdm );

  else if( ex_beta_count == 2 )
    rdm_contributions_2( bra_beta, ket_beta, ex_beta, bra_occ_beta,
      bra_occ_alpha, val, ordm, trdm );

  else rdm_contributions_diag( bra_occ_alpha, bra_occ_beta, val, ordm, trdm );

}

template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta, 
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta,
  double val, double* ordm) {

  const uint32_t ex_alpha_count = ex_alpha.count();
  const uint32_t ex_beta_count  = ex_beta.count();

  if( (ex_alpha_count + ex_beta_count) > 2 ) return;

  if( ex_alpha_count == 2 )
    rdm_contributions_2( bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
      bra_occ_beta, val, ordm );

  else if( ex_beta_count == 2 )
    rdm_contributions_2( bra_beta, ket_beta, ex_beta, bra_occ_beta,
      bra_occ_alpha, val, ordm );

  else rdm_contributions_diag( bra_occ_alpha, bra_occ_beta, val, ordm );

}







template <size_t N>
void HamiltonianGenerator<N>::rotate_hamiltonian_ordm( const double* ordm ) {
  // SVD on ordm to get natural orbitals
  std::vector<double> natural_orbitals(ordm, ordm + norb2_);
  std::vector<double> S(norb_);
  lapack::gesvd( lapack::Job::OverwriteVec, lapack::Job::NoVec, norb_, norb_,
    natural_orbitals.data(), norb_, S.data(), NULL, 1, NULL, 1 );

#if 0
  {
    std::vector<double> tmp(norb2_);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
      norb_, norb_, norb_, 1., natural_orbitals.data(), norb_,
      natural_orbitals.data(), norb_, 0., tmp.data(), norb_ );
    for( auto i = 0; i < norb_; ++i ) tmp[i*(norb_+1)] -= 1.;
    std::cout << "MAX = " << *std::max_element(tmp.begin(),tmp.end(),
      []( auto x, auto y ){ return std::abs(x) < std::abs(y); } ) << std::endl;

    double max_diff = 0.;
    for( auto i = 0; i < norb_; ++i )
    for( auto j = i+1; j < norb_; ++j ) {
      max_diff = std::max( max_diff, 
        std::abs( ordm[i+j*norb_] - ordm[j+i*norb_]));
    }
    std::cout << "MAX = " << max_diff << std::endl;
  }
#endif

  std::vector<double> tmp( norb3_ * norb_), tmp2(norb3_*norb_);
  
  // Transform T
  // T <- N**H * T * N
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    norb_, norb_, norb_, 1., T_pq_, norb_, natural_orbitals.data(), norb_, 
    0., tmp.data(), norb_ );
  blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
    norb_, norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(), norb_, 
    0., T_pq_, norb_ );

  // Transorm V

  // 1st Quarter
  // (pj|kl) = N(i,p) (ij|kl) 
  // W(p,jkl) = N(i,p) * V(i,jkl)
  blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
    norb_, norb3_, norb_, 1., natural_orbitals.data(), norb_, V_pqrs_, norb_, 
    0., tmp.data(), norb_ );

  // 2nd Quarter
  // (pq|kl) = N(j,q) (pj|kl)
  // W_kl(p,q) = V_kl(p,j) N(j,q)
  for( auto kl = 0; kl < norb2_; ++kl ) {
    auto* V_kl = tmp.data()  + kl*norb2_;
    auto* W_kl = tmp2.data() + kl*norb2_;
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      norb_, norb_, norb_, 1., V_kl, norb_, natural_orbitals.data(), norb_,
      0., W_kl, norb_ );

  }

  // 3rd Quarter
  // (pq|rl) = N(k,r) (pq|kl)
  // W_l(pq,r) = V_l(pq,k) N(k,r)
  for( auto l = 0; l < norb_; ++l ) {
    auto* V_l = tmp2.data() + l*norb3_;
    auto* W_l = tmp.data()  + l*norb3_;
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      norb2_, norb_, norb_, 1., V_l, norb2_, natural_orbitals.data(), norb_,
      0., W_l, norb2_ );
  }

  // 4th Quarter
  // (pq|rs) = N(l,s) (pq|rl)
  // W(pqr,s) = V(pqr,l) N(l,s)
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    norb3_, norb_, norb_, 1., tmp.data(), norb3_, natural_orbitals.data(), norb_,
    0., V_pqrs_, norb3_ );

  // Regenerate intermediates
  generate_integral_intermediates( norb_, V_pqrs_ );
  
}

}
