#pragma once
#include <asci/hamiltonian_generator.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <blas.hh>
#include <lapack.hh>
#pragma GCC diagnostic pop

namespace asci {

#if 0
template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta, 
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta,
  double val, matrix_span_t ordm, rank4_span_t trdm) {

  const uint32_t ex_alpha_count = ex_alpha.count();
  const uint32_t ex_beta_count  = ex_beta.count();

  if( (ex_alpha_count + ex_beta_count) > 4 ) return;

  const auto trdm_ptr = trdm.data_handle();
  if( ex_alpha_count == 4 and trdm_ptr) 
    rdm_contributions_4( bra_alpha, ket_alpha, ex_alpha, val, trdm );

  else if( ex_beta_count == 4 and trdm_ptr)
    rdm_contributions_4( bra_beta, ket_beta, ex_beta, val, trdm );

  else if( ex_alpha_count == 2 and ex_beta_count == 2 and trdm_ptr )
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
void HamiltonianGenerator<N>::rdm_contributions_4( spin_det_t bra, 
  spin_det_t ket, spin_det_t ex, double val, rank4_span_t trdm ) {

  auto [o1,v1,o2,v2,sign] = doubles_sign_indices( bra, ket, ex );

  val *= sign * 0.5;
  trdm(v1, o1, v2, o2) += val;
  trdm(v2, o1, v1, o2) -= val;
  trdm(v1, o2, v2, o1) -= val;
  trdm(v2, o2, v1, o1) += val;

}










template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_22( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta,
  double val, rank4_span_t trdm ) {

  auto [o1,v1,sign_a] = 
    single_excitation_sign_indices( bra_alpha, ket_alpha, ex_alpha );
  auto [o2,v2,sign_b] = 
    single_excitation_sign_indices( bra_beta,  ket_beta,  ex_beta  );
  auto sign = sign_a*sign_b;

  val *= sign * 0.5;
  trdm(v1, o1, v2, o2) += val;
  trdm(v2, o2, v1, o1) += val;

}








template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_2( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex,
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta,
  double val, matrix_span_t ordm, rank4_span_t trdm) {

  auto [o1,v1,sign] = single_excitation_sign_indices(bra,ket,ex);

  ordm(v1, o1) += sign * val;
  
  if(trdm.data_handle()) {
    val *= sign * 0.5;
    for( auto p : bra_occ_alpha ) {
      trdm( v1, o1, p ,  p) += val;
      trdm( p , p , v1, o1) += val;
      trdm( v1, p , p , o1) -= val;
      trdm( p , o1, v1,  p) -= val;
    }

    for( auto p : bra_occ_beta ) {
      trdm(v1, o1, p ,  p) += val;
      trdm(p , p , v1, o1) += val;
    }
  }

}








template <size_t N>
void HamiltonianGenerator<N>::rdm_contributions_diag( 
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta,
  double val, matrix_span_t ordm, rank4_span_t trdm ) {

  // One-electron piece
  for( auto p : occ_alpha ) ordm(p,p) += val;
  for( auto p : occ_beta  ) ordm(p,p) += val;

  if(trdm.data_handle()) {
    val *= 0.5;
    // Same-spin two-body term
    for( auto q : occ_alpha )
    for( auto p : occ_alpha ) {
      trdm(p, p, q, q) += val;
      trdm(p, q, p, q) -= val;
    }
    for( auto q : occ_beta )
    for( auto p : occ_beta ) {
      trdm(p, p, q, q) += val;
      trdm(p, q, p, q) -= val;
    }

    // Opposite-spin two-body term
    for( auto q : occ_beta  )
    for( auto p : occ_alpha ) {
      trdm(p, p, q, q) += val;
      trdm(q, q, p, p) += val;
    }
  }

}
#endif






















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
  auto* T_pq_ptr = T_pq_.data_handle();
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    norb_, norb_, norb_, 1., T_pq_ptr, norb_, natural_orbitals.data(), norb_, 
    0., tmp.data(), norb_ );
  blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
    norb_, norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(), norb_, 
    0., T_pq_ptr, norb_ );

  // Transorm V

  // 1st Quarter
  // (pj|kl) = N(i,p) (ij|kl) 
  // W(p,jkl) = N(i,p) * V(i,jkl)
  blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
    norb_, norb3_, norb_, 1., natural_orbitals.data(), norb_, 
    V_pqrs_.data_handle(), norb_, 0., tmp.data(), norb_ );

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
    0., V_pqrs_.data_handle(), norb3_ );

  // Regenerate intermediates
  generate_integral_intermediates( norb_, V_pqrs_ );
  
}

}
