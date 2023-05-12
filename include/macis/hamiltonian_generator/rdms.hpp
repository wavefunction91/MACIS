#pragma once
#include <macis/hamiltonian_generator.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <blas.hh>
#include <lapack.hh>
#pragma GCC diagnostic pop

namespace macis {

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
  generate_integral_intermediates( V_pqrs_ );
  
}

}
