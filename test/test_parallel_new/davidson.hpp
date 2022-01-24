#pragma once
#include <lobpcgxx/lobpcg.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>


void gram_schmidt( int64_t N, int64_t K, const double* V_old, int64_t LDV,
  double* V_new ) {

  std::vector<double> inner(K);
  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, 1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K );
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    N, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N );

  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, 1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K );
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    N, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N );

  auto nrm = blas::nrm2(N,V_new,1);
  blas::scal( N, 1./nrm, V_new, 1 );
}  


double davidson( int64_t max_m, const sparsexx::csr_matrix<double,int32_t>& A, double tol ) {

  int64_t N = A.n();
  std::cout << "N = " << N << ", MAX_M = " << max_m << std::endl;
  std::vector<double> V(N * (max_m+1)), AV(N * (max_m+1)), C((max_m+1)*(max_m+1)), 
    LAM(max_m+1);

  // Extract diagonal and setup guess
  auto D = extract_diagonal_elements( A );
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance( D.begin(), D_min );
  V[min_idx] = 1.;

  // Compute Initial A*V
  sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV.data(), N, V.data()+N);
  gram_schmidt(N, 1, V.data(), N, V.data()+N);



  for( size_t i = 1; i < max_m; ++i ) {

    // AV(:,i) = A * V(:,i)
    sparsexx::spblas::gespmbv(1, 1., A, V.data()+i*N, N, 0., AV.data()+i*N, N );

    const auto k = i + 1;

    // Rayleigh Ritz
    lobpcgxx::rayleigh_ritz( N, k, V.data(), N, AV.data(), N, LAM.data(),
      C.data(), k);

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    double* R = V.data() + (i+1)*N;
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      N, 1, k, 1., AV.data(), N, C.data(), k, 0., R, N );
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      N, 1, k, -LAM[0], V.data(), N, C.data(), k, 1., R, N );

    // Compute residual norm
    auto res_nrm = blas::nrm2( N, R, 1 );
    std::cout << std::scientific << std::setprecision(12);
    std::cout << i << ", " << LAM[0] << ", " << res_nrm << std::endl;
    if( res_nrm < tol ) return LAM[0];

    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R
    for( auto j = 0; j < N; ++j ) {
      R[j] = - R[j] / (D[j] - LAM[0]);
    }

    // Project new vector out form old vectors
    gram_schmidt(N, k, V.data(), N, R);

  } // Davidson iterations

  return LAM[0];
}

