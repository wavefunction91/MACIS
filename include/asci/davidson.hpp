#pragma once
#include <lobpcgxx/lobpcg.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <random>
#include <iomanip>
#include <iostream>


namespace asci {

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


#if 0
template <typename Functor>
double davidson( size_t N, int64_t max_m, const Functor& op, 
  double tol, double* X = NULL, bool print = false ) {

  if(!X) throw std::runtime_error("Davidson: No Guess Provided");

  if(print) std::cout << "N = " << N << ", MAX_M = " << max_m << std::endl;

  std::vector<double> V(N * (max_m+1)), AV(N * (max_m+1)), C((max_m+1)*(max_m+1)), 
    LAM(max_m+1);

  // Extract diagonal and setup guess
  auto D = extract_diagonal_elements( A );
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance( D.begin(), D_min );
  V[min_idx] = 1.;

  // Compute Initial A*V
  //sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N);
  op.operator_action(1, 1., V.data(), N, 0., AV.data(), N);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV.data(), N, V.data()+N);
  gram_schmidt(N, 1, V.data(), N, V.data()+N);



  for( int64_t i = 1; i < max_m; ++i ) {

    // AV(:,i) = A * V(:,i)
    //sparsexx::spblas::gespmbv(1, 1., A, V.data()+i*N, N, 0., AV.data()+i*N, N );
    op.operator_action(1, 1., V.data() + i*N, N, 0., AV.data() + i*N, N);

    const auto k = i + 1;

    // Rayleigh Ritz
    lobpcgxx::rayleigh_ritz( N, k, V.data(), N, AV.data(), N, LAM.data(),
      C.data(), k);

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    double* R = V.data() + (i+1)*N;

    if(X) {
      // If X is non-null, save the Ritz vector 

      // X = V*C
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N, 1, k, 1., V.data(), N, C.data(), k, 0., X, N );

      // R = X
      std::copy_n( X, N, R );

      // R = (AV - LAM[0]*V)*C = AV*C - LAM[0]*X = AV*C - LAM[0]*R
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N, 1, k, 1., AV.data(), N, C.data(), k, -LAM[0], R, N );

    } else {
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N, 1, k, 1., AV.data(), N, C.data(), k, 0., R, N );
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N, 1, k, -LAM[0], V.data(), N, C.data(), k, 1., R, N );
    }

    // Compute residual norm
    auto res_nrm = blas::nrm2( N, R, 1 );
    if(print) {
      std::cout << std::scientific << std::setprecision(12);
      std::cout << std::setw(4) << i << ", " << LAM[0] << ", " << res_nrm 
                << std::endl;
    }
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
#endif












void p_gram_schmidt( int64_t N_local, int64_t K, const double* V_old, int64_t LDV,
  double* V_new, MPI_Comm comm ) {

  std::vector<double> inner(K);
  // Compute local V_old**H * V_new
  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, 1, N_local, 1., V_old, LDV, V_new, N_local, 0., inner.data(), K );

  // Reduce inner product
  MPI_Allreduce( MPI_IN_PLACE, inner.data(), K, MPI_DOUBLE, MPI_SUM, comm );

  // Project locally
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N_local );

  // Repeat
  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, 1, N_local, 1., V_old, LDV, V_new, N_local, 0., inner.data(), K );
  MPI_Allreduce( MPI_IN_PLACE, inner.data(), K, MPI_DOUBLE, MPI_SUM, comm );
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N_local );


  // Normalize
  double dot = blas::dot( N_local, V_new, 1, V_new, 1 );
  MPI_Allreduce( MPI_IN_PLACE, &dot, 1, MPI_DOUBLE, MPI_SUM, comm );
  double nrm = std::sqrt(dot);
  blas::scal(N_local, 1./nrm, V_new, 1);

}  

void p_rayleigh_ritz( int64_t N_local, int64_t K, const double* X, int64_t LDX,
  const double* AX, int64_t LDAX, double* W, double* C, int64_t LDC, 
  MPI_Comm comm ) {

  int world_rank; MPI_Comm_rank( comm, &world_rank );

  // Compute Local inner product
  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, K, N_local, 1., X, LDX, AX, LDAX, 0., C, LDC );

  // Reduce result
  if( LDC != K ) throw std::runtime_error("DIE DIE DIE RR");
  MPI_Allreduce( MPI_IN_PLACE, C, K*K, MPI_DOUBLE, MPI_SUM, comm );

  // Do local diagonalization on rank-0
  if( !world_rank ) {
    lapack::syev( lapack::Job::Vec, lapack::Uplo::Lower, K, C, LDC, W );
  }

  // Broadcast results
  MPI_Bcast( W, K, MPI_DOUBLE, 0, comm );
  MPI_Bcast( C, K*K, MPI_DOUBLE, 0, comm );

}


template <typename Functor>
double p_davidson( int64_t N_local, int64_t max_m, const Functor& op, 
  const double* D_local, double tol, double* X_local, MPI_Comm comm ) {

  if(N_local and !X_local) throw std::runtime_error("Davidson: No Guess Provided");

  int world_rank, world_size;
  MPI_Comm_rank( comm, &world_rank );
  MPI_Comm_size( comm, &world_size );

  bool print = false;
  if( world_rank == 0 and print ) {
    std::cout << "\nDavidson Eigensolver" << std::endl
              << "  MAX_M = " << max_m << std::endl
              << "  RTOL  = " << tol   << std::endl
              << std::endl
              << "Iterations:" << std::endl;
  }

  // Allocations
  std::vector<double> V_local( N_local * (max_m+1) ), AV_local( N_local * (max_m+1) ), 
    C( (max_m+1)*(max_m+1) ), LAM(max_m+1);

  // Copy over guess
  std::copy_n(X_local, N_local, V_local.begin());

  // Compute initial A*V
  //sparsexx::spblas::pgespmv( 1., A, V_local.data(), 0., AV_local.data(), spmv_info );
  op.operator_action(1, 1., V_local.data(), N_local, 0., AV_local.data(), N_local);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV_local.data(), N_local, V_local.data()+N_local);
  p_gram_schmidt(N_local, 1, V_local.data(), N_local, V_local.data()+N_local, comm);

  for( int64_t i = 1; i < max_m; ++i ) {

    // AV(:,i) = A * V(:,i)
    //sparsexx::spblas::pgespmv( 1., A, V_local.data()+i*N_local, 
    //  0., AV_local.data()+i*N_local, spmv_info );
    op.operator_action(1, 1., V_local.data() + i*N_local, N_local, 
      0., AV_local.data() + i*N_local, N_local);
    

    const auto k = i + 1;

    // Rayleigh Ritz
    p_rayleigh_ritz( N_local, k, V_local.data(), N_local, AV_local.data(), 
      N_local, LAM.data(), C.data(), k, comm);

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    double* R_local = V_local.data() + (i+1)*N_local;

    if( X_local ) {
      // If X_local is non-null, save Ritz vector

      // X = V*C
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N_local, 1, k, 1., V_local.data(), N_local, C.data(), k, 0., 
        X_local, N_local );

      // R = X
      std::copy_n( X_local, N_local, R_local );

      // R = (AV - LAM[0]*V)*C = AV*C - LAM[0]*X = AV*C - LAM[0]*R
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N_local, 1, k, 1., AV_local.data(), N_local, C.data(), k, -LAM[0], 
        R_local, N_local );

    } else {

      // Else just compute R Directly 
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N_local, 1, k, 1., AV_local.data(), N_local, C.data(), k, 0., 
        R_local, N_local );
      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        N_local, 1, k, -LAM[0], V_local.data(), N_local, C.data(), k, 1., 
        R_local, N_local );

    }

    // Compute residual norm
    auto res_dot = blas::dot( N_local, R_local, 1, R_local, 1 );
    MPI_Allreduce(MPI_IN_PLACE, &res_dot, 1, MPI_DOUBLE, MPI_SUM, comm );
    auto res_nrm = std::sqrt(res_dot);
    if( print and world_rank == 0 ) {
      std::cout << std::scientific << std::setprecision(12);
      std::cout << std::setw(4) << i << ", " << LAM[0] << ", " << res_nrm << std::endl;
    }
    if( res_nrm < tol ) return LAM[0];


    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R
    for( auto j = 0; j < N_local; ++j ) {
      R_local[j] = - R_local[j] / (D_local[j] - LAM[0]);
    }

    // Project new vector out form old vectors
    p_gram_schmidt(N_local, k, V_local.data(), N_local, R_local, comm);

  } // Davidson iterations

  return LAM[0];

}


} // namespace asci
