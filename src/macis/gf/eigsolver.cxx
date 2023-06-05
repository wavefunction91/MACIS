/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include "macis/gf/eigsolver.hpp"

/***Written by Carlos Mejuto Zaera***/

namespace macis {

void Hste_v(const std::vector<double> &alphas, const std::vector<double> &betas,
            Eigen::VectorXd &eigvals, eigMatD &eigvecs) {
  /*
   * COMPUTES THE EIGENVALUES AND EIGENVECTORS OF A TRIDIAGONAL, SYMMETRIC
   * MATRIX A USING LAPACK.
   */
  eigvals.resize(alphas.size());
  eigvecs.resize(alphas.size(), alphas.size());
  // INITIALIZE VARIABLES
  // COMPUTE EIGENVALUES AND EIGENVECTORS OF THE TRIDIAGONAL MATRIX
  lapack::Job JOBZ = lapack::Job::Vec;
  int N = alphas.size(), LDZ = N;  // SIZES
  std::vector<double> D, E;        // DIAGONAL AND SUB-DIAGONAL ELEMENTS
  std::vector<double> Z;           // EIGENVECTORS
  // INITIALIZE MATRIX
  D.resize(N);
  for(int64_t i = 0; i < N; i++) D[i] = alphas[i];
  E.resize(N - 1);
  for(int64_t i = 1; i < N; i++) E[i - 1] = betas[i];
  // ALLOCATE MEMORY
  Z.resize(N * LDZ);

  // ACTUAL EIGENVALUE CALCULATION
  lapack::steqr(JOBZ, N, D.data(), E.data(), Z.data(), LDZ);
  // SAVE EIGENVECTORS
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) eigvecs(i, j) = Z[i + j * N];
  }
  Z.clear();
  // SAVE EIGENVALUES
  for(int i = 0; i < N; i++) eigvals(i) = D[i];
}

void Hsyev(const eigMatD &H, Eigen::VectorXd &eigvals, eigMatD &eigvecs) {
  /*
   * COMPUTES THE EIGENVALUES AND EIGENVECTORS OF A SYMMETRIC MATRIX A USING
   * LAPACK.
   */
  eigvals.resize(H.rows());
  eigvecs.resize(H.rows(), H.rows());
  // INITIALIZE VARIABLES
  // COMPUTE EIGENVALUES AND EIGENVECTORS, H IS
  // STORED IN THE UPPER TRIANGLE
  lapack::Job JOBZ = lapack::Job::Vec;
  lapack::Uplo UPLO = lapack::Uplo::Upper;
  int N = H.rows(), LDA = N;  // SIZES
  std::vector<double> A;      // MATRIX AND WORKSPACE
  std::vector<double> W;      // EIGENVALUES AND WORKSPACE
  // INITIALIZE MATRIX
  A.resize(N * N);
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) A[i + j * N] = H(i, j);
  }
  // ALLOCATE MEMORY
  W.resize(N);

  // ACTUAL EIGENVALUE CALCULATION
  lapack::syev(JOBZ, UPLO, N, A.data(), LDA, W.data());
  // SAVE EIGENVECTORS
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) eigvecs(i, N - 1 - j) = A[i + j * N];
  }
  A.clear();
  // SAVE EIGENVALUES
  for(int i = 0; i < N; i++) eigvals(N - 1 - i) = W[i];
}

}  // namespace macis
