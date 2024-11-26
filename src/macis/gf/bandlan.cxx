/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include "macis/gf/bandlan.hpp"

#include <random>

inline bool is_file(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

namespace macis {
bool QRdecomp(std::vector<double> &Q, std::vector<double> &R, int Qrows,
              int Qcols) {
  // CALL LAPACK'S QR DECOMPOSITION ROUTINES.
  // INPUT: Q: INPUT MATRIX TO PERFORM A QR DECOMPOSITION FOR. MAY BE
  // RECTANGULAR,
  //           THE NUMBER OF COLUMNS WILL BE ALWAYS LESS THAN THE NUMBER OF
  //           ROWS.
  // OUTPUT: Q: Q MATRIX FROM THE DECOMPOSITION, OVERWRITES INPUT
  //         R: R MATRIX FROM THE DECOMPOSITION. UPPER DIAGONAL, SQUARE.
  //         return : TRUE FOR SUCCESS, FALSE OTHERWISE

  R.clear();
  // PREPARE VARIABLES TO CALL LAPACK
  int M = Qrows, N = Qcols;
  assert(M >= N);
  int LDA = M, INFO = 0;
  std::vector<double> A(M * N, 0.), TAU(N, 0.);

  // INITIALIZE A
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) A[i + j * M] = Q[i + j * M];
  // COMPUTE R MATRIX
  lapack::geqrf(M, N, A.data(), LDA, TAU.data());
  // SAVE THE R MATRIX
  R.resize(N * N);
  for(int i = 0; i < N; i++)
    for(int j = i; j < N; j++) R[i + j * N] = A[i + j * M];

  // NOW, COMPUTE THE ACTUAL Q MATRIX
  int K = N;
  lapack::orgqr(M, N, K, A.data(), LDA, TAU.data());
  // SAVE THE Q MATRIX
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) Q[i + j * M] = A[i + j * M];

  return true;
}

bool QRdecomp_tr(std::vector<double> &Q, std::vector<double> &R, int Qrows,
                 int Qcols) {
  // CALL LAPACK'S QR DECOMPOSITION ROUTINES.
  // INPUT: Q: INPUT MATRIX TO PERFORM A QR DECOMPOSITION FOR. MAY BE
  // RECTANGULAR,
  //           THE NUMBER OF COLUMNS WILL BE ALWAYS LESS THAN THE NUMBER OF
  //           ROWS.
  // OUTPUT: Q: Q MATRIX FROM THE DECOMPOSITION, OVERWRITES INPUT
  //         R: R MATRIX FROM THE DECOMPOSITION. UPPER DIAGONAL, SQUARE.
  //         return : TRUE FOR SUCCESS, FALSE OTHERWISE

  R.clear();
  // PREPARE VARIABLES TO CALL LAPACK
  int M = Qcols, N = Qrows;
  assert(M >= N);
  int LDA = M, INFO = 0;
  std::vector<double> A(M * N, 0.), TAU(N, 0.);

  // INITIALIZE A
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) A[i + j * M] = Q[i + j * M];

  // EVALUATE R MATRIX
  lapack::geqrf(M, N, A.data(), LDA, TAU.data());
  // SAVE THE R MATRIX
  R.resize(N * N);
  for(int i = 0; i < N; i++)
    for(int j = i; j < N; j++) R[i + j * N] = A[i + j * M];

  // NOW, COMPUTE THE ACTUAL Q MATRIX
  int K = N;
  lapack::orgqr(M, N, K, A.data(), LDA, TAU.data());
  // SAVE THE Q MATRIX
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) Q[i + j * M] = A[i + j * M];

  return true;
}

bool GetEigsys(std::vector<double> &mat, std::vector<double> &eigvals,
               std::vector<double> &eigvecs, int matsize) {
  // COMPUTES THE EIGENVALUES AND EIGENVECTORS OF THE SYMMETRIC MATRIX mat BY
  // CALLING LAPACK. WE ASSUME THE UPPER TRIANGULAR PART OF A IS STORED. FIRST,
  // IT BRINGS THE MATRIX INTO TRIANGULAR FORM, THEN COMPUTES THE EIGENVALUES
  // AND EIGENVECTORS. THESE ARE STORED IN eigvals AND eigvecs RESPECTIVELY. THE
  // MATRIX mat IS ERASED DURING COMPUTATION
  eigvals.clear();
  eigvecs.clear();
  // PREPARE VARIABLES FOR LAPACK
  lapack::Uplo UPLO = lapack::Uplo::Upper;
  lapack::Job JOBZ = lapack::Job::Vec;
  int N = matsize, LDA = matsize;
  std::vector<double> A(N * N, 0.), D(N, 0.);

  // INITIALIZE A
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++) A[i + j * N] = mat[i + j * N];
  mat.clear();
  // COMPUTE EIGENVALUES AND EIGENVECTORS
  lapack::heev_2stage(JOBZ, UPLO, N, A.data(), LDA, D.data());

  // NOW, STORE THE EIGENVALUES AND EIGENVECTORS
  eigvals.resize(N);
  for(int i = 0; i < N; i++) eigvals[i] = D[i];
  eigvecs.resize(N * N);
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++) eigvecs[i + j * N] = A[j + i * N];

  return true;
}

bool GetEigsysBand(std::vector<double> &mat, int nSupDiag,
                   std::vector<double> &eigvals, std::vector<double> &eigvecs,
                   int matsize) {
  // COMPUTES THE EIGENVALUES AND EIGENVECTORS OF THE SYMMETRIC BAND MATRIX mat
  // BY CALLING LAPACK. WE ASSUME THE UPPER TRIANGULAR PART OF A IS STORED.
  // FIRST, IT BRINGS THE MATRIX INTO TRIANGULAR FORM, THEN COMPUTES THE
  // EIGENVALUES AND EIGENVECTORS. THESE ARE STORED IN eigvals AND eigvecs
  // RESPECTIVELY. THE MATRIX mat IS ERASED DURING COMPUTATION
  eigvals.clear();
  eigvecs.clear();
  // PREPARE VARIABLES FOR LAPACK
  lapack::Uplo UPLO = lapack::Uplo::Upper;
  lapack::Job VECT = lapack::Job::Vec;
  lapack::Job COMPZ = lapack::Job::UpdateVec;
  int N = matsize, LDQ = matsize, LDAB = nSupDiag + 1;
  std::vector<double> AB((nSupDiag + 1) * N, 0.);
  std::vector<double> D(N, 0.), E(N - 1, 0.), Q(N * N, 0.);

  // INITIALIZE A
  for(int j = 0; j < N; j++)
    for(int i = std::max(0, j - nSupDiag); i <= j; i++)
      AB[nSupDiag + i - j + j * (nSupDiag + 1)] = mat[i + j * N];
  mat.clear();

  // TRANSFORM THE MATRIX TO TRIDIAGONAL FORM
  // NOW, TRANSFORM MATRIX TO TRIDIAGONAL FORM
  lapack::sbtrd(VECT, UPLO, N, nSupDiag, AB.data(), LDAB, D.data(), E.data(),
                Q.data(), LDQ);
  AB.clear();

  // FINALLY, COMPUTE THE EIGENVALUES AND EIGENVECTORS!
  lapack::steqr(COMPZ, N, D.data(), E.data(), Q.data(), LDQ);

  // NOW, STORE THE EIGENVALUES AND EIGENVECTORS
  eigvals.resize(N);
  for(int i = 0; i < N; i++) eigvals[i] = D[i];
  D.clear();
  eigvecs.resize(N * N);
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++) eigvecs[i + j * N] = Q[j + i * N];

  return true;
}

void BandResolvent(
    const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> >
        &H,
    std::vector<double> &vecs, const std::vector<std::complex<double> > &ws,
    std::vector<std::vector<std::complex<double> > > &res, int nLanIts,
    double E0, bool ispart, int nvecs, int len_vec, bool print,
    bool saveGFmats) {
  // COMPUTES THE RESOLVENT (ws - H)^-1 IN MATRIX FORM FOR THE "BASIS" GIVEN BY
  // THE vecs VECTORS AND THE FREQUENCY GRID IN ws. USES THE BAND LANCZOS
  // ALGORITHM. IT GETS STORED IN res.
  using dbl = std::numeric_limits<double>;
  res.clear();
  std::cout << "RESOLVENT ROUTINE: ";
  res.resize(ws.size(), std::vector<std::complex<double> >(
                            nvecs * nvecs, std::complex<double>(0., 0.)));

  // FIRST, COMPUTE QR DECOMPOSITION OF THE "BASIS" VECTORS vecs, NECESSARY FOR
  // LANCZOS
  std::vector<double> R;
  std::cout << "QR DECOMPOSITION ...";
  bool worked = QRdecomp_tr(vecs, R, nvecs, len_vec);
  if(not worked) {
    std::cout << "QR DECOMPOSITION FAILED!!" << std::endl;
    return;
  }
  std::cout << "DONE! ";

  if(print) {
    std::ofstream ofile("QRresVecs.dat", std::ios::out);
    ofile.precision(dbl::max_digits10);
    ofile << "RESULT OF QR DECOMPOSITION: " << std::endl;
    ofile << " New Vectors: " << std::endl;
    for(int i = 0; i < len_vec; i++) {
      for(int j = 0; j < nvecs; j++)
        ofile << std::scientific << vecs[j * len_vec + i] << "    ";
      ofile << std::endl;
    }
    ofile.close();
    ofile.clear();
    ofile.open("QRresRmat.dat", std::ios::out);
    ofile << " R Matrix: " << std::endl;
    for(int i = 0; i < nvecs; i++) {
      for(int j = 0; j < nvecs; j++)
        ofile << std::scientific << R[i + j * nvecs] << "  ";
      ofile << std::endl;
    }
    ofile.close();
  }

  // NEXT, COMPUTE THE BAND LANCZOS
  std::vector<double> bandH;
  std::cout << "BAND LANCZOS ...";
  SparseMatrixOperator Hop(H);
  int nbands = nvecs;
  BandLan<double>(Hop, vecs, bandH, nLanIts, nbands, len_vec, 1.E-6, print);
  std::cout << "DONE! ";

  if(print) {
    std::ofstream ofile("BLH.dat", std::ios::out);
    ofile.precision(dbl::max_digits10);
    ofile << "RESULT OF BAND LANCZOS: " << std::endl;
    ofile << " bandH Matrix: " << std::endl;
    for(int i = 0; i < nLanIts; i++) {
      for(int j = 0; j < nLanIts; j++)
        ofile << std::scientific << bandH[i * nLanIts + j] << "  ";
      ofile << std::endl;
    }
    ofile.close();
  }

  if(nvecs == 1) {
    // ONLY ONE BAND. DIAGONAL GREEN'S FUNCTION ELEMENT.
    // COMPUTE THROUGH CONTINUED FRACTION.
    std::cout << "COMPUTING GF AS CONTINUED FRACTION...";
    std::vector<double> alphas(nLanIts, 0.), betas(nLanIts, 0.);
    for(int i = 0; i < nLanIts; i++)
      alphas[i] =
          ispart ? E0 - bandH[i * nLanIts + i] : bandH[i * nLanIts + i] - E0;
    for(int i = 0; i < nLanIts - 1; i++)
      betas[i + 1] =
          ispart ? -bandH[i * nLanIts + i + 1] : bandH[i * nLanIts + i + 1];
    betas[0] = R[0];
#pragma omp parallel for
    for(int indx_w = 0; indx_w < ws.size(); indx_w++) {
      res[indx_w][0] =
          betas.back() * betas.back() / (ws[indx_w] + alphas.back());
      for(int i = betas.size() - 2; i >= 0; i--)
        res[indx_w][0] =
            betas[i] * betas[i] / (ws[indx_w] + alphas[i] - res[indx_w][0]);
    }
  } else {
    // NEXT, COMPUTE THE EIGENVALUES AND EIGENVECTORS OF THE BAND DIAGONAL
    // KRYLOV HAMILTONIAN
    std::vector<double> eigvals;
    std::vector<double> eigvecs;
    std::cout << "COMPUTING EIGENVALES ...";
    if(ispart)
      for(int rr = 0; rr < nLanIts; rr++) {
        bandH[rr * nLanIts + rr] = E0 - bandH[rr * nLanIts + rr];
        for(int cc = rr + 1; cc < nLanIts; cc++) {
          bandH[rr * nLanIts + cc] = -bandH[rr * nLanIts + cc];
          bandH[cc * nLanIts + rr] = -bandH[cc * nLanIts + rr];
        }
      }
    else
      for(int rr = 0; rr < nLanIts; rr++)
        bandH[rr * nLanIts + rr] = bandH[rr * nLanIts + rr] - E0;

    GetEigsysBand(bandH, std::min(size_t(nvecs), size_t(nLanIts - 1)), eigvals,
                  eigvecs, nLanIts);
    if(print) {
      std::ofstream ofile("BLEigs.dat", std::ios::out);
      ofile.precision(dbl::max_digits10);
      ofile << "RESULT OF EIGENVALUE CALCULATION: " << std::endl;
      ofile << " Eigvals: [";
      for(int i = 0; i < eigvals.size(); i++)
        ofile << std::scientific << eigvals[i] << ", ";
      ofile << std::endl;
      ofile << "Eigvecs: " << std::endl;
      for(int i = 0; i < nLanIts; i++) {
        for(int j = 0; j < nLanIts; j++)
          ofile << std::scientific << eigvecs[i + j * nLanIts] << "  ";
        ofile << std::endl;
      }
      ofile.close();
    }
    std::cout << "DONE! ";
    // FINALLY, COMPUTE S-MATRIX AND RESOLVENT
    std::vector<double> S(nLanIts * nvecs, 0.);
    std::cout << " COMPUTING S MATRIX ...";
    for(int i_lan = 0; i_lan < nLanIts; i_lan++) {
      for(int j_n = 0; j_n < nvecs; j_n++) {
        for(int l = 0; l < nvecs; l++)
          S[i_lan * nvecs + j_n] +=
              eigvecs[i_lan + l * nLanIts] * R[l + j_n * nvecs];
      }
    }
    if(saveGFmats) {
      std::cout << "WRITING S MATRIX AND BAND-LANCZOS EIGENVALUES TO FILE!"
                << std::endl;
      std::string fprefix = ispart ? "particle" : "hole";
      std::ofstream ofile(fprefix + "_S.mat", std::ios::out);
      ofile.precision(dbl::max_digits10);
      for(int i_lan = 0; i_lan < nLanIts; i_lan++) {
        for(int k = 0; k < nvecs; k++)
          ofile << std::scientific << S[i_lan * nvecs + k] << " ";
        ofile << std::endl;
      }
      ofile.close();
      ofile.open(fprefix + "_BLevals.dat", std::ios::out);
      ofile.precision(dbl::max_digits10);
      for(int i_lan = 0; i_lan < nLanIts; i_lan++)
        ofile << std::scientific << eigvals[i_lan] << std::endl;
      ofile.close();
    }
    std::cout << "DONE! COMPUTING RESOLVENT ...";
    if(ispart) {
#pragma omp parallel for
      for(int iw = 0; iw < ws.size(); iw++) {
        for(int k = 0; k < nvecs; k++) {
          for(int l = 0; l < nvecs; l++) {
            for(int i_lan = 0; i_lan < nLanIts; i_lan++) {
              res[iw][k * nvecs + l] += S[i_lan * nvecs + k] * 1. /
                                        (ws[iw] + eigvals[i_lan]) *
                                        S[i_lan * nvecs + l];
            }
          }
        }
      }
    } else {
#pragma omp parallel for
      for(int iw = 0; iw < ws.size(); iw++) {
        for(int k = 0; k < nvecs; k++) {
          for(int l = 0; l < nvecs; l++) {
            for(int i_lan = 0; i_lan < nLanIts; i_lan++) {
              res[iw][l * nvecs + k] += S[i_lan * nvecs + k] * 1. /
                                        (ws[iw] + eigvals[i_lan]) *
                                        S[i_lan * nvecs + l];
            }
          }
        }
      }
    }
  }
  std::cout << "DONE!" << std::endl;
}

}  // namespace macis
