/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

/**
 * @file bandlan.h++
 *
 * @brief Implements simple Band Lanczos routine.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
#pragma once
#include <assert.h>
#include <sys/stat.h>

#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <lobpcgxx/lobpcg.hpp>
#include <map>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <utility>

#include "macis/solvers/davidson.hpp"

namespace macis {

/**
 * @ brief Wrapper for QR decomposition in LAPACK, basically
 *         calling dgeqrf and dorgqr to evaluate the R and
 *         Q matrices, which are returned. Here, the input
 *         matrix has more rows than columns.
 *
 * @param[inout] std::vector<double> &Q: On input,
 *               matrix for which to evaluate the QR decomposition.
 *               On output, Q-matrix.
 * @param[out] std::vector<double> &R: On output, R
 *             matrix in the QR decomposition.
 * @param[in] int Qrows: Nr. of rows in matrix Q.
 * @param[in] int Qcols: Nr. of cols in matrix Q.
 *
 * @returns bool: Error code from LAPACK routines.
 *
 * @author Carlos Mejuto-Zaera
 * @date 25/04/2022
 */
bool QRdecomp(std::vector<double> &Q, std::vector<double> &R, int Qrows,
              int Qcols);

/**
 * @ brief Wrapper for QR decomposition in LAPACK, basically
 *         calling dgeqrf and dorgqr to evaluate the R and
 *         Q matrices, which are returned. Here, the input
 *         matrix has more columns than rows.
 *
 * @param[inout] std::vector<double> &Q: On input,
 *               matrix for which to evaluate the QR decomposition.
 *               On output, Q-matrix.
 * @param[out] std::vector<double> &R: On output, R
 *             matrix in the QR decomposition.
 * @param[in] int Qrows: Nr. of rows in Q.
 * @param[in] int Qcols: Nr. of cols in Q.
 *
 * @returns bool: Error code from LAPACK routines.
 *
 * @author Carlos Mejuto-Zaera
 * @date 25/04/2022
 */
bool QRdecomp_tr(std::vector<double> &Q, std::vector<double> &R, int Qrows,
                 int Qcols);

/**
 * @brief Wrapper to LAPACK routine to evaluate the eigenvectors
 *        and eigenvalues of the symmetric matrix mat.
 *
 * @param[inout] std::vector<double> &mat: Matrix for
 *               which to compute the eigenvalues/vectors. Erased
 *               during computation.
 * @param[out] std::vector<double> &eigvals: Eigenvalues, sorted from smallest
 *             to largest.
 * @param[out] std::vector<double> &eigvecs: Eigenvectors,
 *             stored as row vectors.
 * @param[in] int matsize: Nr. of rows/columns of the square matrix mat.
 *
 * @returns bool: Error code from LAPACK.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
bool GetEigsys(std::vector<double> &mat, std::vector<double> &eigvals,
               std::vector<double> &eigvecs, int matsize);

/**
 * @brief Wrapper to LAPACK routine to evaluate the eigenvectors
 *        and eigenvalues of the symmetric band matrix mat.
 *
 * @param[inout] std::vector<double> &mat: Matrix for
 *               which to compute the eigenvalues/vectors. Erased
 *               during computation.
 * @param[in] int nSupDiag: Nr. of bands.
 * @param[out] std::vector<double> &eigvals: Eigenvalues, sorted from smallest
 *             to largest.
 * @param[out] std::vector<double> &eigvecs: Eigenvectors,
 *             stored as row vectors.
 * @param[in] int matsize: Nr. of rows/cols of the square matrix mat.
 *
 * @returns bool: Error code from LAPACK.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
bool GetEigsysBand(std::vector<double> &mat, int nSupDiag,
                   std::vector<double> &eigvals, std::vector<double> &eigvecs,
                   int matsize);

/**
 * @brief Perform a band Lanczos calculation on the Hamiltonian operator H,
 * starting from vectors qs, for at most nLanIts iterations. The resulting
 * band-diagonal matrix Hamiltonian will be stored in bandH. Note that this
 * implementation does not account for deflations (i.e., pruning the span of the
 * qs for linear dependencies in higher powers of H).
 *
 * @param[in] const sparseexx::csr_matrix<double, int32_t> &H: Hamiltonian
 * oprator. Just needs to implement a matrix vector product.
 * @param[in] std::vector<Cont> &qs: Initial set of vetors to
 * perform the band Lanczos on. Deleted on exit.
 * @param[in] std::vector<Cont> &bandH: On exit, band-diagonal
 * Hamiltonian approximation.
 * @param[inout] int &nLanIts: Number of Lanczos iterations to perform.
 * @param[in] double thres: Threshold determining when to ignore beta's for
 * being too small.
 * @param[in] bool print: If true, write intermediate results to file.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
template <class Cont, class Functor>
void BandLan(const Functor &H, std::vector<Cont> &qs, std::vector<Cont> &bandH,
             int &nLanIts, int nbands, int N, double thres = 1.E-6,
             bool print = false) {
  // BAND LANCZOS ROUTINE. TAKES AS INPUT THE HAMILTONIAN H, INITIAL VECTORS qs
  // AND RETURNS THE BAND HAMILTONIAN bandH. IT PERFORMS nLanIts ITERATIONS,
  // STOPPING IF THE NORM OF ANY NEW KRYLOV VECTOR IS BELOW thres. IF LANCZOS IS
  // STOPPED PREMATURELY , nLanIts IS OVERWRITTEN WITH THE ACTUAL NUMBER OF
  // ITERATIONS! THE qs VECTOR IS ERASED AT THE END OF THE CALCULATION
  using dbl = std::numeric_limits<double>;
  bandH.clear();
  bandH.resize(nLanIts * nLanIts, 0.);

  auto VecNorm = [](const std::vector<Cont> &vec) -> double {
    return std::sqrt(
        std::real(blas::dot(vec.size(), vec.data(), 1, vec.data(), 1)));
  };

  // MAKE SPACE FOR 2 * nbands VECTORS
  qs.resize(2 * nbands * N);
  std::vector<Cont> temp(N, 0.);
  if(print) {
    for(int i = 0; i < nbands; i++) {
      std::ofstream ofile("lanvec_" + std::to_string(i + 1) + ".dat",
                          std::ios::out);
      ofile.precision(dbl::max_digits10);
      for(size_t el = 0; el < N; el++)
        ofile << std::scientific << qs[el + i * N] << std::endl;
      ofile.close();
    }
  }
  // DICTIONARY TO KEEP THE REAL INDICES OF THE KRYLOV VECTORS
  // THIS IS NECESSARY IN ORDER TO ONLY STORE 2* nbands OF THEM
  // AT ANY POINT IN TIME, PLUS ONE SCRATCH VECTOR TO BE DEFINED
  // INSIDE THE FOR LOOP
  std::vector<int> true_indx(nLanIts + 1);
  for(int i = 0; i < nbands; i++) true_indx[i + 1] = i;
  int next_indx = nbands;

  for(int it = 1; it <= nLanIts; it++) {
    int band_indx_i =
        true_indx[it];  // TO WHAT ELEMENT OF THE VECTOR SET DO WE APPLY THIS
    H.operator_action(1, 1., qs.data() + band_indx_i * N, N, 0., temp.data(),
                      N);
    if(print) {
      std::ofstream ofile("Htimes_lanvec_" + std::to_string(it) + ".dat",
                          std::ios::out);
      ofile.precision(dbl::max_digits10);
      for(size_t el = 0; el < temp.size(); el++)
        ofile << std::scientific << temp[el] << std::endl;
      ofile.close();
    }
    for(int jt = std::max(1, it - nbands); jt <= std::min(it - 1, nLanIts);
        jt++) {
      int band_indx_j = true_indx[jt];
      blas::axpy(N, -1. * bandH[(it - 1) * nLanIts + jt - 1],
                 qs.data() + N * band_indx_j, 1, temp.data(), 1);
    }
    for(int jt = it; jt <= std::min(it + nbands - 1, nLanIts); jt++) {
      int band_indx_j = true_indx[jt];
      bandH[(it - 1) * nLanIts + jt - 1] =
          blas::dot(N, temp.data(), 1, qs.data() + band_indx_j * N, 1);
      bandH[(jt - 1) * nLanIts + it - 1] = bandH[(it - 1) * nLanIts + jt - 1];
      blas::axpy(N, -1. * bandH[(it - 1) * nLanIts + jt - 1],
                 qs.data() + N * band_indx_j, 1, temp.data(), 1);
    }
    if(it + nbands <= nLanIts) {
      bandH[(it - 1) * nLanIts + it + nbands - 1] = VecNorm(temp);
      bandH[(it + nbands - 1) * nLanIts + it - 1] =
          bandH[(it - 1) * nLanIts + it + nbands - 1];
      true_indx[it + nbands] = next_indx;
      if(std::abs(bandH[(it - 1) * nLanIts + it + nbands - 1]) < thres) {
        //std::cout
        //    << "BAND LANCZOS STOPPED PREMATURELY DUE TO SMALL NORM! NAMELY "
        //    << bandH[(it - 1) * nLanIts + it + nbands - 1]
        //    << ", STOPPED AT ITERATION: " << it << std::endl;
        //nLanIts = it;
        //bandH.resize(nLanIts * nLanIts);
        //break;
        blas::scal(N, 0., qs.data() + true_indx[it + nbands] * N, 1);
        std::cout << "FOUND A ZERO VECTOR AT POSITION " << next_indx
                  << std::endl;
      } else {
        blas::copy(N, temp.data(), 1, qs.data() + true_indx[it + nbands] * N,
                   1);
        blas::scal(N, 1. / bandH[(it - 1) * nLanIts + it + nbands - 1],
                   qs.data() + true_indx[it + nbands] * N, 1);
        if(print) {
          std::ofstream ofile("lanvec_" + std::to_string(it + nbands) + ".dat",
                              std::ios::out);
          ofile.precision(dbl::max_digits10);
          for(size_t el = 0; el < N; el++)
            ofile << std::scientific << qs[true_indx[it + nbands] * N + el]
                  << std::endl;
          ofile.close();
        }
      }
      next_indx = (next_indx + 1 >= 2 * nbands) ? 0 : next_indx + 1;
    }
  }
  qs.clear();
}

/**
 * @brief Evaluates the expectation values of the resolvent of Hamiltonian H
 * along a frequency grid ws with respect to the vectors vecs, using the band
 * Lanczos algorithm.
 *
 * @param[in] const sparsex::dist_sparse_matrix<sparsexx::csr_matrix<double,
 * int32_t> > &H: Hamiltonian operator.
 * @param[in] std::vector<double> &vecs: Vectors for which to
 * compute the resolvent matrix elements in format res[freq.][iorb1 * norbs +
 * iorb2].
 * @param[in] std::vector<std::complex<double> > &ws: Frequency grid over which
 * to evaluate the resolvent.
 * @param[out] std::vector<std::vector<std::complex<double> > >
 * &res: On exit, resolvent elements.
 * @param[in] int nLanIts: Max number of iterations.
 * @param[in] double E0: Ground state energy, for shifting the resolvent.
 * @param[in] bool ispart: If true, computes resolvent for particle GF,
 * otherwise for hole GF.
 * @param[in] bool print: If true, write intermediate results to file.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
void BandResolvent(
    const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> >
        &H,
    std::vector<double> &vecs, const std::vector<std::complex<double> > &ws,
    std::vector<std::vector<std::complex<double> > > &res, int nLanIts,
    double E0, bool ispart, int nvecs, int len_vec, bool print = false,
    bool saveGFmats = false);

}  // namespace macis
