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

#include "macis/gf/inn_prods.hpp"

typedef std::numeric_limits<double> dbl;

namespace macis {

extern "C" {
extern int dgeqrf_(int *, int *, double *, int *, double *, double *, int *,
                   int *);
extern int dorgqr_(int *, int *, int *, double *, int *, double *, double *,
                   int *, int *);
extern int dsbtrd_(char *, char *, int *, int *, double *, int *, double *,
                   double *, double *, int *, double *, int *);
extern int dsytrd_(char *, int *, double *, int *, double *, double *, double *,
                   double *, int *, int *);
extern int dorgtr_(char *, int *, double *, int *, double *, double *, int *,
                   int *);
extern int dsteqr_(char *, int *, double *, double *, double *, int *, double *,
                   int *);
}

bool QRdecomp(std::vector<std::vector<double> > &Q,
              std::vector<std::vector<double> > &R);

bool QRdecomp_tr(std::vector<std::vector<double> > &Q,
                 std::vector<std::vector<double> > &R);

bool GetEigsys(std::vector<std::vector<double> > &mat,
               std::vector<double> &eigvals,
               std::vector<std::vector<double> > &eigvecs);

bool GetEigsysBand(std::vector<std::vector<double> > &mat, int nSupDiag,
                   std::vector<double> &eigvals,
                   std::vector<std::vector<double> > &eigvecs);

/**
 * @brief Perform a band Lanczos calculation on the Hamiltonian operator H,
 * starting from vectors qs, for at most nLanIts iterations. The resulting
 * band-diagonal matrix Hamiltonian will be stored in bandH. Note that this
 * implementation does not account for deflations (i.e., pruning the span of the
 * qs for linear dependencies in higher powers of H).
 *
 * @param[in] const sparseexx::csr_matrix<double, int32_t> &H: Hamiltonian
 * oprator. Just needs to implement a matrix vector product.
 * @param[in] std::vector<std::vector<Cont> > &qs: Initial set of vetors to
 * perform the band Lanczos on. Deleted on exit.
 * @param[in] std::vector<std::vector<Cont> > &bandH: On exit, band-diagonal
 * Hamiltonian approximation.
 * @param[in] int &nLanIts: Number of Lanczos iterations to perform.
 * @param[in] double thres: Threshold determining when to ignore beta's for
 * being too small.
 * @param[in] bool print: If true, write intermediate results to file.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
template <class Cont>
void MyBandLan(
    const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> >
        &H,
    std::vector<std::vector<Cont> > &qs, std::vector<std::vector<Cont> > &bandH,
    int &nLanIts, double thres = 1.E-6, bool print = false) {
  // BAND LANCZOS ROUTINE. TAKES AS INPUT THE HAMILTONIAN H, INITIAL VECTORS qs
  // AND RETURNS THE BAND HAMILTONIAN bandH. IT PERFORMS nLanIts ITERATIONS,
  // STOPPING IF THE NORM OF ANY NEW KRYLOV VECTOR IS BELOW thres. IF LANCZOS IS
  // STOPPED PREMATURELY , nLanIts IS OVERWRITTEN WITH THE ACTUAL NUMBER OF
  // ITERATIONS! THE qs VECTOR IS ERASED AT THE END OF THE CALCULATION
  bandH.clear();
  bandH.resize(nLanIts, std::vector<Cont>(nLanIts, 0.));

  int nbands = qs.size();
  auto spmv_info = sparsexx::spblas::generate_spmv_comm_info(H);
  // MAKE SPACE FOR 2 * nbands VECTORS
  qs.resize(2 * nbands, std::vector<Cont>(qs[0].size(), 0.));
  std::vector<Cont> temp(qs[0].size(), 0.);
  if(print) {
    for(int i = 0; i < nbands; i++) {
      std::ofstream ofile("lanvec_" + std::to_string(i + 1) + ".dat",
                          std::ios::out);
      ofile.precision(dbl::max_digits10);
      for(size_t el = 0; el < qs[i].size(); el++)
        ofile << std::scientific << qs[i][el] << std::endl;
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
    sparsexx::spblas::pgespmv(1., H, qs[band_indx_i].data(), 0., temp.data(),
                              spmv_info);
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
#pragma omp parallel for
      for(size_t coeff = 0; coeff < temp.size(); coeff++)
        temp[coeff] -= bandH[it - 1][jt - 1] * qs[band_indx_j][coeff];
    }
    for(int jt = it; jt <= std::min(it + nbands - 1, nLanIts); jt++) {
      int band_indx_j = true_indx[jt];
      bandH[it - 1][jt - 1] = MyInnProd(temp, qs[band_indx_j]);
      bandH[jt - 1][it - 1] = bandH[it - 1][jt - 1];
#pragma omp parallel for
      for(size_t coeff = 0; coeff < temp.size(); coeff++)
        temp[coeff] -= bandH[it - 1][jt - 1] * qs[band_indx_j][coeff];
    }
    if(it + nbands <= nLanIts) {
      bandH[it - 1][it + nbands - 1] =
          std::sqrt(std::real(MyInnProd(temp, temp)));
      bandH[it + nbands - 1][it - 1] = bandH[it - 1][it + nbands - 1];
      true_indx[it + nbands] = next_indx;
      if(std::abs(bandH[it - 1][it + nbands - 1]) < thres) {
        std::cout
            << "BAND LANCZOS STOPPED PREMATURELY DUE TO SMALL NORM! NAMELY "
            << bandH[it - 1][it + nbands - 1]
            << ", STOPPED AT ITERATION: " << it << std::endl;
        nLanIts = it;
        for(int i = 0; i < nLanIts; i++) bandH[i].resize(nLanIts);
        bandH.resize(nLanIts);
        break;
#pragma omp parallel for
        for(size_t coeff = 0; coeff < temp.size(); coeff++)
          qs[true_indx[it + nbands]][coeff] = 0.;
        std::cout << "FOUND A ZERO VECTOR AT POSITION " << next_indx
                  << std::endl;
      } else {
#pragma omp parallel for
        for(size_t coeff = 0; coeff < temp.size(); coeff++)
          qs[true_indx[it + nbands]][coeff] =
              temp[coeff] / bandH[it - 1][it + nbands - 1];
        if(print) {
          std::ofstream ofile("lanvec_" + std::to_string(it + nbands) + ".dat",
                              std::ios::out);
          ofile.precision(dbl::max_digits10);
          for(size_t el = 0; el < qs[true_indx[it + nbands]].size(); el++)
            ofile << std::scientific << qs[true_indx[it + nbands]][el]
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
 * @param[in] std::vector<std::vector<double> > &vecs: Vectors for which to
 * compute the resolvent matrix elements.
 * @param[in] std::vector<std::complex<double> > &ws: Frequency grid over which
 * to evaluate the resolvent.
 * @param[out] std::vector<std::vector<std::vector<std::complex<double> > > >
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
    std::vector<std::vector<double> > &vecs,
    const std::vector<std::complex<double> > &ws,
    std::vector<std::vector<std::vector<std::complex<double> > > > &res,
    int nLanIts, double E0, bool ispart, bool print = false,
    bool saveGFmats = false);

}  // namespace macis
