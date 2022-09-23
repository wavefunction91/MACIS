/**
 * @file lobpcg_call.h++
 *
 * @brief Implements simple call to LOBPCG to get the lowest few
 *        eigenstates of a given matrix.
 *
 * @author Carlos Mejuto Zaera
 * @date 19/09/2022
 */
#ifndef __CMZ_CALL_LOBPCG__
#define __CMZ_CALL_LOBPCG__
#include "cmz_ed/utils.h++"
#include <assert.h>
#include <map>
#include <complex>
#include <iomanip>
#include <limits>
#include <fstream>
#include <utility>
#include <sys/stat.h>
#include <lobpcgxx/lobpcg.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>

namespace cmz
{
  namespace ed
  {

    /**
     * @brief Perform a band Lanczos calculation on the Hamiltonian operator H, starting from vectors qs, for at most nLanIts
     *        iterations. Returns the first len(qs) eigenvalues, converged to some accuracy. Note that this implementation
     *        does not account for deflations (i.e., pruning the span of the qs for linear dependencies in higher powers of H).
     *
     * @param[in] const sparseexx::csr_matrix<double, int32_t> &H: Hamiltonian oprator. Just needs to implement a matrix vector product.
     * @param[in] std::vector<std::vector<double> > &qs: Initial set of vetors to perform the band Lanczos on. Deleted on exit.
     * @param[out] std::vector<double> &evals: Lowest len(qs) eigenvalues.
     * @param[out] std::vector<std::vector<double> > &evecs: Lowest len(qs) eigenvectors, in the Krylov basis.
     * @param[in] int &nLanIts: Number of Lanczos iterations to perform.
     * @param[in] double tol: Target tolerance for the eigenvalue convergence.
     * @param[in] double thres: Threshold determining when to ignore beta's for being too small.
     * @param[in] bool print: If true, write intermediate results to file.
     *
     * @author Carlos Mejuto Zaera
     * @date 25/04/2022
     */ 
    void LobpcgGS(
      const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> > &H, 
      size_t dimH,
      size_t nstates,
      std::vector<double> &evals,
      std::vector<double> &X,
      int maxIts, 
      double tol  = 1.E-8,
      bool print = false );

  }// name ed
}// name cmz

#endif
