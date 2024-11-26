/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

/**
 * @file lobpcg.h++
 *
 * @brief Implements simple call to LOBPCG to get the lowest few
 *        eigenstates of a given matrix.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/10/2024
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

namespace macis {

/**
 * @brief Perform a band Lanczos calculation on the Hamiltonian operator H,
 * starting from vectors qs, for at most nLanIts iterations. Returns the first
 * len(qs) eigenvalues, converged to some accuracy. Note that this
 * implementation does not account for deflations (i.e., pruning the span of the
 * qs for linear dependencies in higher powers of H).
 *
 * @param[in] typename Functor &H: Hamiltonian oprator. Just needs to implement
 * a matrix vector product.
 * @param[in] std::vector<std::vector<double> > &qs: Initial set of vetors to
 * perform the band Lanczos on. Deleted on exit.
 * @param[out] std::vector<double> &evals: Lowest len(qs) eigenvalues.
 * @param[out] std::vector<std::vector<double> > &evecs: Lowest len(qs)
 * eigenvectors, in the Krylov basis.
 * @param[in] int &nLanIts: Number of Lanczos iterations to perform.
 * @param[in] double tol: Target tolerance for the eigenvalue convergence.
 * @param[in] double thres: Threshold determining when to ignore beta's for
 * being too small.
 * @param[in] bool print: If true, write intermediate results to file.
 *
 * @author Carlos Mejuto Zaera
 * @date 25/10/2024
 */
template <typename Functor>
void LobpcgGS(size_t dimH, size_t nstates, const Functor& H,
              std::vector<double>& evals, std::vector<double>& X, int maxIts,
              double tol = 1.E-8, bool print = false) {
  // Run LOBPCG to get the first few eigenstates of a given
  // Hamiltonian
  evals.clear();
  evals.resize(nstates, 0.);
  X.clear();
  X.resize(dimH * nstates, 0.);
  // Hop
  lobpcgxx::operator_action_type<double> Hop =
      [&](int64_t n, int64_t k, const double* x, int64_t ldx, double* y,
          int64_t ldy) -> void {
    for(int ik = 0; ik < k; ik++)
      H.operator_action(1, 1., x + ik * n, n, 0., y + ik * n, n);
  };

  // Random vectors
  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&]() { return dist(gen); };
  std::generate(X.begin(), X.end(), rand_gen);
  lobpcgxx::cholqr(dimH, nstates, X.data(), dimH);  // Orthogonalize

  lobpcgxx::lobpcg_settings settings;
  settings.conv_tol = tol;
  settings.maxiter = maxIts;
  lobpcgxx::lobpcg_operator<double> lob_op(Hop);

  std::vector<double> res(nstates);

  try {
    lobpcgxx::lobpcg(settings, dimH, nstates, nstates, lob_op, evals.data(),
                     X.data(), dimH, res.data());
  } catch(std::runtime_error e) {
    std::cout << "Runtime error during lobpcg: " << e.what() << std::endl;
  }

  if(print) {
    std::cout << std::scientific << std::setprecision(10) << std::endl;
    for(int64_t i = 0; i < nstates; ++i) {
      std::cout << "  evals[" << i << "] = " << evals[i] << ",   res[" << i
                << "] = " << res[i] << std::endl;
    }
  }
}

}  // namespace macis
