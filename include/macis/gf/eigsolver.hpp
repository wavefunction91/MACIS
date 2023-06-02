/**
 * @file eigsolver.h++
 * @brief Wrapper to Lapack routine for diagonalizing
 *        a tridiagonal, symmetric matrix.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace macis
{

  using namespace Eigen;

  typedef MatrixXd eigMatD;
  typedef SparseMatrix<double, RowMajor> SpMatD;

  extern "C" {
    extern int dsteqr_(char*, int*, double*, double*, double*, int*, double*, int*); 
    extern int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
  }
  
  /**
   * @brief Computes the eigenvalues and eigenvectors of a tridiagonal, symmetric matrix
   *        using Lapack.
   *
   * @param [in] const std::vector<double> &alphas: Diagonal of the matrix.
   * @param [in] const std::vector<double> &betas: Off-diagonal of the matrix.
   * @param [out] Eigen::VectorXd &eigvals: Eigenvalues.
   * @param [out] Eigen::MatrixXd &eigvecs: Eigenvectors.
   *
   * @author Carlos Mejuto Zaera
   * @date 05/04/2021
   */
  void Hste_v(const std::vector<double> &alphas, const std::vector<double> &betas, VectorXd &eigvals, eigMatD &eigvecs);
  
  /**
   * @brief Computes the eigenvalues and eigenvectors of a tridiagonal, symmetric matrix
   *        using Lapack.
   *
   * @param [in] const std::vector<double> &alphas: Diagonal of the matrix.
   * @param [in] const std::vector<double> &betas: Off-diagonal of the matrix.
   * @param [out] Eigen::VectorXd &eigvals: Eigenvalues.
   * @param [out] Eigen::MatrixXd &eigvecs: Eigenvectors.
   *
   * @author Carlos Mejuto Zaera
   * @date 05/04/2021
   */
  void Hsyev(const eigMatD &H, VectorXd &eigvals, eigMatD &eigvecs);

}// namespace macis
