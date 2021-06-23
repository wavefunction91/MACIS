/**
 * @file eigsolver.h++
 * @brief Wrapper to Lapack routine for diagonalizing
 *        a tridiagonal, symmetric matrix.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
#ifndef __INCLUDE_CMZED_EIGSOLVER__
#define __INCLUDE_CMZED_EIGSOLVER__
#include "cmz_ed/utils.h++"

namespace cmz
{
  namespace ed
  {

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
    void Hste_v(const VecD &alphas, const VecD &betas, VectorXd &eigvals, eigMatD &eigvecs);
    
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

  }// namespace ed
}// namespace cmz

#endif
