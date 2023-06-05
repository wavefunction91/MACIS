/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

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
#include <lapack.hh>

namespace macis {


typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd eigMatD;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatD;

/**
 * @brief Computes the eigenvalues and eigenvectors of a tridiagonal, symmetric
 * matrix using Lapack.
 *
 * @param [in] const std::vector<double> &alphas: Diagonal of the matrix.
 * @param [in] const std::vector<double> &betas: Off-diagonal of the matrix.
 * @param [out] Eigen::VectorXd &eigvals: Eigenvalues.
 * @param [out] Eigen::MatrixXd &eigvecs: Eigenvectors.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
void Hste_v(const std::vector<double> &alphas, const std::vector<double> &betas,
            Eigen::VectorXd &eigvals, eigMatD &eigvecs);

/**
 * @brief Computes the eigenvalues and eigenvectors of a tridiagonal, symmetric
 * matrix using Lapack.
 *
 * @param [in] const std::vector<double> &alphas: Diagonal of the matrix.
 * @param [in] const std::vector<double> &betas: Off-diagonal of the matrix.
 * @param [out] Eigen::VectorXd &eigvals: Eigenvalues.
 * @param [out] Eigen::MatrixXd &eigvecs: Eigenvectors.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
void Hsyev(const eigMatD &H, Eigen::VectorXd &eigvals, eigMatD &eigvecs);

}  // namespace macis
