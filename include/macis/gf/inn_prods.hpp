/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

/**
 * @file inn_prods.h++
 *
 * @brief Inner product routines between vectors.
 * 
 * @author Carlos Mejuto Zaera
 * @date 02/06/2023
 */
#pragma once
#include <Eigen/Core>
#include <complex>
#include <limits>

namespace macis
{
    /**
     * @brief Simple inner product routine between two std::vector<double>
     * 
     * @param [in] const std::vecotr<double>& vecR
     * @param [in] const std::vecotr<double>& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline double MyInnProd(const std::vector<double>& vecR, const std::vector<double>& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      double res = 0.;
      #pragma omp declare reduction \
        (Vsum:double:omp_out=omp_out+omp_in)\
        initializer(omp_priv=0.)
      #pragma omp parallel for reduction (Vsum:res)
      for(size_t i = 0; i < vecR.size();i++) res += vecR[i] * vecL[i];
      return res;
    }
    
    /**
     * @brief Simple inner product routine between two std::vector<std::complex<double> >
     * 
     * @param [in] const std::vecotr<std::complex<double> >& vecR
     * @param [in] const std::vecotr<std::complex<double> >& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline std::complex<double> MyInnProd(const std::vector<std::complex<double> >& vecR, const std::vector<std::complex<double> >& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      std::complex<double> res(0.,0.);
      #pragma omp declare reduction \
        (Vsum:std::complex<double>:omp_out=omp_out+omp_in)\
        initializer(omp_priv=std::complex<double>(0.,0.))
      #pragma omp parallel for reduction (Vsum:res)
      for(size_t i = 0; i < vecR.size();i++) res += conj(vecR[i]) * vecL[i];
      return res;
    }
    
    /**
     * @brief Simple inner product routine between std::vector<double> and std::vector<std::complex<double> >
     * 
     * @param [in] const std::vecotr<std::complex<double> >& vecR
     * @param [in] const std::vecotr<double>& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline std::complex<double> MyInnProd(const std::vector<std::complex<double> >& vecR, const std::vector<double>& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      std::complex<double> res(0.,0.);
      #pragma omp declare reduction \
        (Vsum:std::complex<double>:omp_out=omp_out+omp_in)\
        initializer(omp_priv=std::complex<double>(0.,0.))
      #pragma omp parallel for reduction (Vsum:res)
      for(size_t i = 0; i < vecR.size();i++) res += conj(vecR[i]) * std::complex<double>(vecL[i], 0.);
      return res;
    }
    
    /**
     * @brief Simple inner product routine for two Eigen::VectorXcd 
     * 
     * @param [in] const Eigen::VectorXcd& vecR
     * @param [in] const Eigen::VectorXcd& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline std::complex<double> MyInnProd(const Eigen::VectorXcd& vecR, const Eigen::VectorXcd& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      return vecR.dot(vecL);
    }
    
    /**
     * @brief Simple inner product routine for two Eigen::VectorXd 
     * 
     * @param [in] const Eigen::VectorXd& vecR
     * @param [in] const Eigen::VectorXd& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline double MyInnProd(const Eigen::VectorXd& vecR, const Eigen::VectorXd& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      return vecR.dot(vecL);
    }
}// namespace macis
