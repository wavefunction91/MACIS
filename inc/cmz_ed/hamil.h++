/**
 * @file hamil.h++
 * @brief Base class for Fermionic Hamiltonians. 
 *
 * @author Carlos Mejuto Zaera
 * @date 05/05/2021
 */
#ifndef __INCLUDE_CMZED_HAMIL__
#define __INCLUDE_CMZED_HAMIL__

#include "cmz_ed/utils.h++"
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include <math.h>
#include <utility>

namespace cmz
{
  namespace ed
  {

    /**
     * @brief Class implementing a Fermionic Hamiltonian.
     *        Includes basic matrix element evaluation functions,
     *        and formation of all pairs of connected determinants.
     *        Contains a pointer to a const instance of intgrls::integrals.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/05/2021
     */
    class FermionHamil
    {
      private:
        const intgrls::integrals *pints;
      public:
        FermionHamil( const intgrls::integrals &ints  ) : pints(&ints) {};
        /**
         * @brief Return number of orbitals in the Hamitlonian.
         *
         * @returns unsigned short: Nr. of orbitals.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/05/2021
         */
        unsigned short GetNorbs( ) const { return pints->norbitals; }
        /**
         * @brief Evaluate Hamiltonian matrix element between two
         *        Slater determinants. 
         *
         * @param [in] const slater_det &L_st: Left Slater determinant.
         * @param [in] const slater_det &R_st: Right Slater determinant.
         *
         * @returns double: <L_st|H|R_st>.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/05/2021
         */
        double GetHmatel( const slater_det &L_st, const slater_det &R_st ) const;
        /**
         * @brief Determine list of Slater determinant pairs
         *        with non-zero matrix elements. 
         *
         * @param [in] const SetSlaterDets &stts: Set of Slater
         *             determinants among which to determine
         *             all pairs connected by the Hamiltonian.
         *
         * @returns std::vector<std::pair<size_t, size_t> >: List of 
         *          index-parts for the Slater determinants connected 
         *          by the Hamiltonian.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/05/2021
         */
        std::vector<std::pair<size_t, size_t> > GetHpairs( const SetSlaterDets &stts ) const;
      
    };
    
    /**
     * @brief Compute matrix representation of Hamiltonian operator
     *        in given basis of Slater determinants. 
     *
     * @param [in] const FermionHamil* H: Pointer to Fermionic Hamiltonian. 
     * @param [in] const SetSlaterDets &stts: Slater determinants onto which
     *             to project the Hamiltonian operator. 
     * @param [in] bool print: Flag to set verbose mode on.
     *
     * @returns Eigen::SparseMatrix<double, Eigen::RowMajor>: Sparse matrix
     *          representation of the Hamiltonian. 
     *
     * @author Carlos Mejuto Zaera
     * @date 05/05/2021
     */
    SpMatD GetHmat( const FermionHamil* H, const SetSlaterDets & stts, bool print = false ); 
    
    /**
     * @brief Compute matrix representation of Hamiltonian operator
     *        in given basis of Slater determinants. 
     *        Pairs of Slater determinants with non-zero
     *        Hamiltonian matrix elements provided as input.
     *
     * @param [in] const FermionHamil* H: Pointer to Fermionic Hamiltonian. 
     * @param [in] const SetSlaterDets &stts: Slater determinants onto which
     *             to project the Hamiltonian operator. 
     * @param [in] const std::vector<std::pair<size_t, size_t> >:
     *             Pairs of indices, indicating which Slater determinants
     *             have non-vanishing Hamiltonian matrix elements.
     *
     * @returns Eigen::SparseMatrix<double, Eigen::RowMajor>: Sparse matrix
     *          representation of the Hamiltonian. 
     *
     * @author Carlos Mejuto Zaera
     * @date 05/05/2021
     */
    SpMatD GetHmat_FromPairs( const FermionHamil* H, const SetSlaterDets & stts, const std::vector<std::pair<size_t, size_t> > &pairs ); 

  }// namespace ed
}// namespace cmz

#endif
