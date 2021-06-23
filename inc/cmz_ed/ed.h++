/**
 * @file ed.h++ 
 * @brief Encodes ED routine, taking in integrals and simulation parameters,
 *        and returning the ground state energy and rdms.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/11/2021
 */
#ifndef __INCLUDE_CMZED_ED__
#define __INCLUDE_CMZED_ED__

#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"

namespace cmz
{
  namespace ed
  {

    /**
     * @brief Performs an ED (FCI) ground state calculation
     *        in a many-Fermion system, and returns the ground
     *        state energy and 1-, 2-rdms.
     *
     * @param [in] const Input_t &input: Input dictionary, with information
                   such as nr. of orbitals, electrons, and Lanczos parameters.
     * @param [in] const intgrls::integrals &ints: Integrals defining 
     *             many-Fermion Hamiltonian in 2nd quantization.
     * @param [out] double &E0: Ground state energy.
     * @param [out] std::vector<double> & rdm1: 1-RDM.
     * @param [out] std::vector<double> & rdm2: 2-RDM.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/11/2021
     */
    void RunED( const Input_t &input, const intgrls::integrals &ints, double &E0, std::vector<double> &rdm1, std::vector<double> &rdm2 );

  }// namespace ed
}// namespace cmz

#endif
