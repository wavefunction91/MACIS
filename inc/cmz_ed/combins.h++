/**
 * @file combins.h++
 * @brief Implements a simple routine to build all
 *        possible N-bit strings with k bits set.
 *        Adapted from 
 *        https://www.geeksforgeeks.org/find-combinations-k-bit-numbers-n-bits-set-1-n-k-sorted-order/
 *
 * @author Carlos Mejuto Zaera
 * @date 05/06/2021
 */
#ifndef __INCLUDE_CMZED_COMBINS__
#define __INCLUDE_CMZED_COMBINS__

#include "cmz_ed/utils.h++"

namespace cmz
{
  namespace ed
  {
    
    /**
     * @brief Builds all bit strings of Nbits, with Nset bits set.
     *
     * @param [in] unsigned short Nbits: Size of the bit strings.
     * @param [in] unsigned short Nset : Nr. of bits set.
     *
     * @returns std::vector<unsigned long>: List of all bit strings of
     *          Nbits with Nset bits set.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/06/2021
     */
    std::vector<unsigned long> BuildCombs( unsigned short Nbits, unsigned short Nset);

  }// namespace ed
}// namespace cmz

#endif
