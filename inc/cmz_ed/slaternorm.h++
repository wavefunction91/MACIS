#ifndef __INCLUDE_CMZED_SLATERDET_L__
#define __INCLUDE_CMZED_SLATERDET_L__

#include "cmz_ed/utils.h++"
#include "cmz_ed/combins.h++"
#include "cmz_ed/integrals.h++"
#include<bit>
#include<set>
#include<algorithm>
#include<assert.h>
#include<bitset>

namespace cmz
{
  namespace ed
  {

    /**
     * @brief Class implementing a Slater determinant state, describing a many Fermion state.
     *
     *        Implements functions to check orbital occupation,
     *        as well as performing single excitations, taking
     *        the Fermionic sign into account.
     *
     */
    template<unsigned int N> 
    class slater_det_l
    {
      private:
        std::bitset<N> state;
        unsigned short Norbs, Nup, Ndo;
      public:
        slater_det_l(
                  bitset<N> st_, unsigned short Norbs_,
                  unsigned short Nup_, unsigned short Ndo_
                ) : state(st_), Norbs(Norbs_), Nup(Nup_), Ndo(Ndo_) {} ;
        slater_det_l() :  Norbs(0), Nup(0), Ndo(0) { } ;
     };

  }// namespace ed
}// namespace cmz

#endif
