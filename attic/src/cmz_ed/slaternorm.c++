#ifndef __INCLUDE_CMZED_SLATERDET_L__
#define __INCLUDE_CMZED_SLATERDET_L__

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
    class slater_det_l
    {
      private:
        uint64_t state;
        unsigned short Norbs, Nup, Ndo;
      public:
        slater_det_l(
                  uint64_t st_, unsigned short Norbs_,
                  unsigned short Nup_, unsigned short Ndo_
                ) : state(st_), Norbs(Norbs_), Nup(Nup_), Ndo(Ndo_) {} ;
        slater_det_l() : state(0), Norbs(0), Nup(0), Ndo(0) { } ;
        /**

  }// namespace ed
}// namespace cmz

#endif
