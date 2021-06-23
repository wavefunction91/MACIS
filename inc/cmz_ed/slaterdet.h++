/**
 * @file slaterdet.h++
 * @brief Simple Slater determinant class, essentially 
 *        a bit string. First half of the bitstring 
 *        corresponds to spin-up orbitals, half to spin-down.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
#ifndef __INCLUDE_CMZED_SLATERDET__
#define __INCLUDE_CMZED_SLATERDET__

#include "cmz_ed/utils.h++"
#include "cmz_ed/combins.h++"
#include "cmz_ed/integrals.h++"
#include<bit>
#include<set>
#include<algorithm>
#include<assert.h>

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
     * @author Carlos Mejuto Zaera
     * @date   05/04/2021
     */
    class slater_det
    {
      private:
        unsigned long state;
        unsigned short Norbs, Nup, Ndo;
      public:
        slater_det( 
                  unsigned long st_, unsigned short Norbs_, 
                  unsigned short Nup_, unsigned short Ndo_ 
                ) : state(st_), Norbs(Norbs_), Nup(Nup_), Ndo(Ndo_) { assert(Norbs_ <= 16); } ;
        slater_det() : state(0), Norbs(0), Nup(0), Ndo(0) { } ;
        /**
         * @brief Returns occupation bitstring of Slater determinant. 
         *
         * @returns unsigned long: Occupation bitstring. 
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned long GetState( ) const { return state; }
        /**
         * @brief Returns nr. of orbitals of Slater determinant. 
         *
         * @returns unsigned short: Nr. of orbitals  
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short GetNorbs( ) const { return Norbs; }
        /**
         * @brief Returns nr. of spin up electrons in Slater determinant. 
         *
         * @returns unsigned short: Nr. of spin up electrons.  
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short GetNup  ( ) const { return Nup;   }
        /**
         * @brief Returns nr. of spin down electrons in Slater determinant. 
         *
         * @returns unsigned short: Nr. of spin down electrons.  
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short GetNdo  ( ) const { return Ndo;   }
        /**
         * @brief Check if spin up orbital is occupied. 
         *
         * @param [in] unsigend short i: Orbital to check.
         *
         * @returns bool: True if occupied, False otherwise.  
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        bool IsOccUp( unsigned short i ) const { return state & (1 << i); }
        /**
         * @brief Check if spin down orbital is occupied. 
         *
         * @param [in] unsigend short i: Orbital to check.
         *
         * @returns bool: True if occupied, False otherwise.  
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        bool IsOccDo( unsigned short i ) const { return state & (1 << (i + Norbs)); }
        /**
         * @brief Flips bit in occupation bitstring.
         *        Corresponds to creation/annihilation operator. 
         *
         * @param [in] unsigend short i: Orbital to flip. Can be either 
         *             spin up ([0, Norbs-1]) or down ([Norbs, 2*Norbs-1])
         *
         * @returns bool: True if occupied, False otherwise.  
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        void Flip( unsigned short i ) { state ^= (1 << i); }
        /**
         * @brief Performs single excitation of spin up electron between orbitals a-->i.
         *        The excitation is assumed to be valid, this should be checked beforehand.
         *
         * @param [in] unsigned short i: Virtual orbital to be occupied.
         * @param [in] unsigned short a: Occupied orbital to be emptied.
         *
         * @returns double: Sign from excitation. 
         *
         * @author Carlos Mejuto Zaera
         * @date   05/05/2021
         */
        double SingleExcUp( unsigned short i, unsigned short a );
        /**
         * @brief Performs single excitation of spin down electron between orbitals a-->i.
         *        The excitation is assumed to be valid, this should be checked beforehand.
         *
         * @param [in] unsigned short i: Virtual orbital to be occupied.
         * @param [in] unsigned short a: Occupied orbital to be emptied.
         *
         * @returns double: Sign of single excitation
         *
         * @author Carlos Mejuto Zaera
         * @date   05/05/2021
         */
        double SingleExcDo( unsigned short i, unsigned short a ); 
        /**
         * @brief Returns Slater determinant and sign after single spin up excitation.
         *        The excitation is assumed to be valid, has to be checked beforehand. 
         *
         * @param [in]  unsigned short i: Virtual orbital to be occupied.
         * @param [in]  unsigned short a: Occupied orbital to be emptied.
         * @param [out] double& sign: Sign due to the excitation.
         *
         * @returns slater_det: Slater determinant resulting after the excitation. 
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        slater_det GetSingExUpSt( unsigned short i, unsigned short a, double &sign ) const;
        /**
         * @brief Returns Slater determinant and sign after single spin down excitation.
         *        The excitation is assumed to be valid, has to be checked beforehand. 
         *
         * @param [in]  unsigned short i: Virtual orbital to be occupied.
         * @param [in]  unsigned short a: Occupied orbital to be emptied.
         * @param [out] double& sign: Sign due to the excitation.
         *
         * @returns slater_det: Slater determinant resulting after the excitation. 
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        slater_det GetSingExDoSt( unsigned short i, unsigned short a, double &sign ) const;
        /**
         * @brief Counts different bits between *this and an input Slater determinant. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns unsigned short: Nr of bits different between *this and st.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short CountExc( const slater_det &st ) const { return std::popcount( state ^ st.GetState() ); };
        /**
         * @brief Counts different bits between *this and an input Slater determinant
         *        only in spin up region.. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns unsigned short: Nr of bits different between *this and st in spin up
         *          region.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short CountExcUp( const slater_det &st ) const { return std::popcount( (state ^ st.GetState() ) & ((1 << Norbs) - 1) ); };
        /**
         * @brief Counts different bits between *this and an input Slater determinant
         *        only in spin down region.. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns unsigned short: Nr of bits different between *this and st in spin down
         *          region.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short CountExcDo( const slater_det &st ) const { return CountExc( st ) - CountExcUp( st ); };
        /**
         * @brief Give position of first different bit/orbital between *this 
         *        and input Slater determinant which is set/occupied in *this. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns unsigned short: Position of first bits different 
         *          between *this and st that is set in *this.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short GetFlippedOccIndx( const slater_det &st ) const 
        {
          return (ffs( state & (state ^ st.GetState() ) ) - 1) % Norbs; 
        }
        /**
         * @brief Give position of first different bit/orbital between *this 
         *        and input Slater determinant which is set/occupied in *this.
         *        Considers only spin up electrons. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns unsigned short: Position of first bits different 
         *          between *this and st that is set in *this.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short GetFlippedOccIndxUp( const slater_det &st ) const 
        { 
          return (ffs( state & (state ^ st.GetState() ) & ((1 << Norbs) - 1) ) - 1) % Norbs; 
        }
        /**
         * @brief Give position of first different bit/orbital between *this 
         *        and input Slater determinant which is set/occupied in *this.
         *        Considers only spin down electrons. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns unsigned short: Position of first bits different 
         *          between *this and st that is set in *this.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        unsigned short GetFlippedOccIndxDo( const slater_det &st ) const 
        { 
          return (ffs( state & (state ^ st.GetState() ) & (~((1 << Norbs) - 1)) ) - 1) % Norbs; 
        }
        /**
         * @brief Defines order among Slater determinants, for use in std::set. 
         *        Ordered by the numerical value of the occupation bitstring.
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns bool: True if *this goes before st.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        bool operator< ( const slater_det &st ) const { return state < st.GetState(); }
        /**
         * @brief Overloads == operator for Slater determinants, for use in std::set. 
         *
         * @param [in] const slater_det &st: Slater determinant to compare with *this.
         *
         * @returns bool: True if *this is equal to st.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        bool operator== ( const slater_det &st ) const { return state == st.GetState(); }
        /**
         * @brief Returns string representing the Slater determinant
         *
         * @returns string
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        string ToStr( ) const;
        /**
         * @brief Returns string representing the Slater determinant, assuming it's a bra.
         *
         * @returns string
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        string ToStrBra( ) const;
        /**
         * @brief Returns list of occupied spin up orbitals. 
         *
         * @returns std::vector<unsigned short>: List of occupied 
         *          spin up orbitals.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        std::vector<unsigned short> GetOccOrbsUp() const;
        /**
         * @brief Returns list of occupied spin down orbitals. 
         *
         * @returns std::vector<unsigned short>: List of occupied 
         *          spin down orbitals.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        std::vector<unsigned short> GetOccOrbsDo() const;
        /**
         * @brief Creates list of occupied and empty spin up orbitals. 
         *
         * @param [out] std::vector<unsigned short> &occs:  
         *        Occupied spin up orbitals.
         * @param [out] std::vector<unsigned short> &virts:  
         *        Empty spin up orbitals.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        void GetOccsAndVirtsUp( std::vector<unsigned short> &occs, std::vector<unsigned short> &virts ) const;
        /**
         * @brief Creates list of occupied and empty spin down orbitals. 
         *
         * @param [out] std::vector<unsigned short> &occs:  
         *        Occupied spin down orbitals.
         * @param [out] std::vector<unsigned short> &virts:  
         *        Empty spin down orbitals.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        void GetOccsAndVirtsDo( std::vector<unsigned short> &occs, std::vector<unsigned short> &virts ) const;
        /**
         * @brief Creates list of single and double excitations from
         *        current determinant.
         *
         * @returns std::vector<slater_det>: Vector of single and double excitations.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        std::vector<slater_det> GetSinglesAndDoubles() const;
        /**
         * @brief Creates list of single and double excitations from
         *        current determinant, considering only excitations
         *        generated by the input Hamiltonian.
         *
         * @param [in] const intgrls::integrals *pint: Pointer
         *             to integrals defining many-body Hamiltonian.
         *
         * @returns std::vector<slater_det>: Vector of single and double excitations.
         *
         * @author Carlos Mejuto Zaera
         * @date   05/04/2021
         */
        std::vector<slater_det> GetSinglesAndDoubles( const intgrls::integrals *pint ) const;
    };
    
    typedef std::vector<slater_det> VecSlaterDets;
    typedef std::set<slater_det> SetSlaterDets;
    typedef std::set<slater_det>::const_iterator SetSlaterDets_It;

    /**
     * @brief Builds many-Fermion Hilbert space for specified nr. of orbitals, and electrons of spin up and down.
     *
     * @param [in] unsigned short Norbs: Nr. of orbitals. Has to be <= 16
     * @param [in] unsigned short Nups: Nr of electrons of spin up. Nups <= Norbs.
     * @param [in] unsigned short Ndos: Nr of electrons of spin down. Ndos <= Norbs.
     *
     * @returns SetSlaterDets: std::set<slater_det> including all Slater determinants
     *          in the specified Hilbert space.
     *
     * @author Carlos Mejuto Zaera
     * @date   05/04/2021
     */
    SetSlaterDets BuildFullHilbertSpace( unsigned short Norbs, unsigned short Nups, unsigned short Ndos );
    /**
     * @brief Determine list of Slater determinant pairs
     *        connected by either a single or double excitation.. 
     *
     * @param [in] const SetSlaterDets &stts: Set of Slater
     *             determinants among which to determine
     *             all pairs connected by singles or doubles.
     *
     * @returns std::vector<std::pair<size_t, size_t> >: List of 
     *          index-parts for the Slater determinants connected 
     *          by either singles or doubles.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/11/2021
     */
    std::vector<std::pair<size_t, size_t> > GetSingDoublPairs( const SetSlaterDets &stts );

  }// namespace ed
}// namespace cmz

#endif
