/**
 * @file integrals.h++
 * @brief This code is modified from the corresponding
 *        header in the ASCI library by Stephen J. Cotton.
 *
 *        Here we have the class storing the 1- and
 *        2-body integrals in a many-body quantum
 *        system. We add the capability of reading
 *        them from an FCIDUMP file, receiving them
 *        from a separate code as two vectors, and
 *        rotating the integrals under a one-particle
 *        transformation.
 *
 *        Throughout the code and classes, we assume
 *        that the integrals do not depend on spin, 
 *        and further that they are all real valued.
 *
 *        Within the class, the integrals are stored
 *        in physicist format <ij|ab>. This means, that
 *        the element u[ indexer(i,j,a,b) ] corresponds
 *        to the operator
 *                  c^+_i c^+_j c_b c_a
 *
 *        FCIDUMP files store integrals in chemist 
 *        format (ia|jb) instead, so we change indices
 *        when reading.
 *
 * @author Stephen J. Cotton
 * @author Carlos Mejuto Zaera 
 * @date 05/04/2021
 */
#ifndef __INCLUDE_CMZED_INTEGRALS__
#define __INCLUDE_CMZED_INTEGRALS__

#include "cmz_ed/utils.h++"
#include <cmath>
#include <unordered_map>
#include <mpi.h>

namespace cmz
{
  namespace ed
  {
    namespace intgrls
    {
      /**
       * @brief Indexer for 2-body (hence, 4-index) many-body tensors.
       *        Implements convention to map 4-index tensor into 1d vector.
       *        This indexer works on chemical notation, since it's used.
       *        to read in integrals from FCIDUMP files, which are typically
       *        in chemistry notation.
       *
       * @author Stephen J. Cotton
       * @date 05/04/2021
       */
      class four_indx{
        public:
      
          int64_t i,j,a,b;
      
          four_indx(int64_t i, int64_t a, int64_t j, int64_t b): i(i), j(j), a(a), b(b)
          {
            order();
          }
      
          /**
           * @brief Orders each pair of virt/occ orbital indices by magnitude.
           *        Assumes real orbitals.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          void order() //Apply symmetry, assuming real integrals
          {
            if(i < a) std::swap(i,a);
            if(j < b) std::swap(j,b);
          }
      
          /**
           * @brief Equal operator overload.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool operator ==( const four_indx &idx ) const
          {
            return i == idx.i && a == idx.a && j == idx.j && b == idx.b;
          }
      
          /**
           * @brief Lower than operator overload.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool operator <( const four_indx &idx ) const
          {
            return i < idx.i || i == idx.i && (
                     a < idx.a || a == idx.a && (
                     j < idx.j || j == idx.j && b < idx.b ) );
          }
          
      };
      
      /**
       * @brief Indexer for 1-body (hence, 2-index) many-body tensors.
       *        Implements convention to map 2-index tensor into 1d vector.
       *
       * @author Stephen J. Cotton
       * @date 05/04/2021
       */
      class two_indx{
        public:
      
          int64_t i,a;
      
          two_indx(int64_t i, int64_t a): i(i), a(a)
          {
            order();
          }
      
          /**
           * @brief Orders each pair of virt/occ orbital indices by magnitude.
           *        Assumes real orbitals.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          void order() //Apply symmetry, assuming real integrals
          {
            if(i < a) std::swap(i,a);
          }
          
          /**
           * @brief Equal operator overload.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool operator ==( const two_indx &idx ) const
          {
            return i == idx.i && a == idx.a;
          }
      
          /**
           * @brief Lower than operator overload.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool operator <( const two_indx &idx ) const
          {
            return i < idx.i || i == idx.i && a < idx.a;
          }
      
      };

    }// namespace intgrls
  }// namespace ed
}// namespace cmz

/**
 * @brief Simple hash for four_indx, needed for unordered_map.
 *
 * @author Stephen J. Cotton
 * @date 05/04/2021
 */
template<>
struct std::hash<cmz::ed::intgrls::four_indx>
{
  size_t operator ()( const cmz::ed::intgrls::four_indx &idx ) const
  {
    const auto &[i,a,j,b] = idx;

    auto s = []( int64_t bits, size_t i ) { return i <<= bits; };
    return i + s(16,a) + s(32,j) + s(48,b);
  }
};

/**
 * @brief Simple hash for two_indx, needed for unordered_map.
 *
 * @author Stephen J. Cotton
 * @date 05/04/2021
 */
template<>
struct std::hash<cmz::ed::intgrls::two_indx>
{
  size_t operator ()( const cmz::ed::intgrls::two_indx &idx ) const
  {
    const auto &[i,a] = idx;
    auto s = []( int64_t bits, size_t i ) { return i <<= bits; };
    return i + s(16,a);
  }
};

namespace cmz
{
  namespace ed
  {
    namespace intgrls
    {
      /**
       * @brief Indexing scheme for the integrals class. here,
       *        we will store integrals in physics notation.
       *        The orbital indices will designate spin orbitals,
       *        first spin up, then spin down, such that the spin
       *        up orbitals are indexed from [0, norbs-1], and the
       *        spin down from [norbs, 2*norbs -1]. Still, we assume
       *        identical behaviour between spin up and down, hence
       *        we only index the [0, norbs-1] interval.
       *
       * @author Stephen J. Cotton
       * @date 05/04/2021
       */
      class indexer
      {
        int64_t norbitals;
        public:
         
          /**
           * @brief Constructor, taking nr. of orbitals in the system.
           *
           * @param [in] int64_t norbs: Nr. of orbitals. 
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          indexer(int64_t norbs) : norbitals(norbs) {}
      
          /**
           * @brief Access two-index quantity. 
           *
           * @param [in] int64_t i: Virtual orbital index. 
           * @param [in] int64_t a: Occupied orbital index. 
           *
           * @returns int: Index corresponding to (i,a) two-body index.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          int64_t operator()(int64_t i, int64_t a) const
          {
            i %= norbitals; a %= norbitals; //Assuming integrals are equal for different spin
            return i + norbitals * a;
          }
      
          /**
           * @brief Access four-index quantity. 
           *
           * @param [in] int64_t i: Virtual orbital index 1. 
           * @param [in] int64_t j: Virtual orbital index 2. 
           * @param [in] int64_t a: Occupied orbital index 1. 
           * @param [in] int64_t b: Occupied orbital index 2. 
           *
           * @returns int: Index corresponding to <ij|ab> four-body index.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          int64_t operator()(int64_t i, int64_t j, int64_t a, int64_t b) const
          {
            i %= norbitals; a %= norbitals; //Assuming integrals are equal for different spin
            j %= norbitals; b %= norbitals;
            return i + norbitals * ( a + norbitals * (j + norbitals * b));
          }
      
          /**
           * @brief Check if two spin orbital indices correspond to the
           *        the same spin 
           *
           * @param [in] int64_t i: Orbital index 1. 
           * @param [in] int64_t a: Orbital index 2. 
           *
           * @returns bool: Do the two spin orbital indices correspond to the
           *                same spin? 
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool same_spin(int64_t i, int64_t a) const
          {
            return (i < norbitals && a < norbitals) || (i >= norbitals && a >= norbitals);
          }
      };
      
      /**
       * @brief Integral tables for many-Fermion Hamiltonian.
       *        Implements basic operations, such as reading from
       *        FCIDUMP files and iterable containers, access to
       *        1- and 2-body integrals, active space information,
       *        orbital rotations, and writing integrals to file.
       *
       * @author Stephen J. Cotton
       * @author Carlos Mejuto Zaera
       * @date 05/04/2021
       */
      class integrals{
        public:
          int64_t norbitals;
          int64_t n_iorbitals; //Nr. of inactive  orbitals
          int64_t n_aorbitals; //Nr. of active    orbitals
          int64_t n_sorbitals; //Nr. of secondary orbitals
          std::vector<int> iorbs, aorbs, sorbs; //Lists of orbitals, for iterations
          intgrls::indexer indexer;
      
          double core_energy = 0.;
      
          //Map to store integrals when reading
          std::unordered_map<two_indx,double>  t_store;
          std::unordered_map<four_indx,double> u_store;
      
          //1- and 2-body integrals, for actual calculation
          VecD t, u;
      
          /**
           * @brief Empty table constructor. By default, all orbitals
           *        are assumed to be active.
           *
           * @param [in] int64_t norbitals: Nr. of orbitals.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          integrals( int64_t norbitals ): norbitals(norbitals), n_iorbitals(0), n_aorbitals(norbitals), n_sorbitals(0), indexer(norbitals) { init_orb_lists();}
      
          /**
           * @brief Constructor from FCIDUMP file. 
           *
           * @param [in] int64_t norbitals: Nr. of orbitals.
           * @param [in] const string &file: FCIDUMP file.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          integrals( int64_t norbitals, const string &file): integrals(norbitals)
          {
            read_FCIdump(file);
            copy_to_vectors();
          }
      
          /**
           * @brief Constructor from iterable containers. 
           *
           * @param [in] int64_t norbitals: Nr. of orbitals.
           * @param [in] const iterable &t_in: One-body integrals.
           * @param [in] const iterable &u_in: Two-body integrals.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          template<class iterable>
          integrals( int64_t norbitals, const iterable &t_in, const iterable &u_in) : integrals(norbitals)
          {
            buildFromIterables(t_in, u_in);
            updateStoredInts();
          }
    
          /**
           * @brief Update active space information. 
           *
           * @param [in] int64_t n_iorbs: Nr. of inactive  orbitals.
           * @param [in] int64_t n_aorbs: Nr. of active    orbitals.
           * @param [in] int64_t n_sorbs: Nr. of secondary orbitals.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          void UpdateActiveSpace( int64_t n_iorbs, int64_t n_aorbs, int64_t n_sorbs )
          {
            n_iorbitals = n_iorbs;
            n_aorbitals = n_aorbs;
            n_sorbitals = n_sorbs;
            init_orb_lists();
          }
      
          /**
           * @brief Initialize orbital lists, for easy iteration
           *        through inactive, active and secondary orbitals. 
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          void init_orb_lists()
          {
      
            if(n_iorbitals + n_aorbitals + n_sorbitals != norbitals)
              throw ("Nr. of (inactive + active + secondary) orbitals != Nr. of orbitals");
      
            iorbs.resize(n_iorbitals);
            aorbs.resize(n_aorbitals);
            sorbs.resize(n_sorbitals);
      
            int64_t i;
            for(i = 0; i < n_iorbitals; i++) iorbs[i] = i;
            for(int64_t indx = 0; indx < n_aorbitals; i++, indx++) aorbs[indx] = i;
            for(int64_t indx = 0; indx < n_sorbitals; i++, indx++) sorbs[indx] = i;
          }
      
          /**
           * @brief Access one-body integral (i|a) stored in the
           *        unordered map. 
           *
           * @param [in] int64_t i: Virtual  orbital index.
           * @param [in] int64_t a: Occupied orbital index.
           *
           * @returns double: One-body integral (i|a).
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double map_get( int64_t i, int64_t a ) const
          {
            i %= norbitals;
            a %= norbitals;
      
            try { double x = t_store.at( two_indx( i,a ) ); return x; }
            catch( ... ) { return 0; }
          }
      
          /**
           * @brief Access two-body integral <ij|ab> stored in the
           *        unordered map. Notice that unordered map is stored
           *        in chemical notation (ia|jb). 
           *
           * @param [in] int64_t i: Virtual  orbital index 1.
           * @param [in] int64_t j: Virtual  orbital index 2.
           * @param [in] int64_t a: Occupied orbital index 1.
           * @param [in] int64_t b: Occupied orbital index 2.
           *
           * @returns double: Two-body integral <ij|ab>.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double map_get( int64_t i, int64_t j, int64_t a, int64_t b ) const
          {
            i %= norbitals; j %= norbitals;
            a %= norbitals; b %= norbitals;
      
            try { double x = u_store.at( four_indx( i,a,j,b ) ); return x; }
            catch( ... ) { return 0; }
          }
        
          /**
           * @brief Check if spin orbital index corresponds to spin up.
           *
           * @param [in] int64_t i: Spin orbital index.
           *
           * @returns bool: Does this spin orbital correspond to spin up?
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool isup(int64_t i) const {return i < norbitals;}
          /**
           * @brief Check if spin orbital index corresponds to spin down.
           *
           * @param [in] int64_t i: Spin orbital index.
           *
           * @returns bool: Does this spin orbital correspond to spin down?
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool isdo(int64_t i) const {return !isup(i);}
          /**
           * @brief Check if spin orbital indices corresponds to the same spin.
           *
           * @param [in] int64_t i: Spin orbital index 1.
           * @param [in] int64_t a: Spin orbital index 2.
           *
           * @returns bool: Do these spin orbital indices correspond to the same spin.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool same_spin(int64_t i, int64_t a) const {return (isup(i) && isup(a)) || (isdo(i) && isdo(a));}
      
          /**
           * @brief Access one-body integral (i|a) 
           *
           * @param [in] int64_t i: Virtual  orbital index.
           * @param [in] int64_t a: Occupied orbital index.
           *
           * @returns double: Two-body integral (i|a).
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double get(int64_t i, int64_t a) const
          {
            if(!same_spin(i,a)) return 0.;
            return t[ indexer(i,a) ];
          }
      
          /**
           * @brief Access two-body integral in physics notation <ij|ab>.
           *
           * @param [in] int64_t i: Virtual  orbital index 1.
           * @param [in] int64_t j: Virtual  orbital index 2.
           * @param [in] int64_t a: Occupied orbital index 1.
           * @param [in] int64_t b: Occupied orbital index 2.
           *
           * @returns double: Two-body integral <ij|ab>.
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          double getPhys(int64_t i, int64_t j, int64_t a, int64_t b) const
          {
            if(!same_spin(i,a) || !same_spin(j,b)) return 0.;
            return u[ indexer(i, j, a, b) ];
          }
      
          /**
           * @brief Access two-body integral in chemistry notation (ia|jb).
           *
           * @param [in] int64_t i: Virtual  orbital index 1.
           * @param [in] int64_t j: Virtual  orbital index 2.
           * @param [in] int64_t a: Occupied orbital index 1.
           * @param [in] int64_t b: Occupied orbital index 2.
           *
           * @returns double: Two-body integral (ia|jb).
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          double getChem(int64_t i, int64_t a, int64_t j, int64_t b) const
          {
            if(!same_spin(i,a) || !same_spin(j,b)) return 0.;
            return u[ indexer(i, j, a, b) ];
          }
      
          /**
           * @brief Update one-body integral (i|a) 
           *
           * @param [in] int64_t i: Virtual  orbital index.
           * @param [in] int64_t a: Occupied orbital index.
           * @param [in] double val: New value for integral.
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          void update(int64_t i, int64_t a, double val)
          {
            if(!same_spin(i,a)) return;
            t[ indexer(i,a) ] = val;
          }
      
          /**
           * @brief Update two-body integral in physics notation <ij|ab>.
           *
           * @param [in] int64_t i: Virtual  orbital index 1.
           * @param [in] int64_t j: Virtual  orbital index 2.
           * @param [in] int64_t a: Occupied orbital index 1.
           * @param [in] int64_t b: Occupied orbital index 2.
           * @param [in] double val: New value for <ij|ab>.
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          void update(int64_t i, int64_t j, int64_t a, int64_t b, double val)
          {
            if(!same_spin(i,a) || !same_spin(j,b)) return;
            u[ indexer(i, j, a, b) ] = val;
          }
      
          /**
           * @brief Return core energy. 
           *
           * @returns double: Core energy.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double get_core_energy() const {return core_energy;}
      
          /**
           * @brief Copy integrals from unordered_map, in chemistry
           *        notation, to the 1d vector<double> in physics notation. 
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          void copy_to_vectors()
          {
            t.resize( size_t(pow( norbitals, 2 )) );
            u.resize( size_t(pow( norbitals, 4 )) );
            std::fill(t.begin(), t.end(), 0.);
            std::fill(u.begin(), u.end(), 0.);
      
            for( int64_t i=0; i<norbitals; ++i )
              for( int64_t a=0; a<norbitals; ++a )
                t[ indexer( i,a ) ] = map_get( i,a );
      
            for( int64_t i=0; i<norbitals; ++i )
              for( int64_t j=0; j<norbitals; ++j )
                for( int64_t a=0; a<norbitals; ++a )
                  for( int64_t b=0; b<norbitals; ++b )
                    u[ indexer( i,j,a,b ) ] = map_get( i,j,a,b );
          }
    
          /**
           * @brief Copy integrals from iterables.
           *
           * @param [in] const iterable &t_in: 1-body integrals.
           * @param [in] const iterable &u_in: 2-body integrals.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          template<class iterable>
          void buildFromIterables(const iterable &t_in, const iterable &u_in)
          {
            size_t t_in_size = t_in.size();
            size_t u_in_size = u_in.size();
            size_t t_size = size_t( pow( norbitals, 2 ) );
            size_t u_size = size_t( pow( norbitals, 4 ) );
            if(t_in_size != t_size || u_in_size != u_size)
            {
              std::stringstream ss;
              ss << "Error in buildFromIterables! (t_in, u_in) sizes, which are ( "
                 << t_in_size << ", " << u_in_size << ") do not correspond to "
                 << "nr of orbitals " << norbitals;
              throw ( ss.str() );
            }
            t.resize(t_size);
            u.resize(u_size);
            size_t cont = 0;
            for (const auto t_el : t_in)
            {
              t[cont] = t_el;
              cont++;
            }
            cont = 0;
            for (const auto u_el : u_in)
            {
              u[cont] = u_el;
              cont++;
            }
          }
      
        private:
    
          /**
           * @brief Rotate 1-body tensor under given orbital transformation. 
           * 
           * @param [in] const Eigen::MatrixXd &V: Orbital transformation matrix. 
           * @param [in] const std::vector<double> &A: 1-body tensor in vector form. 
           *
           * @returns std::vector<double>: 1-body tensor in new orbital basis.
           *
           * @author Stephen J. Cotton 
           * @date 05/04/2021
           */
          VecD rotate1( const eigMatD &V, const VecD &A ) const
          {
            VecD tmp(size_t(pow(norbitals,2)), 0.);
            for( int64_t i=0; i<norbitals; ++i )
            {
              for( int64_t a=0; a<norbitals; ++a )
                for( int64_t ii=0; ii<norbitals; ++ii )
                  for( int64_t aa=0; aa<norbitals; ++aa )
                    tmp[ indexer( i,a ) ] += V(ii,i) * A[ indexer( ii,aa ) ] * V(aa,a);
            }
            return tmp;
          }
      
          /**
           * @brief Rotate 2-body tensor under given orbital transformation. 
           * 
           * @param [in] const Eigen::MatrixXd &V: Orbital transformation matrix. 
           * @param [in] const std::vector<double> &A: 2-body tensor in vector form. 
           *
           * @returns std::vector<double>: 2-body tensor in new orbital basis.
           *
           * @author Stephen J. Cotton 
           * @date 05/04/2021
           */
          VecD rotate2( const eigMatD &V, const VecD &A ) const
          {
            VecD tmp(size_t(pow(norbitals,4)), 0.);
            for( int64_t i=0; i<norbitals; ++i ) for( int64_t j=0; j<norbitals; ++j )
            {
              for( int64_t a=0; a<norbitals; ++a ) for( int64_t b=0; b<norbitals; ++b )
                for( int64_t ii=0; ii<norbitals; ++ii )
                  for( int64_t aa=0; aa<norbitals; ++aa )
                    tmp[ indexer( i,j,a,b ) ] += V(ii,i) * A[ indexer( ii,j,aa,b ) ] * V(aa,a);
            }
      
            VecD another_tmp = tmp;
            std::fill(tmp.begin(), tmp.end(), 0.);
            for( int64_t i=0; i<norbitals; ++i ) for( int64_t j=0; j<norbitals; ++j )
            {
              for( int64_t a=0; a<norbitals; ++a ) for( int64_t b=0; b<norbitals; ++b )
                for( int64_t jj=0; jj<norbitals; ++jj )
                  for( int64_t bb=0; bb<norbitals; ++bb )
                    tmp[ indexer( i,j,a,b ) ] += V(jj,j) * another_tmp[ indexer( i,jj,a,bb ) ] * V(bb,b);
            }
      
            return tmp;
          }
    
        public:
      
          /**
           * @brief Rotate 1-body tensor under given orbital transformation. To be called from outside the class. 
           * 
           * @param [in] const Eigen::MatrixXd &V: Orbital transformation matrix. 
           * @param [in] const std::vector<double> &A: 1-body tensor in vector form. 
           *
           * @returns std::vector<double>: 1-body tensor in new orbital basis.
           *
           * @author Stephen J. Cotton 
           * @date 05/04/2021
           */
          VecD rotate1AS( const eigMatD &V, const VecD &A ) const
          {
            VecD tmp(size_t(pow(norbitals,2)), 0.);
            for( int64_t i=0; i<norbitals; ++i )
            {
              for( int64_t a=0; a<norbitals; ++a )
                for( int64_t ii=0; ii<norbitals; ++ii )
                  for( int64_t aa=0; aa<norbitals; ++aa )
                    tmp[ indexer( i,a ) ] += V(ii,i) * A[ indexer( ii,aa ) ] * V(aa,a);
            }
            return tmp;
          }
      
          /**
           * @brief Rotate 2-body tensor under given orbital transformation. Lower scaling version. 
           * 
           * @param [in] const Eigen::MatrixXd &V: Orbital transformation matrix. 
           * @param [in] const std::vector<double> &A: 2-body tensor in vector form. 
           *
           * @returns std::vector<double>: 2-body tensor in new orbital basis.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          VecD rotate2AS( const eigMatD &V, const VecD &A ) const
          {
            VecInt occorbs( iorbs.size() + aorbs.size() );
            std::copy( iorbs.begin(), iorbs.end(), occorbs.begin() );
            std::copy( aorbs.begin(), aorbs.end(), occorbs.begin() + iorbs.size() );
    
            VecD tmp(size_t(pow(norbitals,4)), 0.);
            for( int64_t i=0; i<norbitals; ++i ) for( int64_t j=0; j<norbitals; ++j )
            {
              for( int64_t a=0; a<norbitals; ++a ) for( auto const &bb : occorbs )
                for( int64_t b=0; b<norbitals; ++b )
                    tmp[ indexer( i,j,a,bb ) ] += A[ indexer( i,j,a,b ) ] * V(b,bb);
            }
      
            VecD another_tmp = tmp;
            std::fill(tmp.begin(), tmp.end(), 0.);
            for( int64_t i=0; i<norbitals; ++i ) for( int64_t j=0; j<norbitals; ++j )
            {
              for( int64_t aa=0; aa<norbitals; ++aa ) for( auto const &bb : occorbs )
                for( int64_t a=0; a<norbitals; ++a )
                    tmp[ indexer( i,j,aa,bb ) ] += another_tmp[ indexer( i,j,a,bb ) ] * V(a,aa);
            }
      
            another_tmp = tmp;
            std::fill(tmp.begin(), tmp.end(), 0.);
            for( int64_t i=0; i<norbitals; ++i ) for( int64_t jj=0; jj<norbitals; ++jj )
            {
              for( int64_t aa=0; aa<norbitals; ++aa ) for( auto const &bb : occorbs )
                for( int64_t j=0; j<norbitals; ++j )
                    tmp[ indexer( i,jj,aa,bb ) ] += V(j,jj) * another_tmp[ indexer( i,j,aa,bb ) ];
            }
      
            another_tmp = tmp;
            std::fill(tmp.begin(), tmp.end(), 0.);
            for( int64_t ii=0; ii<norbitals; ++ii ) for( int64_t jj=0; jj<norbitals; ++jj )
            {
              for( int64_t aa=0; aa<norbitals; ++aa ) for( auto const &bb : occorbs )
                for( int64_t i=0; i<norbitals; ++i )
                    tmp[ indexer( ii,jj,aa,bb ) ] += V(i,ii) * another_tmp[ indexer( i,jj,aa,bb ) ];
            }
            
            return tmp;
          }
    
          /**
           * @brief Rotate 1- and 2-RDM with input orbital transformation.
           * 
           * @param [in] const Eigen::MatrixXd &V: Orbital transformation matrix. 
           *
           * @author Stephen J. Cotton 
           * @date 05/04/2021
           */
          void rotate_orbitals(const eigMatD &V)
          {
            t = rotate1(V, t);
            u = rotate2(V, u);
          }
      
          /**
           * @brief Reads integrals from FCIDUMP file, stores them
           *        in unordered_map's.
           *
           * @param [in] const string &file: FCIDUMP file.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          void read_FCIdump(const string &file)
          {
            core_energy = 0.;
            t_store.clear();
            u_store.clear();
            
            std::ifstream ifile(file);
            if(!ifile)
              throw ("ERROR OPENING FILE " + file);
      
            int world_rank; MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
            if( world_rank == 0 ) {
              std::cout << "READING FCIDUMP FILE " << file << std::endl;
            }
      
            string line;
            int64_t i, j, a, b;
            double matel;
      
            int64_t count = 0;
            bool ignored = false; //For ignoring 1-electron energies
      
            while(std::getline(ifile, line))
            {
              std::stringstream ss(line);
              count++;
      
              ss >> i >> a >> j >> b >> matel;
             
              if(abs(matel) < 1.E-16) continue;
      
              bool bad = false; //Flag to check for errors
              if(!ss) bad = true;
      
              //Classify read data
              if( i == 0 && j == 0 && a == 0 && b == 0 )
                core_energy = matel;
              else if( i > 0 && j == 0 && a == 0 && b == 0 )
              {
                if( !ignored )
                {
                  std::cout << "Ignoring 1-electron energies in FCIdump file" << std::endl;
                  ignored = true;
                }
              }
              else if( i == 0 || a == 0 )
                throw "Unless core energy, indices i and a should be greater than zero in FCIdump file";
              else if( j == 0 && b == 0 )
                t_store.insert( { two_indx( i-1, a-1 ), matel } );
              else if( j > 0 && b > 0 )
                u_store.insert( { four_indx( i-1, a-1, j-1, b-1 ), matel } );
              else if( j == 0 && b != 0 || j != 0 && b == 0 )
                throw "Indices j and b should either both be zero, or not, in FCIdump file";
              else
                bad = true;
      
              if( bad ) throw ( "Bad integral read of " + file);
            }	
            if( world_rank == 0 ) {
            std::cout << "Read " << t_store.size() + u_store.size() << " integrals" << std::endl;
            if( core_energy )
              std::cout << "  (also read core energy = " << core_energy << ")" << std::endl;
            }
          }
      
          /**
           * @brief Read integrals from std::vector<double>'s.
           *
           * @param [in] const std::vector<double> &in_t: 1-body integrals.
           * @param [in] const std::vector<double> &in_u: 2-body integrals.
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          void updateInts( const VecD &in_t, const VecD &in_u )
          {
            //Update the integrals used in calculations
            if( t.size() != in_t.size() || u.size() != in_u.size() )
              throw( "Error in integrals.updateInts( in_t, in_u )!! Sizes don't match!" );
            t.assign( in_t.begin(), in_t.end() );
            u.assign( in_u.begin(), in_u.end() );
          }
      
          /**
           * @brief Update unordered_map's with content in 1d vectors.
           *        Necessary to write rotated integrals, since the rotation
           *        subroutines affect the 1d vectors, not the unordered_map's. 
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          void updateStoredInts()
          {
            //Update the stored integrals. Important for printing
            //First, one-body terms
            for (int64_t i = 0; i < norbitals; i++)
            {
              for (int64_t a = 0; a < norbitals; a++)
                t_store.insert_or_assign( two_indx(i,a), t[ indexer(i,a) ] );
            }
            //Finally, two-body terms
            for (int64_t i = 0; i < norbitals; i++)
            {
              for (int64_t j = 0; j < norbitals; j++)
                for (int64_t a = 0; a < norbitals; a++)
                  for (int64_t b = 0; b < norbitals; b++)
                    u_store.insert_or_assign( four_indx(i,a,j,b), u[ indexer(i,j,a,b) ] );
            }
          }
      
          /**
           * @brief Writes integrals to FCIDUMP file. 
           *
           * @param [in] const string &file: FCIDUMP file name.
           *
           * @author Stephen J. Cotton 
           * @date 05/04/2021
           */
          void write_FCIdump( const string &file ) const
          {
            std::ofstream out( file );
            out.precision(12);
            auto w5 = std::setw(5), w20 = std::setw(20);
            const int64_t n = norbitals;
            std::cout << "Writing integrals to " << file << std::endl;
            std::cout << "Going to write " << t_store.size() + u_store.size() << " integrals" << std::endl;
      
            if( core_energy )
              out  << w5 << 0 << w5 << 0 << w5 << 0 << w5 << 0 << w20 << core_energy << std::endl;
      
            for( auto [idx, x] : t_store )
            {
              if( abs(x) < 1.E-10 )
                continue;
              out  << w5 << idx.i+1 << w5 << idx.a+1 << w5 << 0 << w5 << 0 << w20 << x << std::endl;
            }
       
            for( auto [idx, x] : u_store )
            {
              if( abs(x) < 1.E-10 )
                continue;
              out  << w5 << idx.i+1 << w5 << idx.a+1 << w5 << idx.j+1 << w5 << idx.b+1 << w20 << x << std::endl;
            }
          }
      
      };
    }// namespace intgrls

  }// namespace ed
}// namespace cmz
#endif
