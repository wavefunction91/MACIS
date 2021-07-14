/**
 * @file rdms.h++
 *
 * @brief Modified from the integrals.h++ header in
 *        Stephen J. Cotton's ASCI library.
 *        This code contains a small class to access
 *        1- and 2-RDMs for quantum many-body systems.
 *        It mainly includes an indexer to keep the
 *        "matrices" stored as vectors.
 *        For now, we assume spin-block diagonal RDMs,
 *        proportional to the identity matrix.
 *
 * @author Stephen J. Cotton
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
#ifndef __INCLUDE_CMZED_RDMS__
#define __INCLUDE_CMZED_RDMS__

#include "cmz_ed/utils.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/slaterdet.h++"

namespace cmz
{
  namespace ed
  {

    namespace rdm{
    
      /**
       * @brief Indexer class to transform from 2 and 4 index
       *        notations to a 1 index notation, since we will
       *        store many-body tensors such as the 1- and 2-RMDs
       *        as vectors. 
       *
       *        Assumes that spin up and spin down 
       *        orbitals behave absolutely identically (no magn. field)
       *        and hence only consider spin up orbitals. Orbitals are
       *        assumed to be ordered first all up orbitals, then all
       *        down orbitals. Hence spin up orbitals have indices
       *        [0, norbitals-1], and spin down orbitals have indices
       *        [norbitals, 2*norbitals - 1].
       *
       *        Not fundamental for this class, but we will store 2-index
       *        many-body tensors in physics notation <ij|ab>, meaning that
       *        (i,j;a,b) corresponds to c^+_i c^+_j c_b c_a.
       *
       * @author Stephen J. Cotton
       * @date 05/04/2021
       */
      class indexer{
        int64_t norbitals;
        public:
         
          /**
           * @brief Constructor.
           *
           * @param [in] int64_t norbs: Nr. of orbitals in the system.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          indexer(int64_t norbs) : norbitals(norbs) {}
      
          /**
           * @brief Transform from 2 index to single index 
           *        (i;a) -> i + norbitals * a. Considers only
           *        spin up orbitals, by making e.g. i -> i % norbitals.
           *
           * @param [in] int64_t i: Virtual index.
           * @param [in] int64_t a: Occupied index.
           *
           * @return int: Composite single index.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          int64_t operator()(int64_t i, int64_t a) const
          {
            i %= norbitals; a %= norbitals; //Assuming equal terms for both spins, and diagonal rdms
            return i + norbitals * a;
          }
      
          /**
           * @brief Transform from 4 index to single index 
           *        (i,j;a,b) -> i + norbitals * a + norbitals^2 * j
           *        + norbitals^3 * b. Considers only
           *        spin up orbitals, by making e.g. i -> i % norbitals.
           *
           * @param [in] int64_t i: Virtual index 1.
           * @param [in] int64_t j: Virtual index 2.
           * @param [in] int64_t a: Occupied index 1.
           * @param [in] int64_t b: Occupied index 2.
           *
           * @return int: Composite single index.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          int64_t operator()(int64_t i, int64_t j, int64_t a, int64_t b) const
          {
            i %= norbitals; a %= norbitals; //Assuming equal terms for both spins, and diagonal rdms
            j %= norbitals; b %= norbitals;
            return i + norbitals * ( a + norbitals * (j + norbitals * b));
          }
      
          /**
           * @brief Checks if two orbital indices correspond to
           *        same spin spin orbitals. 
           *
           * @param [in] int64_t i: Orbital index 1.
           * @param [in] int64_t a: Orbital index 2.
           *
           * @return bool: True if both indices correspond to spin orbitals
           *         of the same spin.
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
       * @brief RDM class, storing both 1-RDM and 2-RDM as
       *        vectors, with an indexer instance to acess the
       *        elements in 2- and 4-index notation. Check indexer
       *        class for the indexing convention.
       *
       *        Further implements useful methods, such as single-particle
       *        orbital rotations, constructing RDMs from active-space-only
       *        RDMs and reading from file or iterables
       *
       * @author Stephen J. Cotton
       * @author Carlos Mejuto Zaera
       * @date 05/04/2021
       */
      class rdms{
        public:
          int64_t norbitals;
          rdm::indexer indexer;
    
          //1- and 2-RDMs stored as vectors
          VecD rdm1, rdm2;
    
          /**
           * @brief Basic constructor, initializes the RDMs
           *        as zero vectors of the right size.
           * 
           * @param [in] int64_t norbitals: Nr of orbitals in the system.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          rdms( int64_t norbitals ): norbitals(norbitals), indexer( norbitals ) 
          {
            rdm1.resize( int(pow( norbitals,2 )) );
            rdm2.resize( int(pow( norbitals,4 )) );
            std::fill( rdm1.begin(), rdm1.end(), 0. );
            std::fill( rdm2.begin(), rdm2.end(), 0. );
          }
    
          /**
           * @brief Constructor, reads the RDMs
           *        from binary file. 
           * 
           * @param [in] int64_t norbitals: Nr of orbitals in the system.
           * @param [in] const string &file: Binary file to read RDM's from. 
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          rdms( int64_t norbitals, const string &file): rdms(norbitals)
          {
            read_RDMs(file);
          }
    
          /**
           * @brief Constructor, reads the RDMs
           *        from iterable vectors. 
           * 
           * @param [in] int64_t norbitals: Nr of orbitals in the system.
           * @param [in] const iterable &rdm1_in: 1-RDM iterable.
           * @param [in] const iterable &rdm2_in: 2-RDM iterable.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          template<class iterable>
          rdms(int64_t norbitals, const iterable &rdm1_in, const iterable &rdm2_in) : rdms(norbitals)
          {
            buildFromIterables(rdm1_in, rdm2_in);
          }
    
          /**
           * @ Constructor using a many-body state.
           *
           * @param [in] int64_t norbitals: Nr. of orbitals.
           * @param [in] const Eigen::VectorXd &vec: Many-body state, coefficient vector.
           * @param [in] const SetSlaterDets &stts: List of Slater determinants,
           *             defining the many-body state. 
           *
           * @returns rdms: Corresponding RDMs.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/07/2021
           */
          rdms( int64_t norbitals, const VectorXd &vec, const SetSlaterDets &stts );

          /**
           * @brief Copy constructor.
           * 
           * @param [in] const rdms &in_rdms. 
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          rdms( const rdms &in_rdms ) : indexer( in_rdms.indexer )
          {
            norbitals = in_rdms.norbitals;
            rdm1      = in_rdms.rdm1; 
            rdm2      = in_rdms.rdm2; 
          }
    
          /**
           * @brief Check if spin orbital index i corresponds to spin up.
           * 
           * @param [in] int64_t i: Spin orbital index. 
           *
           * @returns bool: True if spin orbital corresponds to spin up.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool isup(int64_t i) const {return i < norbitals;}
          /**
           * @brief Check if spin orbital index i corresponds to spin down.
           * 
           * @param [in] int64_t i: Spin orbital index. 
           *
           * @returns bool: True if spin orbital corresponds to spin down.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool isdo(int64_t i) const {return !isup(i);}
          /**
           * @brief Check if two spin orbital indices correspond to the
           *        same spin.
           * 
           * @param [in] int64_t i: Spin orbital index 1. 
           * @param [in] int64_t a: Spin orbital index 2. 
           *
           * @returns bool: True if spin orbitals have the same spin.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          bool same_spin(int64_t i, int64_t a) const {return (isup(i) && isup(a)) || (isdo(i) && isdo(a));}
    
          /**
           * @brief Get (i;a) element of the 1-RDM. 
           * 
           * @param [in] int64_t i: Virtual orbital index. 
           * @param [in] int64_t a: Occupied orbital index. 
           *
           * @returns double: (i;a) element of 1-RDM.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double get(int64_t i, int64_t a) const
          {
            if(!same_spin(i,a)) return 0.;
            return rdm1[ indexer(i,a) ];
          }
    
          /**
           * @brief Get (i,j;a,b) element of the 2-RDM, in 
           *        physics notation, i.e. <ij|ab>. 
           * 
           * @param [in] int64_t i: Virtual orbital index 1. 
           * @param [in] int64_t j: Virtual orbital index 2. 
           * @param [in] int64_t a: Occupied orbital index 1. 
           * @param [in] int64_t b: Occupied orbital index 2. 
           *
           * @returns double: <ij|ab> element of 2-RDM.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double getPhys(int64_t i, int64_t j, int64_t a, int64_t b) const
          {
            if(!same_spin(i,a) || !same_spin(j,b)) return 0.;
            return rdm2[ indexer(i, j, a, b) ];
          }
    
          /**
           * @brief Get (i,j;a,b) element of the 2-RDM, in 
           *        chemists notation, i.e. (ia|jb). 
           * 
           * @param [in] int64_t i: Virtual orbital index 1. 
           * @param [in] int64_t a: Occupied orbital index 1. 
           * @param [in] int64_t j: Virtual orbital index 2. 
           * @param [in] int64_t b: Occupied orbital index 2. 
           *
           * @returns double: (ia|jb) element of 2-RDM.
           *
           * @author Stephen J. Cotton
           * @date 05/04/2021
           */
          double getChem(int64_t i, int64_t a, int64_t j, int64_t b) const
          {
            if(!same_spin(i,a) || !same_spin(j,b)) return 0.;
            return rdm2[ indexer(i, j, a, b) ];
          }
    
          /**
           * @brief Update (i;a) element of the 1-RDM. 
           * 
           * @param [in] int64_t i: Virtual orbital index. 
           * @param [in] int64_t a: Occupied orbital index. 
           * @param [in] double val: New value for (i;a) element of 1-RDM. 
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          void update(int64_t i, int64_t a, double val)
          {
            if(!same_spin(i,a)) return;
            rdm1[ indexer(i,a) ] = val;
          }
    
          /**
           * @brief Update (i,j;a,b) element of the 2-RDM. 
           *        Uses physics notation, i.e. <ij|ab>
           * 
           * @param [in] int64_t i: Virtual orbital index 1. 
           * @param [in] int64_t j: Virtual orbital index 2. 
           * @param [in] int64_t a: Occupied orbital index 1. 
           * @param [in] int64_t b: Occupied orbital index 2. 
           * @param [in] double val: New value for <ij|ab> element of 2-RDM. 
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          void update(int64_t i, int64_t j, int64_t a, int64_t b, double val)
          {
            if(!same_spin(i,a) || !same_spin(j,b)) return;
            rdm2[ indexer(i, j, a, b) ] = val;
          }
    
          /**
           * @brief Fill 1- and 2-RDMs with input vectors. 
           * 
           * @param [in] const iterable &rdm1_in: Input 1-RDM iterable.
           * @param [in] const iterable &rdm2_in: Input 2-RDM iterable.
           *
           * @author Carlos Mejuto Zaera 
           * @date 05/04/2021
           */
          template<class iterable>
          void buildFromIterables(const iterable &rdm1_in, const iterable &rdm2_in)
          {
            size_t rdm1_in_size = rdm1_in.size();
            size_t rdm2_in_size = rdm2_in.size();
            size_t rdm1_size = size_t( pow( norbitals, 2 ) );
            size_t rdm2_size = size_t( pow( norbitals, 4 ) );
            if(rdm1_in_size != rdm1_size || rdm2_in_size != rdm2_size)
            {
              std::stringstream ss;
              ss << "Error in buildFromIterables! (rdm1_in, rdm2_in) sizes, which are ( "
                 << rdm1_in_size << ", " << rdm2_in_size << ") do not correspond to "
                 << "nr of orbitals " << norbitals;
              throw ( ss.str() );
            }
            rdm1.resize(rdm1_size);
            rdm2.resize(rdm2_size);
            size_t cont = 0;
            for (const auto rdm1_el : rdm1_in)
            {
              rdm1[cont] = rdm1_el;
              cont++;
            }
            cont = 0;
            for (const auto rdm2_el : rdm2_in)
            {
              rdm2[cont] = rdm2_el;
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
           * @brief Rotate 1- and 2-RDM with input orbital transformation.
           * 
           * @param [in] const Eigen::MatrixXd &V: Orbital transformation matrix. 
           *
           * @author Stephen J. Cotton 
           * @date 05/04/2021
           */
          void rotate_orbitals(const eigMatD &V)
          {
            rdm1 = rotate1(V, rdm1);
            rdm2 = rotate2(V, rdm2);
          }
    
          /**
           * @brief Read 1- and 2-RDM from input binary file.
           * 
           * @param [in] const string &file: Name of binary file.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          void read_RDMs(const string &file)
          {
            rdm1.clear(); rdm2.clear();
            rdm1.resize(size_t(pow(norbitals, 2)));
            rdm2.resize(size_t(pow(norbitals, 4)));
            
            std::ifstream ifile(file, std::ios::in | std::ios::binary);
            if(!ifile)
              throw ("Could not open " + file);
    
            int64_t norbs;
       
            ifile.read((char*) &norbs, sizeof(int));
    
            if(norbs != norbitals){
              ifile.close();
              throw ("RDMs stored in " + file + " have a different nr of orbitals as expected!");
            }
             
            ifile.read((char*) &rdm1[0], rdm1.size() * sizeof(double));
            ifile.read((char*) &rdm2[0], rdm2.size() * sizeof(double));
    
            ifile.close();
        
            std::cout << "RDMs READ FROM " << file << std::endl;
          }
    
          /**
           * @brief Return 1- and 2-RDMs, with indices constrained to
           *        the active space in an input integrals instance. 
           * 
           * @param [in] const intgrls::integrals &ints: Integrals instance,
           *             contains the active space specifications.
           *
           * @returns rdms: RDMs instance where many-body tensors are
           *          constrained to active space orbitals only.
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          rdms GetASrdms( const intgrls::integrals &ints ) const 
          {
            // Returns the rdms for the active space
            // defined in ints
            int64_t as_size = ints.aorbs.size();
            VecD rdm1_as( int(pow( as_size, 2 )), 0. );
            VecD rdm2_as( int(pow( as_size, 4 )), 0. );
            rdm::indexer as_indxr( as_size );
    
            // 1-rdm
            for( int64_t p = 0; p < as_size; p++ )
            {
              int64_t t = ints.aorbs[p];
              for( int64_t q = 0;  q < as_size; q++ )
              {
                int64_t u = ints.aorbs[q];
                rdm1_as[ as_indxr(p,q) ] = get(t,u);
              }
            }
    
            // 2-rdm
            for( int64_t p = 0; p < as_size; p++ )
            {
              int64_t t = ints.aorbs[p];
              for( int64_t q = 0;  q < as_size; q++ )
              {
                int64_t u = ints.aorbs[q];
                for( int64_t r = 0; r < as_size; r++ )
                {
                  int64_t v = ints.aorbs[r];
                  for( int64_t s = 0; s < as_size; s++ )
                  {
                    int64_t w = ints.aorbs[s];
                    rdm2_as[ as_indxr(p,q,r,s) ] = getPhys(t,u,v,w);
                  }
                }
              }
            }
    
            return rdms( as_size, rdm1_as, rdm2_as );
          }
    
          /**
           * @brief Construct 1- and 2-RDMs from input RDMs, which are
           *        constrained within a given active space. The active space
           *        is defined by an integrals instance, which is defined over
           *        the same number of orbitals as *this RDMs.  
           * 
           * @param [in] const intgrls::integrals &ints: Integrals instance,
           *             contains the active space specifications.
           * @param [in] VecD &rdm1_as: 1-RDM for the active orbitals defined
           *             by ints. 
           * @param [in] VecD &rdm2_as: 2-RDM for the active orbitals defined
           *             by ints. 
           *
           * @author Carlos Mejuto Zaera
           * @date 05/04/2021
           */
          void UpdateFromAS( intgrls::integrals &ints, VecD &rdm1_as, VecD &rdm2_as )
          {
            // Update the rdm's with the rdm's obtained from
            // an active space calculation. The active space
            // is defined by ints.
            int64_t as_size = ints.aorbs.size();
            if( as_size * as_size != rdm1_as.size() )
              throw( "Error in UpdateFromAS! Active space in integrals does not match the input rdms!" );
            std::fill( rdm1.begin(), rdm1.end(), 0. );
            std::fill( rdm2.begin(), rdm2.end(), 0. );
            rdm::indexer as_indxr( as_size );
            // First, 1-rdm
            for( auto const &i: ints.iorbs )
              rdm1[ indexer(i,i) ] = 2.;
            for( int64_t ac1 = 0; ac1 < as_size; ac1++ )
            {
              int64_t t = ints.aorbs[ac1];
              for( int64_t ac2 = 0; ac2 < as_size; ac2++ )
              {
                int64_t u = ints.aorbs[ac2];
                rdm1[ indexer(t,u) ] = rdm1_as[ as_indxr(ac1, ac2) ];
              }
            } 
            // Finally, 2-rdm
            for( auto const &i: ints.iorbs )
            {
              for( auto const &j: ints.iorbs )
              {
                rdm2[ indexer(i,j,i,j) ] =  4.;
                rdm2[ indexer(i,j,j,i) ] = -2.;
              }
              for( int64_t ac1 = 0; ac1 < as_size; ac1++ )
              {
                int64_t t = ints.aorbs[ac1];
                for( int64_t ac2 = 0; ac2 < as_size; ac2++ )
                {
                  int64_t u = ints.aorbs[ac2];
                  rdm2[ indexer(i,t,i,u) ] =  2. * rdm1_as[ as_indxr(ac1, ac2) ];
                  rdm2[ indexer(i,t,u,i) ] = -1. * rdm1_as[ as_indxr(ac1, ac2) ];
                  rdm2[ indexer(t,i,i,u) ] = -1. * rdm1_as[ as_indxr(ac1, ac2) ];
                  rdm2[ indexer(t,i,u,i) ] =  2. * rdm1_as[ as_indxr(ac1, ac2) ];
                }
              }
            }
            for( int64_t ac1 = 0; ac1 < as_size; ac1++ )
            {
              int64_t t = ints.aorbs[ac1];
              for( int64_t ac2 = 0; ac2 < as_size; ac2++ )
              {
                int64_t u = ints.aorbs[ac2];
                for( int64_t ac3 = 0; ac3 < as_size; ac3++ )
                {
                  int64_t v = ints.aorbs[ac3];
                  for( int64_t ac4 = 0; ac4 < as_size; ac4++ )
                  {
                    int64_t w = ints.aorbs[ac4];
                    rdm2[ indexer(t,u,v,w) ] = rdm2_as[ as_indxr(ac1,ac2,ac3,ac4) ];
                  }
                }
              }
            } 
          }
    
        };
    }// namespace rdm
  }// namespace ed
}// namespace cmz

#endif
