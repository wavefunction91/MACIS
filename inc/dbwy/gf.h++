/**
 * @brief Collection of routines to compute Green's functions
 *        within ED or CI-based approaches. By this, we mean
 *        any approximated approach where the wave function for
 *        which we want to compute the Green's function is written
 *        as a list of determinants. 
 *
 * @author Carlos Mejuto Zaera
 * @date 25/04/2022
 */
#ifndef __INCLUDE_CMZED_GF__
#define __INCLUDE_CMZED_GF__
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <chrono>
#include "cmz_ed/utils.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/bandlan.h++"
#include "cmz_ed/hamil.h++"
#include "sd_operations.hpp"
#include "hamiltonian_generator.hpp"
#include "csr_hamiltonian.hpp"
#include <ips4o.hpp>

namespace cmz
{
  namespace ed
  {

    typedef std::numeric_limits< double > dbl;
    typedef std::chrono::high_resolution_clock Clock;

    template<size_t nbits>
    inline double GetInsertionUpSign( const std::bitset<nbits> &st, size_t orb )
    {
      std::bitset<nbits> mask = dbwy::full_mask<nbits>( orb );
      return ((st & mask).count() % 2 == 1 ? -1. : 1.);
    }

    template<size_t nbits>
    inline double GetInsertionDoSign( const std::bitset<nbits> &st, size_t orb )
    {
      std::bitset<nbits> mask = dbwy::full_mask<nbits>( orb + nbits/2 );
      return ((st & mask).count() % 2 == 1 ? -1. : 1.);
    }

    /**
     * @brief Routine to compute single diagonal Green's function element. 
     *        Essentially, evaluates the resolvent
     *        G(w) = <state_0| 1/(freq - H) |state_0>
     *        using one-band Lanczos and continuous fraction representation. 
     *
     * @param [in] const VectorXd &state_0: Reference state, for which to evaluate
     *             the resolvent. 
     * @param [in] const MatOp &H: Matrix operator representing the effective 
     *             Hamiltonian.
     * @param [in] const VecCompD &freqs: Frequency grid over which to evaluate the 
     *             resolvent.
     * @param [inout] VecCompD &gf: On successful exit, green's function along the
     *                frequency grid.
     * @param [in] int nLanIts: Max. number of Lanczos iterations.
     * @param [in] bool saveABtofile: If true, save the alpha and beta parameters
     *             from the Lanczos calculation to file.
     * @author Carlos Mejuto Zaera
     * @date 28/01/2022
     */
    template<class MatOp>
    void GF_Diag(
      const VectorXd &state_0, 
      const MatOp &H, 
      const VecCompD  &freqs, 
      VecCompD &gf, 
      double E0,
      bool ispart,
      int nLanIts = 1000,
      bool saveABtofile = false,
      std::string fpref = ""){
      
      //FIRST, WE HAVE TO COMPUTE THE LANCZOS alphas AND betas
      double tol = 1.E-6;
      VecD alphas, betas;
    
    
      MyLanczos(state_0, H, nLanIts, alphas, betas, tol);
    
      int kry_size = 0;
      for(int i = 1; i < nLanIts; i++){
        if(abs(betas[i]) <= tol) break;
        kry_size++;
      }
    
      double sign = ispart ? -1. : 1.;

      if( saveABtofile )
      {
        std::ofstream ofile( fpref + "alphas.dat", std::ios::out);
	ofile.precision( dbl::max_digits10 );
        for(int i = 0; i <= kry_size; i++) ofile << scientific << alphas[i] << std::endl;
        ofile.close();
        ofile.open(fpref + "betas.dat", std::ios::out);
	ofile.precision( dbl::max_digits10 );
        for(int i = 0; i <= kry_size; i++) ofile << scientific << betas[i] << std::endl;
        ofile.close();
      }
    
      //FINALLY, WE CAN COMPUTE THE GREEN'S FUNCTION
      gf.clear();
      gf.resize(freqs.size(), CompD(0.,0.));
      
      #pragma omp parallel for
      for (int indx_w = 0; indx_w < freqs.size(); indx_w++){
          gf[indx_w] = betas[kry_size] * betas[kry_size] / (freqs[indx_w] + sign * ( alphas[kry_size] - E0 ) );
          for(int i = kry_size - 1; i >= 0; i--) gf[indx_w] = betas[i] * betas[i] / (freqs[indx_w] + sign * ( alphas[i] - E0 ) - gf[indx_w]); //SINCE I CHOSE betas[0] = normpsi^2
      }
    } 

    template<size_t nbits>
    class BitSetComparator {
      public:
        bool operator()(const std::bitset<nbits>& c1, const std::bitset<nbits>& c2) const
        {
            return dbwy::bitset_less( c1, c2 ); 
        }
    };

    /**
     * @brief Routine to build basis to compute the basis for Green's function
     *        calculation on the state |wfn> considering orbital orb. Templated
     *        to allow for differente slater determinant implementations (thinking
     *        beyond ED). 
     *
     * @tparam class DetType: Slater determiant type. Should implement functions to check
     *         for orbital occupation, returning state bitstring and finding single excitations
     *         accounting for active space structure.
     * @tparam DetStateType: State bitstring type. Encodes fermionic state in second quatization,
     *         should allow for basic bit operations, and being an index in a dictionary.
     *
     * @param [in] int orb: Orbital index for the Green's function basis. 
     * @param [in] bool sp_up: If true, the orbital is spin up, otherwise it's spin
     *             down. 
     * @param [in] bool is_part: If true, compute GF basis for particle sector (particle addition), 
     *             otherwise compute it for hole sector (particle removal).
     * @param [in] const VectorXd: Coefficient of the wave function describing the
     *             state whose Green's function we are computing. 
     * @param [in] const std::vector<DetType> &old_basis: Basis of determinants describing the
     *             wave function wfn. 
     * @param [out] std::vector<DetType> &new_basis: On return, the built basis for the Green's
     *              function calculation.
     * @param [in] const VecD &occs: Occupation numbers of all orbitals. Used to define
     *             active spaces.
     * @param [in] const Input_t &input: Includes input parameters, in particular:
     *             o) double asThres: Threshold for the occupation numbers to define the
     *             active space. If 0+asThres <= occ[i] <= 2 - asThres, then the orbital 
     *             i is added into the active space.
     *             o) double GFseedThreshold: Threshold to determine from which determinants to get
     *             excitations in the basis construction.
     *             o) int tot_SD: Number of layers of single excitations to build the basis.
     *             o) size_t trunc_size: Max. size for the Green's function basis.
     *
     * @author Carlos Mejuto Zaera
     * @date 28/01/2022
     */
    template<size_t nbits, typename index_t = int32_t>
    void get_GF_basis_AS_1El(
      int orb, 
      bool sp_up, 
      bool is_part, 
      const VectorXd &wfn, 
      const std::vector<std::bitset<nbits> > &old_basis, 
      std::vector<std::bitset<nbits> > &new_basis, 
      const VecD &occs, 
      const Input_t &input)
    {
      //CARLOS: BUILDS BASIS FOR THE ADD SPACE NEEDED TO DESCRIBE THE DIAGONAL PARTICLE
      //GF ELEMENT OF ORBITAL orb.
    
      size_t      norbs = getParam<int>( input, "norbs" );
      size_t trunc_size = getParam<int>( input, "trunc_size" );
      int        tot_SD = getParam<int>( input,     "tot_SD" );
      double GFseedThreshold = getParam<double>( input, "GFseedThresh" );
      double    asThres = getParam<double>( input,   "asThresh" );

      cout << "COMPUTING GF SPACE IN *" << (is_part? "PARTICLE" : "HOLE") << "* SECTOR FOR ORBITAL " << orb << ", WITH SPIN *" << (sp_up ? "UP" : "DOWN") << "*" << endl;

      time_t loop1 = time(NULL);
      auto loop1C = Clock::now();
      size_t ndets = old_basis.size(); 
      size_t cgf   = -1;
      size_t sporb = sp_up ? orb : orb + nbits / 2;
      std::bitset<nbits> uni_string;
      std::vector<std::bitset<nbits> > founddets;
      founddets.reserve(trunc_size);
      std::map<std::bitset<nbits>, size_t, BitSetComparator<nbits> > founddet_pos, basedet_pos;
      typename std::map<std::bitset<nbits>, size_t, BitSetComparator<nbits> >::iterator it;
      //ACTIVE SPACE
      std::vector<uint32_t> as_orbs;
      for(size_t i = 0; i < occs.size(); i++){
        if(occs[i] >= asThres && occs[i] <= (1. - asThres)) as_orbs.push_back(i);
      }
      cout << "ACTIVE SPACE:   [";
      for(int iii = 0; iii < as_orbs.size(); iii++) cout << as_orbs[iii] << ", ";
      cout << "]" << endl;
    
      //INITIALIZE THE DICTIONARY OF BASE DETERMINANTS
      for(size_t iii = 0; iii < ndets; iii++)
        basedet_pos[old_basis[iii]] = iii;
      
      //LOOP OVER ALL STATES IN THE BASIS AND BUILD THE BASIS
      for(size_t iii = 0; iii < ndets; iii++)
      {
        size_t norb1 = orb;
        //CHECK WHETHER IT CORRESPONDS TO this GF: a_i^+|wfn> OR a_i|wfn>
        bool ingf = false;
        if( is_part && !old_basis[iii][sporb]) // PARTICLE
            ingf = true;
        else if( !is_part && old_basis[iii][sporb]) // HOLE
            ingf = true;
	if( ingf )
        {
          //YES, ADD TO LIST
          std::bitset<nbits> temp = old_basis[iii]; 
          temp.flip(sporb);
          it = founddet_pos.find(temp);
          if(it == founddet_pos.end())
          {
            cgf++;
            founddet_pos[temp] = cgf;
            founddets.push_back(temp);
          }
        }
      }
    
      //IF NO STATE FOUND, THIS BASIS IS EMPTY
      if(cgf+1 == 0)
      {
        new_basis.resize(cgf+1);
        return;
      }
      //NOW, ADD SINGLE-DOUBLES. THIS AMOUNTS TO ADDING THE
      //SINGLE EXCITATIONS ON TOP OF FOUNDDET
      cout << "BEFORE ADDING SINGLE-DOUBLES, cgf = " << cgf << endl;
      size_t orig_cgf = cgf;
    
      std::cout << "Nr. OF STATES: " << cgf+1 << endl;
    
      size_t norb = orb;
      VectorXd b = Eigen::VectorXd::Zero(cgf+1);
      //COMPUTE VECTOR b IN THE NEW BASIS
      for(size_t ndet = 0; ndet < cgf+1; ndet++)
      {
        //CHECK, CAN ndet COME FROM ai^+|GS> / ai|wfn> WITH THE ORBITAL *orb?
        bool fromwfn = false;
        if( is_part && founddets[ndet][sporb])
            fromwfn = true;
        else if ( !is_part && !founddets[ndet][sporb] )
            fromwfn = true;
        if( fromwfn )
        {
          //YES, CHECK WHETHER THE GENERATING STATE COMES FROM THE BASE DETERMINANT SPACE
          std::bitset<nbits> temp = founddets[ndet];
	  temp.flip(sporb);
          it = basedet_pos.find(temp);
          if(it != basedet_pos.end()){
            //IT DOES COME INDEED FROM A DETERMINANT IN THE ORIGINAL GROUND STATE
            //THUS, IT CONTRIBUTES TO wfns1
            double sign = sp_up ? GetInsertionUpSign( temp, orb ) : GetInsertionDoSign( temp, orb );
            double fac = wfn(it->second); 
            b(ndet) += fac * sign; 
          }
        }
      }
    
      cout << "GOING TO ITERATIVELY ADD SINGLES AND DOUBLES!" << endl;
      cout << "--BEFORE ADDING SINGLE-DOUBLES, nterms = " << cgf+1 << endl;
      size_t orig_nterms = cgf+1;
    
      int startSD = 0, endSD = orig_nterms-1, GFseed = 0;
      for(int nSD = 1; nSD <= tot_SD && cgf <= trunc_size; nSD++){
        for(int iii = startSD; iii <= endSD && cgf <= trunc_size; iii++){
          if(nSD == 1) if(abs(b(iii)) < GFseedThreshold) continue; //FROM THE ORIGINAL SET, ONLY CONSIDER THE MOST IMPORTANT ONES
          //GET SINGLES
          GFseed += (nSD == 1) ? 1 : 0;
          std::vector<std::bitset<nbits> > tdets;
          dbwy::generate_singles_spin_as( norbs, founddets[iii], tdets, as_orbs );

          for(size_t jjj = 0; jjj < tdets.size(); jjj++){
            it = founddet_pos.find(tdets[jjj]);
            if(it == founddet_pos.end()){ // FOR ZERO STATES ONLY, ADD: and matches > 0
              cgf++;
              founddet_pos[tdets[jjj]] = cgf;
              founddets.push_back(tdets[jjj]);
            }
          }
        }
        startSD = endSD + 1;
        endSD = cgf;
      }
      cout << "--AFTER CUTTING BY: " << GFseedThreshold << ", GF-SPACE WITH GF SEED: " << GFseed << ", WE HAVE STATES: " << cgf+1 << endl;
    
      //NOW THAT WE FOUND THE BASIS, JUST STORE IT
      new_basis.resize(cgf+1);
      new_basis.assign(founddets.begin(), founddets.end());
    }

    /**
     * @brief Routine to prepare the wave functions for the Lanczos computation of the Green's function (be it 
     *        Band Lanczos or regular Lanczos). 
     *
     * @tparam class DetType: Slater determiant type. Should implement functions to check
     *         for orbital occupation, returning state bitstring and finding single excitations
     *         accounting for active space structure.
     * @tparam DetStateType: State bitstring type. Encodes fermionic state in second quatization,
     *         should allow for basic bit operations, and being an index in a dictionary.
     *
     * @param [in] const VectorXd &base_wfn: Wave function from which to compute the Green's function. 
     * @param [in] const VecInt &GF_orbs: Orbital indices for which to compute the GF. 
     * @param [in] const std::vector<bool> &is_up: Flags regarding the spin of each orbital. True means
     *             spin up, false means spin down.
     * @param [in] const std::vector<DetType> &base_dets: Vector of the Slater determinants in the basis. 
     * @param [in] const std::vector<DetType> &GF_gets: Vector of the Slater determinants in the GF basis. 
     * @param [in] bool is_part: Flag to determine GF sector. For true, we are in the particle sector. For
     *             false we are in the hole sector.
     * @param [out] VecInt& todelete: On return, contains a list of the orbitals for which the
     *              Green's function vector ai/ai+|base_wfn> vanishes, and hence is
     *              eliminated from the list of wave functions. 
     * @param [in] double zero_thresh: Threshold to decide whether a computed vector is zero or not, judging by
     *             the magnitude of the norm.
     *
     * @author Carlos Mejuto Zaera
     * @date 01/02/2022
     */
    template<size_t nbits, typename index_t = int32_t>
    std::vector<VecD> BuildWfn4Lanczos( 
       const VectorXd &base_wfn, 
       const VecInt &GF_orbs, 
       const std::vector<bool> &is_up, 
       const std::vector<std::bitset<nbits> > &base_dets, 
       const std::vector<std::bitset<nbits> > &GF_dets, 
       bool is_part, 
       VecInt &todelete, 
       double zero_thresh = 1.E-7 )
    {
      //INITIALIZE THE DICTIONARY OF BASE DETERMINANTS
      std::map<std::bitset<nbits>, size_t, BitSetComparator<nbits> > base_dets_pos;
      for(size_t iii = 0; iii < base_dets.size(); iii++)
        base_dets_pos[base_dets[iii]] = iii;
      
      //PREPARE THE WAVEFUNCTIONS FOR THE BAND LANCZOS
      size_t nterms = GF_dets.size(); 
      std::vector<VecD> wfns(GF_orbs.size(), VecD(nterms,0.));
      for(size_t iorb = 0; iorb < GF_orbs.size(); iorb++){
        int    orb = GF_orbs[iorb];
        bool sp_up = is_up[iorb];
        //int  sporb = sp_up ? orb : orb + Norbs;
        int  sporb = sp_up ? orb : orb + nbits/2;
        //BUILD THE WAVEFUNCTION FOR ORBITAL orb
        for(size_t ndet = 0; ndet < nterms; ndet++){
          //CHECK, CAN ndet COME FROM ai^+|GS> *OR* ai|GS> WITH THE ORBITAL orb?
          bool in_gf = false;
	  if( is_part && GF_dets[ndet][sporb]) // PARTICLE
              in_gf = true;
          else if( !is_part && !GF_dets[ndet][sporb] ) // HOLE
              in_gf = true;
          if( in_gf )
          {
            //YES, CHECK WHETHER THE GENERATING STATE COMES FROM THE BASE DETERMINANT SPACE
            std::bitset<nbits> temp( GF_dets[ndet] );
            temp.flip(sporb);
            typename std::map<std::bitset<nbits>, size_t, BitSetComparator<nbits> >::const_iterator it = base_dets_pos.find(temp);
            if(it != base_dets_pos.end()){
              //IT DOES COME INDEED FROM A DETERMINANT IN THE ORIGINAL GROUND STATE
              //THUS, IT CONTRIBUTES TO wfns1
              double sign = sp_up ? GetInsertionUpSign( temp, orb ) : GetInsertionDoSign( temp, orb );
              double fac = base_wfn(it->second); 
              wfns[iorb][ndet] += fac * sign; 
            }
          }
        }
      }

      //CHECK WHETHER ANY OF THE VECTORS IS EXACTLY ZERO. IF SO, TAKE IT OUT!
      todelete.clear();
      for(int orb_indx = 0; orb_indx < GF_orbs.size(); orb_indx++){
        double st_nrm = 0.;
        #pragma omp declare reduction \
          (Vsum:double:omp_out=omp_out+omp_in)\
          initializer(omp_priv=0.)
        #pragma omp parallel for reduction (Vsum:st_nrm)
        for(size_t iii = 0; iii < nterms; iii++) st_nrm += wfns[orb_indx][iii] * wfns[orb_indx][iii];
        if(abs(st_nrm) <= zero_thresh) todelete.push_back(orb_indx);
       }
       for(int i = 0; i < todelete.size(); i++) wfns.erase(wfns.begin() + todelete[i] - i);
       cout << "ORBITALS WITH NO CORRESPONING ADD-VECTOR: [";
       for(int i = 0; i < todelete.size(); i++) cout << todelete[i] << ", ";
       cout << "]" << endl;

       return wfns;
    }

    /**
     * @brief Routine to write Green's function to file. It stores the frequency grid together with
     *        the full GF matrix. Also, in case of a multi-orbitla GF, it stores the orbital
     *        indices in a separate file. 
     *
     * @param [in] const std::vector<std::vector<std::vector<std::complex double> > > > &GF: 
     *             GF to store. Written as GF[freq-axis][orb1][orb2].
     * @param [in] const VecCompD &ws: Frequency grid.
     * @param [in] const VecInt &GF_orbs: Orbital indices for the Green's function, as requested
     *             originally in the GF computation. Some of them may not have been actually
     *             used, if the corresponding ai/ai+|wfn0> states were zero.
     * @param [in] const VecInt &todelete: List of deleted orbital indices in the case just described.
     * @param [in] const bool is_part: Flag to label GF file depending on whether it is a particle
     *             GF (True), or a hole GF (False).
     *
     * @author Carlos Mejuto Zaera
     * @date 02/02/2022
     */
    void write_GF( const std::vector<std::vector<std::vector<std::complex<double> > > > &GF, const VecCompD &ws, const VecInt &GF_orbs, const VecInt &todelete, const bool is_part );

    /**
     * @brief Routine to run Green's function calculation at zero temperature from some input ref. wave function.
     *        Allows for choosing to compute particle/hole GF, using normal/band Lanczos,  
     *        and simplifying the calculation by virtue of exploiting some active space structure.
     *
     * @tparam class DetType: Slater determiant type. Should implement functions to check
     *         for orbital occupation, returning state bitstring and finding single excitations
     *         accounting for active space structure.
     * @tparam DetStateType: State bitstring type. Encodes fermionic state in second quatization,
     *         should allow for basic bit operations, and being an index in a dictionary.
     *
     * @param [out] std::vector<std::vector<std::vector<std::complex<double> > > >: On output,
     *             contains the computed Green's function, in format GF[freq.][orb1][orb2].
     * @param [in] const VectorXd &wfn0: Reference wave function from which to compute the GF.
     * @param [in] const FermionHamil &H: Fermionic Hamiltonian defining the system.
     * @param [in] const std::vector<DetType> &base_dets: Basis of Slater determinants for the
     *             description of wfn0. 
     * @param [in] const double energ: Energy of reference state wfn0.
     * @param [in] const bool is_part: Flag to determine GF sector. For true, we are in 
     *             the particle sector. Otherwise, we are in the hole sector.
     * @param [in] const VecCompD &ws: Frequency grid over which to compute the Green's function.
     * @param [in] const VecD &occs: Occupation numbers for each orbital.
     * @param [in] const Input_t input: Dictionary with various parameters for Green's function calculation.
     *
     * @author Carlos Mejuto Zaera
     * @date 01/02/2022
     */
    template<size_t nbits, typename index_t = int32_t>
    void RunGFCalc(
      std::vector<std::vector<std::vector<std::complex<double> > > > &GF,
      const VectorXd &wfn0, 
      dbwy::HamiltonianGenerator<nbits> &Hgen, 
      const std::vector<std::bitset<nbits> > &base_dets, 
      const double energ,
      const bool is_part, 
      const VecCompD& ws, 
      const VecD & occs, 
      const Input_t &input)
    {
    
      // READ INPUT
      const size_t trunc_size = getParam<int>( input, "trunc_size" );
      const int    tot_SD     = getParam<int>( input,     "tot_SD" );
      const double GFseedThreshold = getParam<double>( input, "GFseedThresh");
      const double asThres    = getParam<double>( input, "asThresh" );
      const bool use_bandLan  = getParam<bool>( input, "use_bandLan");
      const VecInt GF_orbs_basis = getParam<VecInt>( input, "GF_orbs_basis" );
      const VecInt GF_orbs_comp  = getParam<VecInt>( input,  "GF_orbs_comp" );
      const std::vector<bool> is_up_basis = getParam<std::vector<bool> >( input, "is_up_basis" );
      const std::vector<bool> is_up_comp  = getParam<std::vector<bool> >( input,  "is_up_comp" );
      double h_el_tol = 1.E-6;
     
      int nLanIts;
      try{ nLanIts = getParam<int>( input, "nLanIts" ); } catch(...){ nLanIts = 1000; } 
      bool print, writeGF;
      try{ print = getParam<bool>( input, "printGF" ); } catch(...){ print = false;}
      try{ writeGF = getParam<bool>( input, "writeGF" ); } catch(...){ writeGF = false;}
      bool saveGFmats;
      try{ saveGFmats = getParam<bool>( input, "saveGFmats" ); } catch(...){ saveGFmats = false;}

      time_t loop1 = time(NULL), loop2 = time(NULL);
      auto loop1C = Clock::now(), loop2C = Clock::now();
      size_t ndets = base_dets.size();

      //FIRST, BUILD THE BASIS SEQUENTIALLY BY FORMING THE BASES OF EACH ORBITAL AND
      //ADDING THEM TOGETHER
      std::vector<std::bitset<nbits> > gf_dets, gf_dets_tmp;
      for(size_t iorb = 0; iorb < GF_orbs_basis.size(); iorb++){
        get_GF_basis_AS_1El<nbits, index_t>( GF_orbs_basis[iorb], is_up_basis[iorb], is_part, wfn0, base_dets, 
                                                    gf_dets_tmp, occs, input );
        gf_dets.insert( gf_dets.end(), gf_dets_tmp.begin(),
                                       gf_dets_tmp.end() );

        gf_dets_tmp.clear();
        ips4o::parallel::sort( gf_dets.begin(), gf_dets.end(), 
          []( auto x, auto y ) { return dbwy::bitset_less(x, y); });
        typename std::vector<std::bitset<nbits> >::iterator b_it = std::unique(gf_dets.begin(), gf_dets.end());
        gf_dets.resize(std::distance(gf_dets.begin(), b_it)); 
        std::cout << "---> BASIS HAS NOW: " << gf_dets.size() << " ELMENTS!! BY ORBITAL " << iorb + 1 << "/" << GF_orbs_basis.size() << std::endl;
      }
    
      size_t nterms = gf_dets.size();
      cout << "---> FINAL ADD BASIS HAS " << nterms << " ELEMENTS" << endl;
    
      loop1 = time(NULL);
      loop1C = Clock::now();
      auto hamil = dbwy::make_dist_csr_hamiltonian<index_t>( MPI_COMM_WORLD, gf_dets.begin(), 
                                                             gf_dets.end(), Hgen, h_el_tol );
      loop2 = time(NULL);
      loop2C = Clock::now();
      std::cout << setprecision(3)<<"Building " << (is_part ? "*PARTICLE*" : "*HOLE*") << " Hamiltonian: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(loop2C - loop1C).count())/1000 << std::endl;
      //NOW, PERFORM THE BAND LANCZOS ON THE TRUNCATED SPACE
      //WE ALREADY BUILT THE HAMILTONIANS
    
      if(nterms < nLanIts)
        nLanIts = nterms;

      //PREPARE THE WAVEFUNCTIONS FOR THE BAND LANCZOS
      VecInt todelete;
      std::vector<VecD> wfns = BuildWfn4Lanczos<nbits, index_t>( wfn0, GF_orbs_comp, is_up_comp, 
                                                                         base_dets, gf_dets, is_part, todelete );
    
      // //ACTUALLY COMPUTE THE GF!
       time_t GF_loop1 = time(NULL);
       auto GF_loop1C = Clock::now(); 
    
       if( use_bandLan )
       {
         BandResolvent(hamil, wfns, ws, GF, nLanIts, energ, is_part, print, saveGFmats);
       }
       else
       {
         //DO SIMPLE LANCZOS FOR ALL GF ELEMENTS
         SparsexDistSpMatOp hamil_wrap( hamil );
         GF.resize(ws.size(), std::vector<std::vector<std::complex<double> > >(wfns.size(), std::vector<std::complex<double> >(wfns.size(), std::complex<double>(0.,0.)) )); 
         for(int i = 0; i < wfns.size(); i++)
         {
           std::vector<std::complex<double> > tGF;
           //DIAGONAL ELEMENT
           std::cout << "DOING ELEMENT (" << i << ", " << i << ")" << std::endl;
           VectorXd twfn = Eigen::Map<VectorXd, Eigen::Unaligned>(wfns[i].data(), nterms);
	   std::string fpref_basis = is_part ? "particle" : "hole";
	   std::string fpref = fpref_basis + "_" + std::to_string( i ) + "_" + std::to_string( i );
           GF_Diag<SparsexDistSpMatOp>(twfn, hamil_wrap, ws, tGF, energ, is_part, nLanIts, saveGFmats, fpref);
           for(int iw = 0; iw < ws.size(); iw++) 
             GF[iw][i][i] = tGF[iw];
           for(int j = i+1; j < wfns.size(); j++)
           {
             //OFF DIAGONAL ELEMENTS
             std::cout << "DOING ELEMENT (" << i << ", " << j << ")" << std::endl;
             for(size_t iii = 0; iii < nterms; iii++)
               twfn(iii) = wfns[i][iii] + wfns[j][iii];
	     fpref = fpref_basis + "_" + std::to_string( i ) + "_" + std::to_string( j ) + "_a";
             GF_Diag<SparsexDistSpMatOp>(twfn, hamil_wrap, ws, tGF, energ, is_part, nLanIts, saveGFmats, fpref);
             for(int iw = 0; iw < ws.size(); iw++) 
               GF[iw][i][j] += 0.25 * tGF[iw];
             for(size_t iii = 0; iii < nterms; iii++)
               twfn(iii) = wfns[i][iii] - wfns[j][iii];
	     fpref = fpref_basis + "_" + std::to_string( i ) + "_" + std::to_string( j ) + "_b";
             GF_Diag<SparsexDistSpMatOp>(twfn, hamil_wrap, ws, tGF, energ, is_part, nLanIts);
             for(int iw = 0; iw < ws.size(); iw++)
             { 
               GF[iw][i][j] -= 0.25 * tGF[iw];
               GF[iw][j][i]  = GF[iw][i][j];
             }
           }
         }
       }
    
       time_t GF_loop2 = time(NULL);
       auto GF_loop2C = Clock::now(); 
       std::cout << setprecision(3)<<"Computing GF with " << (use_bandLan ? " *Band Lanczos*" : "*Regular Lanczos*") << double(std::chrono::duration_cast<std::chrono::milliseconds>(GF_loop2C - GF_loop1C).count())/1000 << std::endl;
    
       if( writeGF ) 
         write_GF( GF, ws, GF_orbs_comp, todelete, is_part );
    }

  }// namespace ed
}// namespace cmz

#endif
