#pragma once
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include "cmz_ed/utils.h++"
#include "dbwy/csr_hamiltonian.hpp"
#include "dbwy/davidson.hpp"
#include "dbwy/sd_build.hpp"
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include "cmz_ed/freq_grids.h++"
#include "dbwy/gf.h++"

#include <bitset>

#include "dbwy/asci_util.hpp"

namespace dbwy {

template <size_t N>
std::vector<std::bitset<N>> build_combs( uint64_t nbits, uint64_t nset ) {

  std::vector<bool> v(nbits, false);
  std::fill_n( v.begin(), nset, true );
  std::vector<std::bitset<N>> store;

  do {

    std::bitset<N> temp = 0ul;
    std::bitset<N> one  = 1ul;
    for( uint64_t i = 0; i < nbits; ++i )
    if( v[i] ) {
      temp = temp | (one << i);
    }
    store.emplace_back(temp);

  } while ( std::prev_permutation( v.begin(), v.end() ) );

  return store;

}

template <size_t N>
std::vector<std::bitset<N>> build_hilbert_space(
  size_t norbs, size_t nalpha, size_t nbeta
) {

  // Get all alpha and beta combs
  auto alpha_dets = build_combs<N>( norbs, nalpha );
  auto beta_dets  = build_combs<N>( norbs, nbeta  );

  std::vector<std::bitset<N>> states;
  states.reserve( alpha_dets.size() * beta_dets.size() );
  for( auto alpha_det : alpha_dets )
  for( auto beta_det  : beta_dets  ) {
    std::bitset<N> state = alpha_det | (beta_det << (N/2));
    states.emplace_back( state );
  }

  return states;
 
}

template< size_t nbits>
auto run_asci_w_GF(
  const cmz::ed::Input_t &input
) {

  double Eret = 0.;
  std::vector<double> X_local; // Eigenvectors
  std::vector<std::bitset<nbits>> dets;
  vector<vector<vector<complex<double> > > > GF;
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  bool   quiet  = cmz::ed::getParam<bool>( input, "quiet" );
  bool real_ints = cmz::ed::getParam<bool>( input, "real_ints" ); 
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );
  
  if( norb > nbits/2 ) throw std::runtime_error("Not Enough Bits...");
  
  
  // Read in the integrals 
  bool just_singles;
  cmz::ed::intgrls::integrals ints(norb, fcidump, just_singles, real_ints);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Hamiltonian Matrix Element Generator
  //dbwy::DoubleLoopHamiltonianGenerator<nbits> 
  //  ham_gen( norb, ints.u.data(), ints.t.data() );
  dbwy::SDBuildHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );
  ham_gen.SetJustSingles( just_singles );
  
  // Compute HF Energy
  std::vector<double> orb_ens( norb );
  for( int i = 0; i < norb; i++ )
    orb_ens[i] = ints.get( i, i ); 
  const std::bitset<nbits> hf_det = 
    dbwy::canonical_hf_determinant<nbits>( nalpha,nbeta, orb_ens );
  //const std::bitset<nbits> hf_det = 
  //  dbwy::canonical_hf_determinant<nbits>(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if( world_rank == 0 && !quiet ) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }
  
  
  auto print_asci  = [&](double E) {
    if( world_rank == 0 && !quiet ) {
      std::cout << "  * E(ASCI)   = " << E + ints.core_energy << " Eh" << std::endl;
      std::cout << "  * E_c(ASCI) = " << (E - EHF)*1000 << " mEh" << std::endl;
    }
  };
  
  //  Run ASCI
  size_t ntdets_max    = cmz::ed::getParam<int>( input, "ntdets_max" );
  size_t ncdets_max    = cmz::ed::getParam<int>( input, "ncdets_max" );
  size_t niter_max     = cmz::ed::getParam<int>( input, "niter_max" );
  double coeff_thresh  = cmz::ed::getParam<int>( input, "coeff_thresh" );
  int nstates          = 1;
  try { nstates = cmz::ed::getParam<int>( input, "nstates" ); } catch(...){ }
  int    n_orb_rots    = 0;
  try{ n_orb_rots = cmz::ed::getParam<int>( input, "n_orb_rots" ); } catch(...){ }
  {
  if(world_size != 1) throw "NO MPI"; // Disable MPI for now
  
  auto bitset_comp = [](auto x, auto y){ return dbwy::bitset_less(x,y); };
  
  // Staring with HF
  if(world_rank == 0 && !quiet)
    std::cout << "* Initializing ASCI with HF" << std::endl;
  dets.push_back(hf_det);
  X_local = {1.0};
  auto EASCI = EHF;
  print_asci( EASCI );
       
  // Grow wfn w/o refinement
  if( n_orb_rots > 0 )
    std::tie(EASCI, dets, X_local) = dbwy::asci_grow_with_rot( ntdets_max, ncdets_max, 8, EASCI,
      std::move(dets), std::move(X_local), ham_gen, norb, 1e-12, 100, 1e-8,
      print_asci, n_orb_rots, quiet, nstates );
  else
    std::tie(EASCI, dets, X_local) = dbwy::asci_grow( ntdets_max, ncdets_max, 8, EASCI,
      std::move(dets), std::move(X_local), ham_gen, norb, 1e-12, 100, 1e-8,
      print_asci, quiet, nstates );
  
  // Refine wfn
  std::tie(EASCI, dets, X_local) = dbwy::asci_refine( ncdets_max, 1e-6, niter_max,
    EASCI, std::move(dets), std::move(X_local), ham_gen, norb, 1e-12, 100, 1e-8,
    print_asci, quiet, nstates );
  
  Eret = EASCI + ints.core_energy;
  } // ASCI 

  // Green's function calculation
  { 
    double EASCI = Eret - ints.core_energy;
    Eigen::VectorXd psi0  = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( X_local.data(), X_local.size() );
    std::vector<double> ordm( norb*norb, 0. );
    ham_gen.form_rdms( dets.begin(), dets.end(), 
                       dets.begin(), dets.end(),
                       X_local.data(), ordm.data() );

    std::vector<double> occs(norb, 0.);
    for( int i = 0; i < norb; i++ )
      occs[i] += ordm[i + i * norb] / 2.;

    std::cout << "Occs: ";
    for( const auto oc : occs )
      std::cout << oc << ", ";
    std::cout << endl;

    auto ws = cmz::ed::GetFreqGrid( input );

    // Particle sector
    bool is_part = true;
    cmz::ed::RunGFCalc<nbits>( GF, psi0, ham_gen, dets, EASCI, is_part, ws, occs, input );

    // Hole sector
    is_part = false;
    std::vector<std::vector<std::vector<std::complex<double> > > > GF_tmp;
    cmz::ed::RunGFCalc<nbits>( GF_tmp, psi0, ham_gen, dets, EASCI, is_part, ws, occs, input );
    for( int i = 0; i < GF_tmp.size(); i++ )
      for( int j = 0; j < GF_tmp[i].size(); j++)
        for( int k = 0; k < GF_tmp[i][j].size(); k++)
          GF[i][j][k] += GF_tmp[i][j][k];
  } // GF
  return std::tuple( Eret, GF );
}

template< size_t nbits>
auto run_asci_w_1rdm(
  const cmz::ed::Input_t &input
) {

  double Eret = 0.;
  std::vector<double> X_local; // Eigenvectors
  std::vector<std::bitset<nbits>> dets;
  std::vector<double> ordm;
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  bool   quiet  = cmz::ed::getParam<bool>( input, "quiet" );
  bool real_ints = cmz::ed::getParam<bool>( input, "real_ints" );
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );
  bool read_wfn =
    cmz::ed::getParam<bool>( input, "read_wfn");
  std::string wfn_file;
  if( read_wfn )
    wfn_file = 
      cmz::ed::getParam<std::string>( input, "wfn_file" );
  
  if( norb > nbits/2 ) throw std::runtime_error("Not Enough Bits...");
  
  
  // Read in the integrals 
  bool just_singles;
  cmz::ed::intgrls::integrals ints(norb, fcidump, just_singles, real_ints);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Hamiltonian Matrix Element Generator
  //dbwy::DoubleLoopHamiltonianGenerator<nbits> 
  //  ham_gen( norb, ints.u.data(), ints.t.data() );
  dbwy::SDBuildHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );
  ham_gen.SetJustSingles( just_singles );
  
  // Compute HF Energy
  std::vector<double> orb_ens( norb );
  for( int i = 0; i < norb; i++ )
    orb_ens[i] = ints.get( i, i ); 
  const std::bitset<nbits> hf_det = 
    dbwy::canonical_hf_determinant<nbits>( nalpha,nbeta, orb_ens );
  //const std::bitset<nbits> hf_det = 
  //  dbwy::canonical_hf_determinant<nbits>(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if( world_rank == 0 && !quiet ) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }
  
  
  auto print_asci  = [&](double E) {
    if( world_rank == 0 && !quiet ) {
      std::cout << "  * E(ASCI)   = " << E + ints.core_energy << " Eh" << std::endl;
      std::cout << "  * E_c(ASCI) = " << (E - EHF)*1000 << " mEh" << std::endl;
    }
  };
  
  if(read_wfn) {
    // Read WFN
  
    {
      std::ifstream wfn(wfn_file);
      std::string line;
      std::getline( wfn, line );
      while( std::getline(wfn, line) ) {
        std::stringstream ss{line};
      std::string coeff, det;
      ss >> coeff >> det;
      dets.emplace_back( dbwy::from_canonical_string<nbits>(det));
      }
    }
    if(world_rank == 0 && !quiet)
      std::cout << "NDETS = " << dets.size() << std::endl;
    for( auto det : dets ) std::cout << det << std::endl;
    auto E = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
      1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD, quiet );
    print_asci( E );
    if(!quiet)
      std::cout << "**** MY PRINT ***" << std::endl;
    ham_gen.matrix_element(dets[0], dets[1]);
  } else {
  
  //  Run ASCI
  size_t ntdets_max    = cmz::ed::getParam<int>( input, "ntdets_max" );
  size_t ncdets_max    = cmz::ed::getParam<int>( input, "ncdets_max" );
  size_t niter_max     = cmz::ed::getParam<int>( input, "niter_max" );
  double coeff_thresh  = cmz::ed::getParam<int>( input, "coeff_thresh" );
  int nstates          = 1;
  try { nstates = cmz::ed::getParam<int>( input, "nstates" ); } catch(...){ }
  int    n_orb_rots    = 0;
  try{ n_orb_rots = cmz::ed::getParam<int>( input, "n_orb_rots" ); } catch(...) { }
  {
  if(world_size != 1) throw "NO MPI"; // Disable MPI for now
  
  auto bitset_comp = [](auto x, auto y){ return dbwy::bitset_less(x,y); };
  
  // Staring with HF
  if(world_rank == 0 && !quiet)
    std::cout << "* Initializing ASCI with HF" << std::endl;
  dets.push_back(hf_det);
  X_local = {1.0};
  auto EASCI = EHF;
  print_asci( EASCI );
       
  // Grow wfn w/o refinement
  if( n_orb_rots > 0 )
    std::tie(EASCI, dets, X_local) = dbwy::asci_grow_with_rot( ntdets_max, ncdets_max, 8, EASCI,
      std::move(dets), std::move(X_local), ham_gen, norb, 1e-12, 100, 1e-8,
      print_asci, n_orb_rots, quiet, nstates );
  else
    std::tie(EASCI, dets, X_local) = dbwy::asci_grow( ntdets_max, ncdets_max, 8, EASCI,
      std::move(dets), std::move(X_local), ham_gen, norb, 1e-12, 100, 1e-8,
      print_asci, quiet, nstates );
  
  // Refine wfn
  std::tie(EASCI, dets, X_local) = dbwy::asci_refine( ncdets_max, 1e-6, niter_max,
    EASCI, std::move(dets), std::move(X_local), ham_gen, norb, 1e-12, 100, 1e-8,
    print_asci, quiet, nstates );
  
  Eret = EASCI + ints.core_energy;
  
  bool print_wf;
  try{ print_wf = cmz::ed::getParam<bool>( input, "print_wf" ); }catch (...) { print_wf = false; }
  if( print_wf )
  {
    std::ofstream ofile( "final_asci_wf.dat", std::ios::out );

    ofile << dets.size() << std::endl;
    auto w = std::setw(25);

    for( int idet = 0; idet < dets.size(); idet++ )
       ofile << X_local[idet] << w << dbwy::to_canonical_string( dets[idet] ) << std::endl; 

    ofile.close();
  }
  // Compute the 1-rdm
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;
  typename std::vector<std::bitset<nbits>>::iterator det_st = dets.begin();
  typename std::vector<std::bitset<nbits>>::iterator det_en = dets.end();

  ordm.resize( norb*norb, 0. );

  auto rdm_st = clock_type::now();
  ham_gen.form_rdms( det_st, det_en, det_st, det_en, X_local.data(), ordm.data() );
  auto rdm_en = clock_type::now();

  if(world_rank == 0 && !quiet)
    std::cout << "  * 1-RDM calculation    = "
              << duration_type( rdm_en - rdm_st ).count() << std::endl;
  } // ASCI 

  }

  return std::tuple( Eret, ordm );
}

template< size_t nbits>
auto run_ed_w_1rdm(
  const cmz::ed::Input_t &input
) {

  double Eret = 0.;
  std::vector<double> X_local; // Eigenvectors
  std::vector<std::bitset<nbits>> dets;
  std::vector<double> ordm;
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  bool   quiet  = cmz::ed::getParam<bool>( input, "quiet" );
  bool real_ints = cmz::ed::getParam<bool>( input, "real_ints" );
  if( norb > 16 )
    throw( std::runtime_error("Error in run_ed_w_1rdm! Asked for ED with more than 16 orbitals!") );
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );
  bool read_wfn =
    cmz::ed::getParam<bool>( input, "read_wfn");
  std::string wfn_file;
  if( read_wfn )
    wfn_file = 
      cmz::ed::getParam<std::string>( input, "wfn_file" );
  
  if( norb > nbits/2 ) throw std::runtime_error("Not Enough Bits...");
  
  
  // Read in the integrals 
  bool just_singles;
  cmz::ed::intgrls::integrals ints(norb, fcidump, just_singles, real_ints);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Hamiltonian Matrix Element Generator
  //dbwy::DoubleLoopHamiltonianGenerator<nbits> 
  //  ham_gen( norb, ints.u.data(), ints.t.data() );
  dbwy::SDBuildHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );
  ham_gen.SetJustSingles( just_singles );
  
  // Compute HF Energy
  std::vector<double> orb_ens( norb );
  for( int i = 0; i < norb; i++ )
    orb_ens[i] = ints.get( i, i ); 
  const std::bitset<nbits> hf_det = 
    dbwy::canonical_hf_determinant<nbits>( nalpha,nbeta, orb_ens );
  //const std::bitset<nbits> hf_det = 
  //  dbwy::canonical_hf_determinant<nbits>(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if( world_rank == 0 && !quiet ) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }
  
  
  auto print_ed  = [&](double E) {
    if( world_rank == 0 && !quiet ) {
      std::cout << "  * E(ED)   = " << E + ints.core_energy << " Eh" << std::endl;
      std::cout << "  * E_c(ED) = " << (E - EHF)*1000 << " mEh" << std::endl;
    }
  };
  
  if(read_wfn) {
    // Read WFN
  
    {
      std::ifstream wfn(wfn_file);
      std::string line;
      std::getline( wfn, line );
      while( std::getline(wfn, line) ) {
        std::stringstream ss{line};
      std::string coeff, det;
      ss >> coeff >> det;
      dets.emplace_back( dbwy::from_canonical_string<nbits>(det));
      }
    }
    if(world_rank == 0 && !quiet)
      std::cout << "NDETS = " << dets.size() << std::endl;
    for( auto det : dets ) std::cout << det << std::endl;
    auto E = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
      1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD, quiet );
    print_ed( E );
    if(!quiet)
      std::cout << "**** MY PRINT ***" << std::endl;
    ham_gen.matrix_element(dets[0], dets[1]);
  } else {
  
  //  Run ED
  {
  if(world_size != 1) throw "NO MPI"; // Disable MPI for now
  
  int nstates          = 1;
  try { nstates = cmz::ed::getParam<int>( input, "nstates" ); } catch(...){ }
  // Fill the determinant list with all possible dets
  dets = dbwy::generate_full_hilbert_space<nbits>( norb, nalpha, nbeta );
  if(world_rank == 0 && !quiet)
    std::cout << "* NDETS in ED = " << dets.size() << std::endl;
  auto EED = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD, quiet, nstates );
  Eret = EED + ints.core_energy;
  print_ed( EED );
  
  bool print_wf;
  try{ print_wf = cmz::ed::getParam<bool>( input, "print_wf" ); }catch (...) { print_wf = false; }
  if( print_wf )
  {
    std::ofstream ofile( "final_ed_wf.dat", std::ios::out );

    struct wf_pair
    {
      std::string s;
      double coeff;
    };
    std::vector<wf_pair> pairs;
    pairs.reserve( dets.size() );
    for( int idet = 0; idet < dets.size(); idet++ )
    {
      wf_pair p = { to_canonical_string(dets[idet]), X_local[idet] };
      pairs.push_back( p );
    }
    std::sort( pairs.begin(), pairs.end(), []( wf_pair &a, wf_pair &b )->bool{ return abs(a.coeff) > abs(b.coeff); } );

    ofile << dets.size() << std::endl;
    auto w = std::setw(25);

    for( int idet = 0; idet < dets.size(); idet++ )
       ofile << pairs[idet].coeff << w << pairs[idet].s << std::endl; 

    ofile.close();
  }
  
  // Compute the 1-rdm
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;
  typename std::vector<std::bitset<nbits>>::iterator det_st = dets.begin();
  typename std::vector<std::bitset<nbits>>::iterator det_en = dets.end();

  ordm.resize( norb*norb, 0. );

  auto rdm_st = clock_type::now();
  ham_gen.form_rdms( det_st, det_en, det_st, det_en, X_local.data(), ordm.data() );
  auto rdm_en = clock_type::now();

  if(world_rank == 0 && !quiet)
    std::cout << "  * 1-RDM calculation    = "
              << duration_type( rdm_en - rdm_st ).count() << std::endl;
  } // ED

  }

  return std::tuple( Eret, ordm );
}

template< size_t nbits>
auto run_ed_w_GF(
  const cmz::ed::Input_t &input
) {

  double Eret = 0.;
  std::vector<double> X_local; // Eigenvectors
  std::vector<std::bitset<nbits>> dets;
  vector<vector<vector<complex<double> > > > GF;
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  bool   quiet  = cmz::ed::getParam<bool>( input, "quiet" );
  bool real_ints = cmz::ed::getParam<bool>( input, "real_ints" );
  if( norb > 16 )
    throw( std::runtime_error("Error in run_ed_w_1rdm! Asked for ED with more than 16 orbitals!") );
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );
  bool read_wfn =
    cmz::ed::getParam<bool>( input, "read_wfn");
  std::string wfn_file;
  if( read_wfn )
    wfn_file = 
      cmz::ed::getParam<std::string>( input, "wfn_file" );
  
  if( norb > nbits/2 ) throw std::runtime_error("Not Enough Bits...");
  
  
  // Read in the integrals 
  bool just_singles;
  cmz::ed::intgrls::integrals ints(norb, fcidump, just_singles, real_ints);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Hamiltonian Matrix Element Generator
  //dbwy::DoubleLoopHamiltonianGenerator<nbits> 
  //  ham_gen( norb, ints.u.data(), ints.t.data() );
  dbwy::SDBuildHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );
  ham_gen.SetJustSingles( just_singles );
  
  // Compute HF Energy
  std::vector<double> orb_ens( norb );
  for( int i = 0; i < norb; i++ )
    orb_ens[i] = ints.get( i, i ); 
  const std::bitset<nbits> hf_det = 
    dbwy::canonical_hf_determinant<nbits>( nalpha,nbeta, orb_ens );
  //const std::bitset<nbits> hf_det = 
  //  dbwy::canonical_hf_determinant<nbits>(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if( world_rank == 0 && !quiet ) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }
  
  
  auto print_ed  = [&](double E) {
    if( world_rank == 0 && !quiet ) {
      std::cout << "  * E(ED)   = " << E + ints.core_energy << " Eh" << std::endl;
      std::cout << "  * E_c(ED) = " << (E - EHF)*1000 << " mEh" << std::endl;
    }
  };
  
  if(read_wfn) {
    // Read WFN
  
    {
      std::ifstream wfn(wfn_file);
      std::string line;
      std::getline( wfn, line );
      while( std::getline(wfn, line) ) {
        std::stringstream ss{line};
      std::string coeff, det;
      ss >> coeff >> det;
      dets.emplace_back( dbwy::from_canonical_string<nbits>(det));
      }
    }
    if(world_rank == 0 && !quiet)
      std::cout << "NDETS = " << dets.size() << std::endl;
    for( auto det : dets ) std::cout << det << std::endl;
    auto E = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
      1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD, quiet );
    print_ed( E );
    if(!quiet)
      std::cout << "**** MY PRINT ***" << std::endl;
    ham_gen.matrix_element(dets[0], dets[1]);
  } else {
  
  //  Run ED
  {
  if(world_size != 1) throw "NO MPI"; // Disable MPI for now
  
  int nstates          = 1;
  try { nstates = cmz::ed::getParam<int>( input, "nstates" ); } catch(...){ }
  // Fill the determinant list with all possible dets
  dets = dbwy::generate_full_hilbert_space<nbits>( norb, nalpha, nbeta );
  if(world_rank == 0 && !quiet)
    std::cout << "* NDETS in ED = " << dets.size() << std::endl;
  auto EED = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD, quiet, nstates );
  Eret = EED + ints.core_energy;
  std::cout << "EED = " << EED << std::endl;
  print_ed( EED );
  
  bool print_wf;
  try{ print_wf = cmz::ed::getParam<bool>( input, "print_wf" ); }catch (...) { print_wf = false; }
  if( print_wf )
  {
    std::ofstream ofile( "final_ed_wf.dat", std::ios::out );

    struct wf_pair
    {
      std::string s;
      double coeff;
    };
    std::vector<wf_pair> pairs;
    pairs.reserve( dets.size() );
    for( int idet = 0; idet < dets.size(); idet++ )
    {
      wf_pair p = { to_canonical_string(dets[idet]), X_local[idet] };
      pairs.push_back( p );
    }
    std::sort( pairs.begin(), pairs.end(), []( wf_pair &a, wf_pair &b )->bool{ return abs(a.coeff) > abs(b.coeff); } );

    ofile << dets.size() << std::endl;
    auto w = std::setw(25);

    for( int idet = 0; idet < dets.size(); idet++ )
       ofile << pairs[idet].coeff << w << pairs[idet].s << std::endl; 

    ofile.close();
  }
  
  } // ED

  }

  // Green's function calculation
  { 
    double EED = Eret - ints.core_energy;
    Eigen::VectorXd psi0  = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( X_local.data(), X_local.size() );
    std::vector<double> ordm( norb*norb, 0. );
    ham_gen.form_rdms( dets.begin(), dets.end(), 
                       dets.begin(), dets.end(),
                       X_local.data(), ordm.data() );

    std::vector<double> occs(norb, 0.);
    for( int i = 0; i < norb; i++ )
      occs[i] += ordm[i + i * norb] / 2.;

    std::cout << "Occs: ";
    for( const auto oc : occs )
      std::cout << oc << ", ";
    std::cout << endl;

    auto ws = cmz::ed::GetFreqGrid( input );

    // Particle sector
    bool is_part = true;
    cmz::ed::RunGFCalc<nbits>( GF, psi0, ham_gen, dets, EED, is_part, ws, occs, input );

    // Hole sector
    is_part = false;
    std::vector<std::vector<std::vector<std::complex<double> > > > GF_tmp;
    cmz::ed::RunGFCalc<nbits>( GF_tmp, psi0, ham_gen, dets, EED, is_part, ws, occs, input );
    for( int i = 0; i < GF_tmp.size(); i++ )
      for( int j = 0; j < GF_tmp[i].size(); j++)
        for( int k = 0; k < GF_tmp[i][j].size(); k++)
          GF[i][j][k] += GF_tmp[i][j][k];
  } // GF
  return std::tuple( Eret, GF );
}

}
