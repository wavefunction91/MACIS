#include <mpi.h>
#include <iostream>
#include "cmz_ed/utils.h++"
#include "dbwy/csr_hamiltonian.hpp"
#include "dbwy/davidson.hpp"
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"

#include <bitset>

#include "dbwy/asci_util.hpp"

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

int main( int argc, char* argv[] ) {

  MPI_Init(NULL,NULL);
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  { // MPI Scope

  if( argc != 2 ) {
    std::cout << "Must Specify Input" << std::endl;
    return 1;
  }

  // Read Input
  std::string in_file = argv[1];
  cmz::ed::Input_t input;
  cmz::ed::ReadInput( in_file, input );

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );
  bool read_wfn =
    cmz::ed::getParam<bool>( input, "read_wfn");
  std::string wfn_file;
  if( read_wfn )
    wfn_file = 
      cmz::ed::getParam<std::string>( input, "wfn_file" );

  constexpr size_t nbits = 128;
  if( norb > nbits/2 ) throw std::runtime_error("Not Enough Bits...");


  // Read in the integrals 
  cmz::ed::intgrls::integrals ints(norb, fcidump);
  MPI_Barrier(MPI_COMM_WORLD);

  // Hamiltonian Matrix Element Generator
  dbwy::DoubleLoopHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );

  // Compute HF Energy
  const std::bitset<nbits> hf_det = 
    dbwy::canonical_hf_determinant<nbits>(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if(world_rank == 0) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }


  auto print_asci  = [&](double E) {
    if(world_rank == 0) {
      std::cout << "  * E(ASCI)   = " << E + ints.core_energy << " Eh" << std::endl;
      std::cout << "  * E_c(ASCI) = " << (E - EHF)*1000 << " mEh" << std::endl;
    }
  };

  if(read_wfn) {
    // Read WFN

    std::vector<std::bitset<nbits>> dets;
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
    std::cout << "NDETS = " << dets.size() << std::endl;
    for( auto det : dets ) std::cout << det << std::endl;
    std::vector<double> X_local; // Eigenvectors
    auto E = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
      1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD );
    print_asci( E );
    std::cout << "**** MY PRINT ***" << std::endl;
    ham_gen.matrix_element(dets[0], dets[1]);
  } else {

  //  Run ASCI
  size_t ndets_max     = 10000;
  size_t niter_max     = 6;
  double coeff_thresh  = 1e-4;
  {
  if(world_size != 1) throw "NO MPI"; // Disable MPI for now

  auto bitset_comp = [](auto x, auto y){ return dbwy::bitset_less(x,y); };
  std::vector<double> X_local; // Eigenvectors

#if 0
  // Start with CISD
  std::cout << "* Initializing ASCI with CISD" << std::endl;
  auto dets = dbwy::generate_cisd_hilbert_space<nbits>( norb, hf_det );
  std::sort(dets.begin(), dets.end(), bitset_comp);

  // Diagonalize CISD Hamiltonian
  double EASCI = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD );
#else
  // Staring with HF
  std::cout << "* Initializing ASCI with HF" << std::endl;
  std::vector<std::bitset<nbits>> dets = {hf_det};
  X_local = {1.0};
  auto EASCI = EHF;
#endif
  print_asci( EASCI );
       
  // ASCI Loop
  for( size_t iter = 0; iter < niter_max; ++iter ) {

    std::cout << "\n* ASCI Iteration: " << iter << std::endl;

    // Reorder the dets / coefficients
    if( dets.size() > 1 )
      dbwy::reorder_ci_on_coeff( dets, X_local, MPI_COMM_WORLD );

    size_t nkeep = std::min(120ul,dets.size());

    // Do ASCI Search
    dets = asci_search( ndets_max, dets.begin(), dets.begin() + nkeep,
      EASCI, X_local, norb, ham_gen.T_pq_, ham_gen.G_red_.data(), 
      ham_gen.V_red_.data(), ham_gen.G_pqrs_.data(), ham_gen.V_pqrs_, ham_gen );

    // Rediagonalize
    auto E = dbwy::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
      1e-12, 100, 1e-8, X_local, MPI_COMM_WORLD );

    // Print iteration results
    print_asci( E );

    // Check for convergence
    if( std::abs(E - EASCI) < 1e-6 ) {
      std::cout << "ASCI Converged" << std::endl;
      break;
    }
    std::cout << "  * dE        = " << (E - EASCI)*1000 << " mEh" << std::endl;
    EASCI = E;

  }

#if 0
  dbwy::reorder_ci_on_coeff( dets, X_local, MPI_COMM_WORLD );
  for( auto i = 0; i < 50; ++i ) {
    std::cout << X_local[i] << ", ";
    for( auto j = 0; j < nbits/2; ++j ) {
      if( dets[i].test(j) and dets[i].test(j+nbits/2) ) std::cout << "2";
      else if( dets[i].test(j) ) std::cout << "u";
      else if( dets[i].test(j+nbits/2) ) std::cout << "d";
      else std::cout << "0";
    }
    std::cout << std::endl;
  }
#endif

  } // ASCI 

  }

  } // MPI Scope
  MPI_Finalize();

}
