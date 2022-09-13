#include <asci/fcidump.hpp>
#include <asci/davidson.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/util/selected_ci_diag.hpp>
#include <asci/util/fock_matrices.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/orbital_gradient.hpp>

#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "ini_input.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

int main(int argc, char** argv) {

  constexpr size_t nwfn_bits = 64;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  {

  // Read Input Options
  std::vector< std::string > opts( argc );
  for( int i = 0; i < argc; ++i ) opts[i] = argv[i];

  auto input_file = opts.at(1);
  INIFile input(input_file);

  // Required Keywords
  auto fcidump_fname = input.getData<std::string>("CI.FCIDUMP");
  auto nalpha        = input.getData<size_t>("CI.NALPHA");
  auto nbeta         = input.getData<size_t>("CI.NBETA");

  if( nalpha != nbeta ) throw std::runtime_error("NALPHA != NBETA");

  // Read FCIDUMP File
  size_t norb  = asci::read_fcidump_norb(fcidump_fname);
  size_t norb2 = norb  * norb;
  size_t norb3 = norb2 * norb;
  size_t norb4 = norb2 * norb2;

  if( norb > nwfn_bits/2 ) throw std::runtime_error("Not Enough Bits");

  std::vector<double> T(norb2), V(norb4);
  auto E_core = asci::read_fcidump_core(fcidump_fname);
  asci::read_fcidump_1body(fcidump_fname, T.data(), norb);
  asci::read_fcidump_2body(fcidump_fname, V.data(), norb);
  
  // Set up active space
  size_t n_inactive = 0;
  if(input.containsData("CI.NINACTIVE")) {
    n_inactive = input.getData<size_t>("CI.NINACTIVE");
  }

  if( n_inactive >= norb )
    throw std::runtime_error("NINACTIVE >= NORB");

  size_t n_active = norb - n_inactive;
  if(input.containsData("CI.NACTIVE")) {
    n_active = input.getData<size_t>("CI.NACTIVE");
  }

  if( n_inactive + n_active > norb )
    throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

  size_t n_virtual = norb - n_active - n_inactive;

  if( !world_rank ) {
    std::cout << "Wavefunction Data:" << std::endl
              << "  * FCIDUMP = " << fcidump_fname << std::endl
              << "  * N_ALPHA = " << nalpha        << std::endl
              << "  * N_BETA  = " << nbeta         << std::endl
              << std::endl
              << "Active Space: " << std::endl
              << "  * N_ORB      = " << norb       << std::endl
              << "  * N_INACTIVE = " << n_inactive << std::endl
              << "  * N_ACTIVE   = " << n_active   << std::endl
              << "  * N_VIRTUAL  = " << n_virtual  << std::endl
              << std::endl;
  }


  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);

  using asci::NumOrbital;
  using asci::NumInactive;
  using asci::NumActive;
  using asci::NumVirtual;

  // Extract active two body interaction
  asci::active_subtensor_2body(NumActive(n_active), 
    NumInactive(n_inactive), V.data(), norb, 
    V_active.data(), n_active);

  // Compute inactive Fock
  std::vector<double> F_inactive(norb2);
  asci::inactive_fock_matrix( NumOrbital(norb), 
    NumInactive(n_inactive), T.data(), norb, 
    V.data(), norb, F_inactive.data(), norb );




  // Replace T_active with F_inactive
  asci::active_submatrix_1body(NumActive(n_active), 
    NumInactive(n_inactive), F_inactive.data(), norb, 
    T_active.data(), n_active);

  // Compute Inactive energy
  auto E_inactive = asci::inactive_energy(NumInactive(n_inactive), 
    T.data(), norb, F_inactive.data(), norb);
  std::cout << std::scientific << std::setprecision(12);
  std::cout << "E(inactive) = " << E_inactive << std::endl;
  

  
  // Hamiltonian Matrix Element Generator
  asci::DoubleLoopHamiltonianGenerator<nwfn_bits>
    ham_gen( n_active, V_active.data(), T_active.data() );

  // Compute HF Energy
  const auto hf_det = asci::canonical_hf_determinant<nwfn_bits>(nalpha, nbeta);
  const auto EHF    = ham_gen.matrix_element(hf_det, hf_det);  
  if(world_rank == 0) {
    std::cout << "E(HF) = " << EHF + E_inactive + E_core << " Eh" << std::endl;
    //std::cout << "E(HF) = " << EHF  << " Eh" << std::endl;
  }


  // Compute Lowest Energy Eigenvalue (ED) 
  auto dets = asci::generate_hilbert_space<nwfn_bits>(n_active, nalpha, nbeta);
  std::vector<double> X_local;
  auto E0 = asci::selected_ci_diag(dets.begin(), dets.end(), ham_gen, 1e-16,
    15, 1e-8, X_local, MPI_COMM_WORLD );
  
  if(world_rank == 0) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(CI)   = " << E0 + E_inactive + E_core << " Eh" << std::endl;
  }

  // Compute RDMs
  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm( active_ordm.size() * active_ordm.size() );
  ham_gen.form_rdms( dets.begin(), dets.end(), dets.begin(), dets.end(),
    X_local.data(), active_ordm.data(), active_trdm.data() );

  // Compute CI energy from RDMs
  double ERDM = blas::dot( active_ordm.size(), active_ordm.data(), 1, T_active.data(), 1 );
  ERDM += blas::dot( active_trdm.size(), active_trdm.data(), 1, V_active.data(), 1 );
  std::cout << "E(RDM)  = " << ERDM + E_inactive + E_core << std::endl;


  std::vector<double> F(norb2);
#if 0
  asci::generalized_fock_matrix_comp_mat1(norb, n_inactive, 
    n_active, F_inactive.data(), norb, V.data(), norb,
    active_ordm.data(), n_active, active_trdm.data(), 
    n_active, F.data(), norb);
#else
  asci::generalized_fock_matrix_comp_mat2(NumOrbital(norb), 
    NumInactive(n_inactive), NumActive(n_active), T.data(), 
    norb, V.data(), norb, active_ordm.data(), n_active, 
    active_trdm.data(), n_active, F.data(), norb);
#endif

  // Compute Energy from Generalied Fock Matrix
  auto E_FOCK = asci::energy_from_generalized_fock(
    NumInactive(n_inactive), NumActive(n_active), T.data(), 
    norb, active_ordm.data(), n_active, F.data(), norb);
  std::cout << "E(FOCK) = " << E_FOCK + E_core << std::endl;


  // Numerical Orbital gradient
  std::vector<double> OGrad(norb*norb);
  asci::numerical_orbital_gradient(NumOrbital(norb), 
    NumInactive(n_inactive), NumActive(n_active),
    T.data(), norb, V.data(), norb, active_ordm.data(), 
    n_active, active_trdm.data(), n_active, OGrad.data(),
    norb);

  std::cout << std::endl;
  std::cout << "Active - Inactive Orbital Gradient" << std::endl;
  for(size_t i = 0; i < n_inactive; ++i) 
  for(size_t a = 0; a < n_active;   ++a){
    auto ia = i + (a+n_inactive)*norb;
    auto ai = (a+n_inactive) + i*norb;
    auto exact = 2*(F[ai] - F[ia]);
    auto numer = OGrad[ai];
    std::cout << "  " << i << " " << a << ": " << exact << ", " << std::abs(exact - numer) << std::endl; 
  }

  std::cout << "Virtual - Inactive Orbital Gradient" << std::endl;
  for(size_t i = 0; i < n_inactive; ++i) 
  for(size_t a = 0; a < n_virtual;  ++a) {
    auto ia = i + (a+n_inactive+n_active)*norb;
    auto ai = (a+n_inactive+n_active) + i*norb;
    auto exact = 2*(F[ai] - F[ia]);
    auto numer = OGrad[ai];
    std::cout << "  " << i << " " << a << ": " << exact << ", " << std::abs(exact - numer) << std::endl; 
  }


  std::cout << "Virtual - Active Orbital Gradient" << std::endl;
  for(size_t i = 0; i < n_active;  ++i) 
  for(size_t a = 0; a < n_virtual; ++a) {
    auto ia = (i+n_inactive) + (a+n_inactive+n_active)*norb;
    auto ai = (a+n_inactive+n_active) + (i+n_inactive)*norb;
    auto exact = 2*(F[ai] - F[ia]);
    auto numer = OGrad[ai];
    std::cout << "  " << i << " " << a << ": " << exact << ", " << std::abs(exact - numer) << std::endl; 
  }
  
  } // MPI Scope

  MPI_Finalize();

}
