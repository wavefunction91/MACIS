#include <asci/fcidump.hpp>
#include <asci/davidson.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/util/selected_ci_diag.hpp>
#include <asci/util/fock_matrices.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/mcscf.hpp>
#include <asci/util/orbital_gradient.hpp>

#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "ini_input.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

template <typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

template <size_t nbits>
auto compute_casci_rdms(asci::NumOrbital norb, size_t nalpha, size_t nbeta,
  double* T, double* V, double* ORDM, double* TRDM, MPI_Comm comm) {

  int rank; MPI_Comm_rank(comm, &rank);

  // Hamiltonian Matrix Element Generator
  asci::DoubleLoopHamiltonianGenerator<nbits> ham_gen( norb.get(), V, T );

  // Compute HF Energy
  const auto hf  = asci::canonical_hf_determinant<nbits>(nalpha, nbeta);
  double     EHF = ham_gen.matrix_element(hf, hf);
  
  // Compute Lowest Energy Eigenvalue (ED)
  std::vector<double> C;
  auto dets = asci::generate_hilbert_space<nbits>(norb.get(), nalpha, nbeta);
  double E0 = asci::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    1e-16, 20, 1e-8, C, comm);

  // Compute RDMs
  ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
    C.data(), ORDM, TRDM);

  return std::make_pair(EHF, E0);
}




int main(int argc, char** argv) {

  std::cout << std::scientific << std::setprecision(12);

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

  std::string rdm_fname;
  if(input.containsData("CI.RDMFILE")) {
    rdm_fname = input.getData<std::string>("CI.RDMFILE");
  }

  if( !world_rank ) {
    std::cout << "Wavefunction Data:" << std::endl
              << "  * FCIDUMP = " << fcidump_fname << std::endl;
    if(rdm_fname.size())
    std::cout << "  * RDMFILE = " << rdm_fname     << std::endl;
    std::cout << "  * N_ALPHA = " << nalpha        << std::endl
              << "  * N_BETA  = " << nbeta         << std::endl
              << std::endl
              << "Active Space: " << std::endl
              << "  * N_ORB      = " << norb       << std::endl
              << "  * N_INACTIVE = " << n_inactive << std::endl
              << "  * N_ACTIVE   = " << n_active   << std::endl
              << "  * N_VIRTUAL  = " << n_virtual  << std::endl
              << std::endl << std::endl;

    std::cout << "READ " << T.size() << " 1-body integrals and " << V.size() << " 2-body integrals " << std::endl;
    std::cout << "ECORE = " << E_core << std::endl;
    std::cout << "TSUM = " << vec_sum(T) << std::endl;
    std::cout << "VSUM = " << vec_sum(V) << std::endl;
     
  }


  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);

  using asci::NumOrbital;
  using asci::NumInactive;
  using asci::NumActive;
  using asci::NumVirtual;

  std::vector<double> F_inactive(norb2);
#if 0
  // Extract active two body interaction from V
  asci::active_subtensor_2body(NumActive(n_active), 
    NumInactive(n_inactive), V.data(), norb, 
    V_active.data(), n_active);

  // Compute inactive Fock in full MO space
  asci::inactive_fock_matrix( NumOrbital(norb), 
    NumInactive(n_inactive), T.data(), norb, 
    V.data(), norb, F_inactive.data(), norb );

  // Replace T_active with active-active block of F_inactive
  asci::active_submatrix_1body(NumActive(n_active), 
    NumInactive(n_inactive), F_inactive.data(), norb, 
    T_active.data(), n_active);
#else
  // Compute active-space Hamiltonian and inactive Fock matrix
  asci::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
    NumInactive(n_inactive), T.data(), norb, V.data(), norb,
    F_inactive.data(), norb, T_active.data(), n_active,
    V_active.data(), n_active);
#endif


  if(world_rank == 0) {
    std::cout << "FINACTIVE_SUM = " << vec_sum(F_inactive) << std::endl;
    std::cout << "VACTIVE_SUM   = " << vec_sum(V_active)   << std::endl;
    std::cout << "TACTIVE_SUM   = " << vec_sum(T_active)   << std::endl; 
  }

  // Compute Inactive energy
  auto E_inactive = asci::inactive_energy(NumInactive(n_inactive), 
    T.data(), norb, F_inactive.data(), norb);

  if(world_rank == 0) {
    std::cout << std::endl;
    std::cout << "E(inactive) = " << E_inactive << std::endl;
  }
  

  // Storage for active RDMs
  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm( active_ordm.size() * active_ordm.size() );
  
  // Compute or Read active RDMs
  double EHF, E0;
  if(rdm_fname.size()) {
    std::vector<double> full_ordm(norb2), full_trdm(norb4);
    asci::read_rdms_binary(rdm_fname, norb, full_ordm.data(), norb,
      full_trdm.data(), norb );
    asci::active_submatrix_1body(NumActive(n_active), NumInactive(n_inactive),
      full_ordm.data(), norb, active_ordm.data(), n_active);
    asci::active_subtensor_2body(NumActive(n_active), NumInactive(n_inactive),
      full_trdm.data(), norb, active_trdm.data(), n_active);
  } else {
    std::tie(EHF, E0) = 
      compute_casci_rdms<nwfn_bits>(NumOrbital(n_active), nalpha, nbeta, 
        T_active.data(), V_active.data(), active_ordm.data(), active_trdm.data(),
        MPI_COMM_WORLD);
  }
  if(world_rank == 0) {
    std::cout << "ORDMSUM = " << vec_sum(active_ordm) << std::endl;
    std::cout << "TRDMSUM = " << vec_sum(active_trdm) << std::endl;
    std::cout << std::endl;
    std::cout << "E(HF)   = " << EHF + E_inactive + E_core << " Eh" << std::endl;
    std::cout << "E(CI)   = " << E0  + E_inactive + E_core << " Eh" << std::endl;
  }

  // Compute CI energy from RDMs
  double ERDM = blas::dot( active_ordm.size(), active_ordm.data(), 1, T_active.data(), 1 );
  ERDM += blas::dot( active_trdm.size(), active_trdm.data(), 1, V_active.data(), 1 );
  std::cout << "E(RDM)  = " << ERDM + E_inactive + E_core << std::endl;

  // Compute Generalized Fock matrix
  std::vector<double> F(norb2);
#if 0
  // Compute given inactive Fock
  asci::generalized_fock_matrix_comp_mat1(norb, n_inactive, 
    n_active, F_inactive.data(), norb, V.data(), norb,
    active_ordm.data(), n_active, active_trdm.data(), 
    n_active, F.data(), norb);
#else
  // Compute directly from full MO tensors and active RDMs
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

#if 0

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


  // check orbital rotated energy
  size_t i = 0, a = 3;
  double dk = 0.001;
  double max_k = 0.5;
  size_t nk = max_k/dk;
  std::vector<double> K(norb2), U(norb2);

  // Compute K
  std::fill(K.begin(), K.end(), 0);
  //K[a + i*norb] =  1.0;
  //K[i + a*norb] = -1.0;
  asci::fock_to_gradient(NumOrbital(norb),NumInactive(n_inactive),
    NumActive(n_active), NumVirtual(n_virtual), F.data(), norb,
    K.data(), norb);
  for( auto& x : K ) x = -x;
  

  for(size_t ik = 0; ik < nk; ++ik) {

    // Compute U = EXP[-alpha * K]
    asci::compute_orbital_rotation(NumOrbital(norb), ik*dk, K.data(), norb,
      U.data(), norb);

    // Compute Rotated Energy
    auto E = asci::orbital_rotated_energy(NumOrbital(norb), NumInactive(n_inactive),
      NumActive(n_active), T.data(), norb, V.data(), norb, active_ordm.data(),
      n_active, active_trdm.data(), n_active, U.data(), norb);

    std::cout << ik*dk << ", " << E << ", " << E - E_FOCK << std::endl;
    
  }

#else

  // Optimize Orbitals
  std::vector<double> K(norb2);
  asci::optimize_orbitals(NumOrbital(norb), NumInactive(n_inactive),
    NumActive(n_active), NumVirtual(n_virtual), E_core, T.data(), norb,
    V.data(), norb, active_ordm.data(), n_active, active_trdm.data(),
    n_active, K.data(), norb);

#endif
  
  } // MPI Scope

  MPI_Finalize();

}
