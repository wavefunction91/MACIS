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

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/stopwatch.h>

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
  //spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%n] %v");

  constexpr size_t nwfn_bits = 64;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  {
  // Create Logger
  auto console = spdlog::stdout_color_mt("standalone driver");
  auto bfgs_logger = spdlog::null_logger_mt("bfgs");

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

  #define OPT_KEYWORD(STR, RES, DTYPE) \
  if(input.containsData(STR)) {        \
    RES = input.getData<DTYPE>(STR);   \
  }
  
  // Set up active space
  size_t n_inactive = 0;
  OPT_KEYWORD("CI.NINACTIVE", n_inactive, size_t);

  if( n_inactive >= norb )
    throw std::runtime_error("NINACTIVE >= NORB");

  size_t n_active = norb - n_inactive;
  OPT_KEYWORD("CI.NACTIVE", n_active, size_t);

  if( n_inactive + n_active > norb )
    throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

  size_t n_virtual = norb - n_active - n_inactive;

  // Misc optional files
  std::string rdm_fname, fci_out_fname;
  OPT_KEYWORD("CI.RDMFILE",     rdm_fname,     std::string);
  OPT_KEYWORD("CI.FCIDUMP_OUT", fci_out_fname, std::string);
  

  // MCSCF Settings
  asci::MCSCFSettings mcscf_settings;
  OPT_KEYWORD("MCSCF.MAX_MACRO_ITER", mcscf_settings.max_macro_iter,     size_t);
  OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL" , mcscf_settings.orb_grad_tol_mcscf, double);
  OPT_KEYWORD("MCSCF.BFGS_TOL",       mcscf_settings.orb_grad_tol_bfgs,  double);
  OPT_KEYWORD("MCSCF.BFGS_MAX_ITER",  mcscf_settings.max_bfgs_iter,      size_t);
  OPT_KEYWORD("MCSCF.CI_RES_TOL",     mcscf_settings.ci_res_tol,         double);
  OPT_KEYWORD("MCSCF.CI_MAX_SUB",     mcscf_settings.ci_max_subspace,    size_t);
  OPT_KEYWORD("MCSCF.CI_MATEL_TOL",   mcscf_settings.ci_matel_tol,       double);

  if( !world_rank ) {
    console->info("[Wavefunction Data]:");
    console->info("  * FCIDUMP = {}", fcidump_fname );
    if( rdm_fname.size() ) console->info("  * RDMFILE = {}", rdm_fname );
    if( fci_out_fname.size() ) 
      console->info("  * FCIDUMP_OUT = {}", fci_out_fname);
    //console->info("  * N_ALPHA = {}", nalpha);
    //console->info("  * N_BETA  = {}", nbeta);
    //console->info("[Active Space]:");
    //console->info("  * N_ORB      = {}", norb);
    //console->info("  * N_INACTIVE = {}", n_inactive);
    //console->info("  * N_ACTIVE   = {}", n_active);
    //console->info("  * N_VIRTUAL  = {}", n_virtual);

    console->debug("READ {} 1-body integrals and {} 2-body integrals", 
      T.size(), V.size());
    console->debug("ECORE = {:.12f}", E_core); 
    console->debug("TSUM  = {:.12f}", vec_sum(T));
    console->debug("VSUM  = {:.12f}", vec_sum(V));
  }


  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);

  using asci::NumElectron;
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
    console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(F_inactive));
    console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(V_active)  );
    console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(T_active)  ); 
  }

  // Compute Inactive energy
  auto E_inactive = asci::inactive_energy(NumInactive(n_inactive), 
    T.data(), norb, F_inactive.data(), norb);

  if(world_rank == 0) {
    console->info("E(inactive) = {:.12f}", E_inactive);
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
    console->info("E(HF)   = {:.12f} Eh", EHF + E_inactive + E_core);
    console->info("E(CI)   = {:.12f} Eh", E0  + E_inactive + E_core);
  }

#if 0
  std::vector<double> a1rdm(active_ordm.size()), a2rdm(active_trdm.size());
  compute_casci_rdms<nwfn_bits>(NumOrbital(n_active), nalpha, nbeta, 
    T_active.data(), V_active.data(), a1rdm.data(), a2rdm.data(),
    MPI_COMM_WORLD);

  std::vector<double> a1rdm_diff(a1rdm.size());
  for( auto i = 0, v = 0; v < n_active; ++v) 
  for( auto w = 0;        w < n_active; ++w, ++i) {
    a1rdm_diff[i] = a1rdm[i] - active_ordm[i];
    console->info("A1RDM ({:3},{:3}) {:15.5e} {:15.5e} {:15.5e}",w,v,a1rdm[i],active_ordm[i],a1rdm_diff[i]);
  }

  std::vector<double> a2rdm_diff(a2rdm.size());
  for( auto i = 0, v = 0; v < n_active; ++v) 
  for( auto w = 0;        w < n_active; ++w) 
  for( auto x = 0;        x < n_active; ++x) 
  for( auto y = 0;        y < n_active; ++y, ++i) {
    a2rdm_diff[i] = a2rdm[i] - active_trdm[i];
    console->info("A2RDM ({:3},{:3},{:3},{:3}) {:15.5e} {:15.5e} {:15.5e}",y,x,w,v,a2rdm[i],active_trdm[i],a2rdm_diff[i]);
  }
#endif


  

#if 1
  if(world_rank == 0) {
    console->debug("ORDMSUM = {:.12f}", vec_sum(active_ordm));
    console->debug("TRDMSUM = {:.12f}", vec_sum(active_trdm));
  }

  // Compute CI energy from RDMs
  double ERDM = blas::dot( active_ordm.size(), active_ordm.data(), 1, T_active.data(), 1 );
  ERDM += blas::dot( active_trdm.size(), active_trdm.data(), 1, V_active.data(), 1 );
  console->info("E(RDM)  = {:.12f} Eh", ERDM + E_inactive + E_core);

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
  console->info("E(FOCK) = {:.12f} Eh", E_FOCK + E_core);

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

#if 0
  // Optimize Orbitals
  std::vector<double> K(norb2);
  asci::optimize_orbitals(NumOrbital(norb), NumInactive(n_inactive),
    NumActive(n_active), NumVirtual(n_virtual), E_core, T.data(), norb,
    V.data(), norb, active_ordm.data(), n_active, active_trdm.data(),
    n_active, K.data(), norb);
#else
  // CASSCF
  asci::casscf_bfgs( mcscf_settings, NumElectron(nalpha), NumElectron(nbeta),
    NumOrbital(norb), NumInactive(n_inactive), NumActive(n_active),
    NumVirtual(n_virtual), E_core, T.data(), norb, V.data(), norb, 
    active_ordm.data(), n_active, active_trdm.data(), n_active,
    MPI_COMM_WORLD);

  if(fci_out_fname.size())
  asci::write_fcidump(fci_out_fname,norb, T.data(), norb, V.data(), norb, E_core);
#endif

#endif
#endif
  
  } // MPI Scope

  MPI_Finalize();

}
