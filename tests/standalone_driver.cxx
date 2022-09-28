#include <asci/fcidump.hpp>
#include <asci/util/cas.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/util/fock_matrices.hpp>

#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "ini_input.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/stopwatch.h>
#include <spdlog/cfg/env.h>

using asci::NumElectron;
using asci::NumOrbital;
using asci::NumInactive;
using asci::NumActive;
using asci::NumVirtual;


template <typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

#if 0
template <size_t nbits>
auto compute_casci_rdms(asci::NumOrbital norb, size_t nalpha, size_t nbeta,
  double* T, double* V, double* ORDM, double* TRDM, MPI_Comm comm) {

  int rank; MPI_Comm_rank(comm, &rank);

  // Hamiltonian Matrix Element Generator
#if 0
  asci::DoubleLoopHamiltonianGenerator<nbits> ham_gen( norb.get(), V, T );
#else
  size_t no = norb.get();
  asci::DoubleLoopHamiltonianGenerator<nbits> ham_gen( 
    asci::matrix_span<double>(T,no,no),
    asci::rank4_span<double>(V,no,no,no,no) 
  );
#endif

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
    C.data(), asci::matrix_span<double>(ORDM,no,no),
    asci::rank4_span<double>(TRDM,no,no,no,no));

  return std::make_pair(EHF, E0);
}
#endif




int main(int argc, char** argv) {

  std::cout << std::scientific << std::setprecision(12);
  //spdlog::set_level(spdlog::level::debug);
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("[%n] %v");

  constexpr size_t nwfn_bits = 64;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  {
  // Create Logger
  auto console = spdlog::stdout_color_mt("standalone driver");
  //auto bfgs_logger = spdlog::null_logger_mt("bfgs");
  //auto davidson_logger = spdlog::null_logger_mt("davidson");
  //auto ci_logger = spdlog::null_logger_mt("ci_solver");
  auto diis_logger = spdlog::null_logger_mt("diis");

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
  OPT_KEYWORD("MCSCF.MAX_ORB_STEP",   mcscf_settings.max_orbital_step,   double);
  OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL" , mcscf_settings.orb_grad_tol_mcscf, double);
  //OPT_KEYWORD("MCSCF.BFGS_TOL",       mcscf_settings.orb_grad_tol_bfgs,  double);
  //OPT_KEYWORD("MCSCF.BFGS_MAX_ITER",  mcscf_settings.max_bfgs_iter,      size_t);
  OPT_KEYWORD("MCSCF.ENABLE_DIIS",    mcscf_settings.enable_diis,        bool  );
  OPT_KEYWORD("MCSCF.DIIS_START_ITER",mcscf_settings.diis_start_iter,    size_t);
  OPT_KEYWORD("MCSCF.DIIS_NKEEP",     mcscf_settings.diis_nkeep,         size_t);
  OPT_KEYWORD("MCSCF.CI_RES_TOL",     mcscf_settings.ci_res_tol,         double);
  OPT_KEYWORD("MCSCF.CI_MAX_SUB",     mcscf_settings.ci_max_subspace,    size_t);
  OPT_KEYWORD("MCSCF.CI_MATEL_TOL",   mcscf_settings.ci_matel_tol,       double);

  if( !world_rank ) {
    console->info("[Wavefunction Data]:");
    console->info("  * FCIDUMP = {}", fcidump_fname );
    if( rdm_fname.size() ) console->info("  * RDMFILE = {}", rdm_fname );
    if( fci_out_fname.size() ) 
      console->info("  * FCIDUMP_OUT = {}", fci_out_fname);

    console->debug("READ {} 1-body integrals and {} 2-body integrals", 
      T.size(), V.size());
    console->debug("ECORE = {:.12f}", E_core); 
    console->debug("TSUM  = {:.12f}", vec_sum(T));
    console->debug("VSUM  = {:.12f}", vec_sum(V));
  }

  // Compute canonical orbital energies
  //std::vector<double> eps(norb);
  //asci::canonical_orbital_energies(NumOrbital(norb), NumInactive(nalpha),
  //  T.data(), norb, V.data(), norb, eps.data());
  
  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);

  // Compute active-space Hamiltonian and inactive Fock matrix
  std::vector<double> F_inactive(norb2);
  asci::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
    NumInactive(n_inactive), T.data(), norb, V.data(), norb,
    F_inactive.data(), norb, T_active.data(), n_active,
    V_active.data(), n_active);

  console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(F_inactive));
  console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(V_active)  );
  console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(T_active)  ); 

  // Compute Inactive energy
  auto E_inactive = asci::inactive_energy(NumInactive(n_inactive), 
    T.data(), norb, F_inactive.data(), norb);
  console->info("E(inactive) = {:.12f}", E_inactive);
  

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
    std::vector<double> C_local;
    E0 = 
      asci::compute_casci_rdms<asci::DoubleLoopHamiltonianGenerator<64>>(
        asci::MCSCFSettings{}, NumOrbital(n_active), nalpha, nbeta, 
        T_active.data(), V_active.data(), active_ordm.data(), active_trdm.data(),
        C_local, MPI_COMM_WORLD);
    //console->info("E(HF)   = {:.12f} Eh", EHF + E_inactive + E_core);
    console->info("E(CI)   = {:.12f} Eh", E0  + E_inactive + E_core);
  }

  console->debug("ORDMSUM = {:.12f}", vec_sum(active_ordm));
  console->debug("TRDMSUM = {:.12f}", vec_sum(active_trdm));

  // Compute CI energy from RDMs
  double ERDM = blas::dot( active_ordm.size(), active_ordm.data(), 1, T_active.data(), 1 );
  ERDM += blas::dot( active_trdm.size(), active_trdm.data(), 1, V_active.data(), 1 );
  console->info("E(RDM)  = {:.12f} Eh", ERDM + E_inactive + E_core);

#if 0
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
#endif

  // CASSCF
  asci::casscf_diis( mcscf_settings, NumElectron(nalpha), NumElectron(nbeta),
    NumOrbital(norb), NumInactive(n_inactive), NumActive(n_active),
    NumVirtual(n_virtual), E_core, T.data(), norb, V.data(), norb, 
    active_ordm.data(), n_active, active_trdm.data(), n_active,
    MPI_COMM_WORLD);

  if(fci_out_fname.size())
    asci::write_fcidump(fci_out_fname,norb, T.data(), norb, V.data(), norb, E_core);

  } // MPI Scope

  MPI_Finalize();

}
