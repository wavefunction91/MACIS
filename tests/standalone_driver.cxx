#include <asci/fcidump.hpp>
#include <asci/util/cas.hpp>
//#include <asci/util/asci_search.hpp>
//#include <asci/util/asci_iter.hpp>
#include <asci/util/asci_grow.hpp>
#include <asci/util/asci_refine.hpp>
#include <asci/util/orbital_energies.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/util/fock_matrices.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/moller_plesset.hpp>

#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <map>

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
using asci::NumCanonicalOccupied;
using asci::NumCanonicalVirtual;

enum class Job {
  CI,
  MCSCF
};

enum class CIExpansion {
  CAS,
  ASCI
};

std::map<std::string, Job> job_map = {
  {"CI",    Job::CI},
  {"MCSCF", Job::MCSCF}
};

std::map<std::string, CIExpansion> ci_exp_map = {
  {"CAS", CIExpansion::CAS},
  {"ASCI", CIExpansion::ASCI}
};




template <typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}


int main(int argc, char** argv) {

  std::cout << std::scientific << std::setprecision(12);
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("[%n] %v");

  constexpr size_t nwfn_bits = 64;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  {
  // Create Logger
  auto console = spdlog::stdout_color_mt("standalone_driver");

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

  // Set up job
  std::string job_str = "MCSCF";
  OPT_KEYWORD("CI.JOB", job_str, std::string);
  Job job;
  try { job = job_map.at(job_str); }
  catch(...){ throw std::runtime_error("Job Not Recognized"); }

  std::string ciexp_str = "CAS";
  OPT_KEYWORD("CI.EXPANSION", ciexp_str, std::string);
  CIExpansion ci_exp;
  try { ci_exp = ci_exp_map.at(ciexp_str); }
  catch(...){ throw std::runtime_error("CI Expansion Not Recognized"); }

  
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
    console->info("  * JOB     = {}", job_str);
    console->info("  * CIEXP   = {}", ciexp_str);
    console->info("  * FCIDUMP = {}", fcidump_fname );
    if( fci_out_fname.size() ) 
      console->info("  * FCIDUMP_OUT = {}", fci_out_fname);

    console->debug("READ {} 1-body integrals and {} 2-body integrals", 
      T.size(), V.size());
    console->debug("ECORE = {:.12f}", E_core); 
    console->debug("TSUM  = {:.12f}", vec_sum(T));
    console->debug("VSUM  = {:.12f}", vec_sum(V));
  }

  // Setup printing
  bool print_davidson = false, print_ci = false, print_mcscf = true,
       print_diis = false;
  OPT_KEYWORD("PRINT.DAVIDSON", print_davidson, bool );
  OPT_KEYWORD("PRINT.CI",       print_ci,       bool );
  OPT_KEYWORD("PRINT.MCSCF",    print_mcscf,    bool );
  OPT_KEYWORD("PRINT.DIIS",     print_diis,     bool );

  if(not print_davidson) spdlog::null_logger_mt("davidson");
  if(not print_ci)       spdlog::null_logger_mt("ci_solver");
  if(not print_mcscf)    spdlog::null_logger_mt("mcscf");
  if(not print_diis)     spdlog::null_logger_mt("diis");

  // MP2 Guess Orbitals
  {
  size_t nocc_canon = n_inactive + nalpha;
  size_t nvir_canon = norb - nocc_canon;
  size_t noc2 = nocc_canon * nocc_canon;
  size_t nvc2 = nvir_canon * nvir_canon;
  std::vector<double> MP2_RDM(norb*norb,0.0);


#if 0
  // Compute canonical eigenenergies
  // XXX: This will not generally replicate full precision
  // with respect to those returned by the eigen solver
  std::vector<double> eps(norb);
  asci::canonical_orbital_energies(NumOrbital(norb), 
    NumInactive(nocc_canon), T.data(), norb, V.data(), norb, 
    eps.data() );

  std::vector<double> T2(noc2 * nvc2);

  for( auto i = 0ul; i < nocc_canon; ++i )
  for( auto j = 0ul; j < nocc_canon; ++j )
  for( auto a = 0ul; a < nvir_canon; ++a )
  for( auto b = 0ul; b < nvir_canon; ++b ) {
    const auto a_off = a + nocc_canon;
    const auto b_off = b + nocc_canon;

    // T2(i,j,a,b) = (ia|jb) / (eps[i] + eps[j] - eps[a] - eps[b])
    T2[i + j*nocc_canon + a*noc2 + b*noc2*nvir_canon] =
      V[i + a_off*norb + j*norb2 + b_off*norb3] /
      (
         eps[i] + eps[j] - eps[a_off] - eps[b_off]
      );
  }

  double EMP2 = 0.0;
  for( auto i = 0ul; i < nocc_canon; ++i )
  for( auto j = 0ul; j < nocc_canon; ++j )
  for( auto a = 0ul; a < nvir_canon; ++a )
  for( auto b = 0ul; b < nvir_canon; ++b ) {
    const auto a_off = a + nocc_canon;
    const auto b_off = b + nocc_canon;
    const double V_abij = V[a_off + i*norb + b_off*norb2 + j*norb3];
    const double V_abji = V[a_off + j*norb + b_off*norb2 + i*norb3];

    const double t2_ijab = T2[i + j*nocc_canon + a*noc2 + b*noc2*nvir_canon];
    EMP2 += t2_ijab * ( 2*V_abij - V_abji );
  }

  std::cout << "EMP2 = " << EMP2 << std::endl;
  
  // OO-block
  for( auto i = 0ul; i < nocc_canon; ++i )
  for( auto j = 0ul; j < nocc_canon; ++j ) {
    double tmp = 0.0;
    for(auto k = 0ul; k < nocc_canon; ++k )
    for(auto a = 0ul; a < nvir_canon; ++a )
    for(auto b = 0ul; b < nvir_canon; ++b ) {
      // D(i,j) -= T2(i,k,a,b) * (2*T2(j,k,a,b) - T2(j,k,b,a))
      tmp += T2[i + k*nocc_canon + a*noc2 + b*noc2*nvir_canon] *
        (
          2 * T2[j + k*nocc_canon + a*noc2 + b*noc2*nvir_canon] - 
              T2[j + k*nocc_canon + b*noc2 + a*noc2*nvir_canon] 
        );
    }
    MP2_RDM[i + j*norb] = -2*tmp;
    if(i == j) MP2_RDM[i + j*norb] += 2.0;
  }

  // VV-block
  for(auto a = 0ul; a < nvir_canon; ++a )
  for(auto b = 0ul; b < nvir_canon; ++b ) {
    double tmp = 0;
    for(auto i = 0ul; i < nocc_canon; ++i )
    for(auto j = 0ul; j < nocc_canon; ++j ) 
    for(auto c = 0ul; c < nvir_canon; ++c ) {
      // D(a,b) -= T2(i,j,c,a) * (2*T2(i,j,c,b) - T2(i,j,b,c))
      tmp += T2[i + j*nocc_canon + c*noc2 + a*noc2*nvir_canon] *
        (
          2 * T2[i + j*nocc_canon + c*noc2 + b*noc2*nvir_canon] - 
              T2[i + j*nocc_canon + b*noc2 + c*noc2*nvir_canon] 
        );
    }
    MP2_RDM[a+nocc_canon + (b+nocc_canon)*norb] = 2*tmp;
  }
#else
  asci::mp2_1rdm( NumOrbital(norb), NumCanonicalOccupied(nocc_canon),
    NumCanonicalVirtual(nvir_canon), T.data(), norb, V.data(), norb,
    MP2_RDM.data(), norb);
#endif


  for( auto& x : MP2_RDM ) x = -x;
  std::vector<double> W_occ(norb);
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb, 
    MP2_RDM.data(), norb, W_occ.data());
  for(auto& x : W_occ) x = -x;

  // Transform Hamiltonian
  asci::two_index_transform(norb,norb,T.data(),norb,MP2_RDM.data(),
    norb,T.data(),norb);
  asci::four_index_transform(norb,norb,981230128,V.data(),norb,MP2_RDM.data(),
    norb,V.data(),norb);

  }


  
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
  
  double E0 = 0;

  // CI
  if( job == Job::CI ) {

    using generator_t = asci::DoubleLoopHamiltonianGenerator<64>;
    std::vector<double> C_local;
    if( ci_exp == CIExpansion::CAS ) {
      E0 = 
      asci::CASRDMFunctor<generator_t>::rdms(mcscf_settings, NumOrbital(n_active), 
        nalpha, nbeta, T_active.data(), V_active.data(), active_ordm.data(), 
        active_trdm.data(), C_local, MPI_COMM_WORLD);
      E0 += E_inactive + E_core;
    } else {

      spdlog::null_logger_mt("asci_search");
      asci::ASCISettings asci_settings;
      asci_settings.ntdets_max = 100000;
      asci_settings.ncdets_max = 10000;
      asci_settings.grow_factor = 2;
      std::vector<asci::wfn_t<64>> dets = {
        asci::canonical_hf_determinant<64>(nalpha, nalpha)
      };

      generator_t ham_gen( 
        asci::matrix_span<double>(T_active.data(),n_active,n_active),
        asci::rank4_span<double>(V_active.data(),n_active,n_active,n_active,
          n_active)
      );

      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C_local = {1.0};
      #if 0
      dets = asci::asci_search(asci_settings, 100, dets.begin(), dets.end(), EHF,
        C_local, n_active, T_active.data(), ham_gen.G_red(), ham_gen.V_red(),
        ham_gen.G(), V_active.data(), ham_gen, MPI_COMM_WORLD);
      #elif 0
      std::tie(E0, dets, C_local) = asci::asci_iter( asci_settings, mcscf_settings,
        100, E0, std::move(dets), std::move(C_local), ham_gen, n_active, 
        MPI_COMM_WORLD );
      #else
      std::tie(E0, dets, C_local) = asci::asci_grow( asci_settings, 
        mcscf_settings, E0, std::move(dets), std::move(C_local), ham_gen, 
        n_active, MPI_COMM_WORLD );
      std::tie(E0, dets, C_local) = asci::asci_refine( asci_settings, 
        mcscf_settings, E0, std::move(dets), std::move(C_local), ham_gen, 
        n_active, MPI_COMM_WORLD );
      #endif
      E0 += E_inactive + E_core;
        
    }


  // MCSCF
  } else if( job == Job::MCSCF ) {

    // Possibly read active RDMs
    if(rdm_fname.size()) {
      console->info("  * RDMFILE = {}", rdm_fname );
      std::vector<double> full_ordm(norb2), full_trdm(norb4);
      asci::read_rdms_binary(rdm_fname, norb, full_ordm.data(), norb,
        full_trdm.data(), norb );
      asci::active_submatrix_1body(NumActive(n_active), NumInactive(n_inactive),
        full_ordm.data(), norb, active_ordm.data(), n_active);
      asci::active_subtensor_2body(NumActive(n_active), NumInactive(n_inactive),
        full_trdm.data(), norb, active_trdm.data(), n_active);

      // Compute CI energy from RDMs
      double ERDM = blas::dot( active_ordm.size(), active_ordm.data(), 1, 
        T_active.data(), 1 );
      ERDM += blas::dot( active_trdm.size(), active_trdm.data(), 1, 
        V_active.data(), 1 );
      console->info("E(RDM)  = {:.12f} Eh", ERDM + E_inactive + E_core);
    }


    // CASSCF
    E0 =
    asci::casscf_diis( mcscf_settings, NumElectron(nalpha), NumElectron(nbeta),
      NumOrbital(norb), NumInactive(n_inactive), NumActive(n_active),
      NumVirtual(n_virtual), E_core, T.data(), norb, V.data(), norb, 
      active_ordm.data(), n_active, active_trdm.data(), n_active,
      MPI_COMM_WORLD);
  }

  console->info("E(CI)  = {:.12f} Eh", E0);

  if(fci_out_fname.size())
    asci::write_fcidump(fci_out_fname,norb, T.data(), norb, V.data(), norb, E_core);

  } // MPI Scope

  MPI_Finalize();

}
