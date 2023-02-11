#pragma once
#include <asci/util/asci_iter.hpp>
#include <asci/util/mpi.hpp>
#include <asci/util/transform.hpp>

namespace asci {

template <size_t N, typename index_t = int32_t>
auto asci_grow( ASCISettings asci_settings, MCSCFSettings mcscf_settings,
  double E0, std::vector<wfn_t<N>> wfn, std::vector<double> X,
  HamiltonianGenerator<N>& ham_gen, size_t norb, MPI_Comm comm ) {

  auto world_rank = comm_rank(comm);

  //spdlog::null_logger_mt("asci_search");
  auto logger = spdlog::get("asci_grow");
  if(!logger) logger = world_rank ? 
    spdlog::null_logger_mt ("asci_grow") :
    spdlog::stdout_color_mt("asci_grow");

  logger->info("[ASCI Grow Settings]:");
  logger->info("  NTDETS_MAX = {:6}, NCDETS_MAX = {:6}, GROW_FACTOR = {}",
    asci_settings.ntdets_max, 
    asci_settings.ncdets_max, 
    asci_settings.grow_factor );

  const std::string fmt_string = 
    "iter = {:4}, E0 = {:20.12e}, dE = {:14.6e}, WFN_SIZE = {}";

  logger->info(fmt_string, 0, E0, 0.0, wfn.size());
  // Grow wfn until max size, or until we get stuck
  size_t prev_size = wfn.size();
  size_t iter = 1;
  while( wfn.size() < asci_settings.ntdets_max ) {
    size_t ndets_new = std::min(
      std::max(asci_settings.ntdets_min, wfn.size() * asci_settings.grow_factor),
      asci_settings.ntdets_max
    );
    double E;
    std::tie(E, wfn, X) = asci_iter<N,index_t>( asci_settings,
      mcscf_settings, ndets_new, E0, std::move(wfn), std::move(X),
      ham_gen, norb, comm );
    if( ndets_new != wfn.size() )
      throw std::runtime_error("Wavefunction didn't grow enough...");

    logger->info(fmt_string, iter++, E, E - E0, wfn.size());
    if(asci_settings.grow_with_rot and wfn.size() >= asci_settings.rot_size_start) {
      logger->info("  * Forming RDMs");
      std::vector<double> ordm(norb * norb, 0.0), trdm(norb*norb*norb*norb, 0.0);
      matrix_span<double> ORDM(ordm.data(), norb, norb);
      rank4_span <double> TRDM(trdm.data(), norb, norb, norb, norb);
      ham_gen.form_rdms( wfn.begin(), wfn.end(), wfn.begin(), wfn.end(),
        X.data(), ORDM, TRDM );
      //ham_gen.rotate_hamiltonian_ordm( ordm.data() );
      std::vector<double> ONS(norb);
      for(auto &x : ordm) x *= -1.0;
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb,
        ordm.data(), norb, ONS.data());
      for( auto& x : ONS ) x *= -1.0;
      //for(auto x : ONS) std::cout << x << std::endl;
      logger->info("  * ON_SUM = {}", std::accumulate(ONS.begin(), ONS.end(), 0.0));;

      logger->info("  * Doing Natural Orbital Rotation");
      asci::two_index_transform(norb,norb, ham_gen.T(), norb,
        ordm.data(), norb, ham_gen.T(), norb);
      asci::four_index_transform(norb, norb, 0, 
        ham_gen.V(), norb, ordm.data(), norb, ham_gen.V(),
        norb);
      ham_gen.generate_integral_intermediates(ham_gen.V_pqrs_);
      std::vector<double> C_local;
      logger->info("  * Rediagonalizing");
      selected_ci_diag( wfn.begin(), wfn.end(), ham_gen, 
        mcscf_settings.ci_matel_tol, mcscf_settings.ci_max_subspace,
        mcscf_settings.ci_res_tol, C_local, comm );
      X = std::move(C_local);
    }

      

    E0 = E;
  }

  return std::make_tuple(E0, wfn, X);

}

} // namespace asci_grow
