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
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

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
  auto grow_st = hrt_t::now();
  while( wfn.size() < asci_settings.ntdets_max ) {
    size_t ndets_new = std::min(
      std::max(asci_settings.ntdets_min, wfn.size() * asci_settings.grow_factor),
      asci_settings.ntdets_max
    );
    double E;
    auto ai_st = hrt_t::now();
    std::tie(E, wfn, X) = asci_iter<N,index_t>( asci_settings,
      mcscf_settings, ndets_new, E0, std::move(wfn), std::move(X),
      ham_gen, norb, comm );
    auto ai_en = hrt_t::now();
    dur_t ai_dur = ai_en - ai_st;
    logger->trace("  * ASCI_ITER_DUR = {:.2e} ms", ai_dur.count() );
    if( ndets_new > wfn.size() )
      throw std::runtime_error("Wavefunction didn't grow enough...");

    logger->info(fmt_string, iter++, E, E - E0, wfn.size());
    if(asci_settings.grow_with_rot and wfn.size() >= asci_settings.rot_size_start) {
      auto grow_rot_st = hrt_t::now();

      // Form RDMs: TODO Make 1RDM-only work
      logger->trace("  * Forming RDMs");
      auto rdm_st = hrt_t::now();
      std::vector<double> ordm(norb * norb, 0.0), trdm(norb*norb*norb*norb, 0.0);
      matrix_span<double> ORDM(ordm.data(), norb, norb);
      rank4_span <double> TRDM(trdm.data(), norb, norb, norb, norb);
      ham_gen.form_rdms( wfn.begin(), wfn.end(), wfn.begin(), wfn.end(),
        X.data(), ORDM, TRDM );
      auto rdm_en = hrt_t::now();
      dur_t rdm_dur = rdm_en - rdm_st;
      logger->trace("    * RDM_DUR = {:.2e} ms", rdm_dur.count() );

      // Compute Natural Orbitals
      logger->trace("  * Forming Natural Orbitals");
      auto nos_st = hrt_t::now();
      std::vector<double> ONS(norb);
      for(auto &x : ordm) x *= -1.0;
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb,
        ordm.data(), norb, ONS.data());
      for( auto& x : ONS ) x *= -1.0;
      //for(auto x : ONS) std::cout << x << std::endl;
      auto nos_en = hrt_t::now();
      dur_t nos_dur = nos_en - nos_st;
      logger->trace("    * NOS_DUR = {:.2e} ms", nos_dur.count() );

      logger->debug("  * ON_SUM = {.6f}", 
        std::accumulate(ONS.begin(), ONS.end(), 0.0));;

      logger->trace("  * Doing Natural Orbital Rotation");
      auto rot_st = hrt_t::now();
      asci::two_index_transform(norb,norb, ham_gen.T(), norb,
        ordm.data(), norb, ham_gen.T(), norb);
      asci::four_index_transform(norb, norb, 0, 
        ham_gen.V(), norb, ordm.data(), norb, ham_gen.V(),
        norb);
      ham_gen.generate_integral_intermediates(ham_gen.V_pqrs_);
      auto rot_en = hrt_t::now();
      dur_t rot_dur = rot_en - rot_st;
      logger->trace("    * ROT_DUR = {:.2e} ms", rot_dur.count() );

      logger->trace("  * Rediagonalizing");
      auto rdg_st = hrt_t::now();
      std::vector<double> C_local;
      selected_ci_diag( wfn.begin(), wfn.end(), ham_gen, 
        mcscf_settings.ci_matel_tol, mcscf_settings.ci_max_subspace,
        mcscf_settings.ci_res_tol, C_local, comm );
      // TODO: Scatter
      X = std::move(C_local);
      auto rdg_en = hrt_t::now();
      dur_t rdg_dur = rdg_en - rdg_st;
      logger->trace("    * ReDiag_DUR = {:.2e} ms", rdg_dur.count());


      auto grow_rot_en = hrt_t::now();
      logger->trace("  * GROW_ROT_DUR = {:.2e} ms", 
        dur_t(grow_rot_en-grow_rot_st).count());
    }

      

    E0 = E;
  }
  auto grow_en = hrt_t::now();
  dur_t grow_dur = grow_en - grow_st;
  logger->info("* GROW_DUR = {:.2e} ms", grow_dur.count() );

  return std::make_tuple(E0, wfn, X);

}

} // namespace asci_grow
