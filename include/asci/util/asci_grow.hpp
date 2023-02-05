#pragma once
#include <asci/util/asci_iter.hpp>
#include <asci/util/mpi.hpp>

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
      std::max(100ul, wfn.size() * asci_settings.grow_factor),
      asci_settings.ntdets_max
    );
    double E;
    std::tie(E, wfn, X) = asci_iter<N,index_t>( asci_settings,
      mcscf_settings, ndets_new, E0, std::move(wfn), std::move(X),
      ham_gen, norb, comm );
    if( ndets_new != wfn.size() )
      throw std::runtime_error("Wavefunction didn't grow enough...");

    logger->info(fmt_string, iter++, E, E - E0, wfn.size());
    E0 = E;
  }

  return std::make_tuple(E0, wfn, X);

}

} // namespace asci_grow
