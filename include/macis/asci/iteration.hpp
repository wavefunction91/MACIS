#pragma once
#include <macis/asci/determinant_search.hpp>
#include <macis/util/selected_ci_diag.hpp>
#include <macis/util/mcscf.hpp>

namespace macis {

template <size_t N, typename index_t = int32_t>
auto asci_iter( ASCISettings asci_settings, MCSCFSettings mcscf_settings,
  size_t ndets_max, double E0, std::vector<wfn_t<N>> wfn, 
  std::vector<double> X, HamiltonianGenerator<N>& ham_gen, size_t norb,
  MPI_Comm comm) {

  // Sort wfn on coefficient weights
  if( wfn.size() > 1 ) reorder_ci_on_coeff( wfn, X );

  // Sanity check on search determinants
  size_t nkeep = std::min(asci_settings.ncdets_max, wfn.size());

  // Perform the ASCI search
  wfn = asci_search( asci_settings, ndets_max, wfn.begin(), 
    wfn.begin() + nkeep, E0, X, norb, ham_gen.T(), ham_gen.G_red(), 
    ham_gen.V_red(), ham_gen.G(), ham_gen.V(), ham_gen, comm );

  // Rediagonalize
  std::vector<double> X_local; // Precludes guess reuse
  auto E = selected_ci_diag<N,index_t>( wfn.begin(), wfn.end(), ham_gen, 
    mcscf_settings.ci_matel_tol, mcscf_settings.ci_max_subspace, 
    mcscf_settings.ci_res_tol, X_local, comm);

  auto world_size = comm_size(comm);
  auto world_rank = comm_rank(comm);
  if( world_size > 1 ) {
    // Broadcast X_local to X
    const size_t wfn_size = wfn.size();
    const size_t local_count = wfn_size / world_size;
    X.resize(wfn.size());
    
    MPI_Allgather( X_local.data(), local_count, MPI_DOUBLE, 
      X.data(), local_count, MPI_DOUBLE, comm );
    if( wfn_size % world_size ) {
      const size_t nrem = wfn_size % world_size;
      auto* X_rem = X.data() + world_size * local_count;
      if( world_rank == world_size - 1 ) {
        const auto* X_loc_rem = X_local.data() + local_count;
        std::copy_n( X_loc_rem, nrem, X_rem );
      }
      MPI_Bcast( X_rem, nrem, MPI_DOUBLE, world_size-1, comm );
    }
  } else {
    // Avoid copy
    X = std::move(X_local);
  }

  return std::make_tuple(E, wfn, X);

}

}
