#pragma once
#include <asci/util/asci_search.hpp>
#include <asci/util/selected_ci_diag.hpp>
#include <asci/util/mcscf.hpp>

namespace asci {

template <size_t N, typename index_t = int32_t>
auto asci_iter( ASCISettings asci_settings, MCSCFSettings mcscf_settings,
  size_t ndets_max, double E0, std::vector<wfn_t<N>> wfn, 
  std::vector<double> X_local, HamiltonianGenerator<N>& ham_gen, size_t norb,
  MPI_Comm comm) {

  // Sort wfn on coefficient weights
  if( wfn.size() > 1 ) reorder_ci_on_coeff( wfn, X_local, comm );

  // Sanity check on search determinants
  size_t nkeep = std::min(asci_settings.ncdets_max, wfn.size());

  // Perform the ASCI search
  wfn = asci_search( asci_settings, ndets_max, wfn.begin(), wfn.begin() + nkeep, 
    E0, X_local, norb, ham_gen.T(), ham_gen.G_red(), ham_gen.V_red(), 
    ham_gen.G(), ham_gen.V(), ham_gen, comm );

  // Rediagonalize
  std::fill(X_local.begin(), X_local.end(), 0); // This precludes guess reuse
  auto E = selected_ci_diag<N,index_t>( wfn.begin(), wfn.end(), ham_gen, 
    mcscf_settings.ci_matel_tol, mcscf_settings.ci_max_subspace, 
    mcscf_settings.ci_res_tol, X_local, comm);

  return std::make_tuple(E, wfn, X_local);

}

}
