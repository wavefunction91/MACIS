/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/util/mcscf.hpp>
#include <macis/types.hpp>
#include <macis/util/selected_ci_diag.hpp>

namespace macis {

template <typename HamGen>
double compute_casci_rdms(MCSCFSettings settings, NumOrbital norb, 
  size_t nalpha, size_t nbeta, double* T, double* V, double* ORDM, 
  double* TRDM, std::vector<double>& C, MPI_Comm comm) {

  constexpr auto nbits = HamGen::nbits;

  int rank; MPI_Comm_rank(comm, &rank);

  // Hamiltonian Matrix Element Generator
  size_t no = norb.get();
  HamGen ham_gen( 
    matrix_span<double>(T,no,no),
    rank4_span<double>(V,no,no,no,no) 
  );
  
  // Compute Lowest Energy Eigenvalue (ED)
  auto dets = generate_hilbert_space<nbits>(norb.get(), nalpha, nbeta);
  double E0 = selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    settings.ci_matel_tol, settings.ci_max_subspace, settings.ci_res_tol, C, 
    comm, true);

  // Compute RDMs
  ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
    C.data(), matrix_span<double>(ORDM,no,no), 
    rank4_span<double>(TRDM,no,no,no,no));

  return E0;
}


template <typename HamGen>
struct CASRDMFunctor {
  template <typename... Args>
  static auto rdms(Args&&... args) {
    return compute_casci_rdms<HamGen>(std::forward<Args>(args)...);
  }
};

}
