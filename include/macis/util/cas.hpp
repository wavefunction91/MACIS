/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/solvers/selected_ci_diag.hpp>
#include <macis/types.hpp>
#include <macis/util/mcscf.hpp>

namespace macis {

/**
 *  @brief Compute the CAS-CI 1- and 2-RDMs
 *
 *  @tparam HamGen Type of the Hamiltonian Generator
 *
 *  @param[in] settings Settings for the CI calculation
 *  @param[in] norb     Number of orbitals
 *  @param[in] nalpha   Number of alpha electrons
 *  @param[in] nbeta    Number of beta electrons
 *  @param[in] Tu       The spin up   one-body Hamiltonian
 *  @param[in] V        The two-body Hamiltonian
 *  @param[out] ORDM    The CAS-CI 1-RDM
 *  @param[out] TRDM    The CAS-CI 2-RDM
 *  @param[out] C       The CAS-CI CI vector
 *  @param[in]  comm    MPI Communicator on which to solve the EVP.
 *  @param[in] Td       The spin down one-body Hamiltonian
 */
template <typename HamGen>
double compute_casci_rdms(
    MCSCFSettings settings, NumOrbital norb, size_t nalpha, size_t nbeta,
    double* Tu, double* V, double* ORDM, double* TRDM,
    std::vector<double>& C MACIS_MPI_CODE(, MPI_Comm comm), double* Td = NULL,
    double* ORDMd = NULL, double* TRDMud = NULL, double* TRDMdu = NULL,
    double* TRDMdd = NULL) {
  constexpr auto nbits = HamGen::nbits;

#ifdef MACIS_ENABLE_MPI
  int rank;
  MPI_Comm_rank(comm, &rank);
#else
  int rank = 0;
#endif

  // Hamiltonian Matrix Element Generator
  size_t no = norb.get();
  HamGen ham_gen(matrix_span<double>(Tu, no, no),
                 rank4_span<double>(V, no, no, no, no));
  if(Td) ham_gen.ReadTdo(matrix_span<double>(Td, no, no));

  // Compute Lowest Energy Eigenvalue (ED)
  auto dets = generate_hilbert_space<nbits>(norb.get(), nalpha, nbeta);
  double E0 =
      selected_ci_diag(dets.begin(), dets.end(), ham_gen, settings.ci_matel_tol,
                       settings.ci_max_subspace, settings.ci_res_tol, C,
                       MACIS_MPI_CODE(comm, ) true, settings.ci_nstates);

  // Compute RDMs
  if(ORDMd == NULL)
    ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                      C.data(), matrix_span<double>(ORDM, no, no),
                      rank4_span<double>(TRDM, no, no, no, no));
  else
    ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                      C.data(), matrix_span<double>(ORDM, no, no),
                      matrix_span<double>(ORDMd, no, no),
                      rank4_span<double>(TRDM, no, no, no, no),
                      rank4_span<double>(TRDMud, no, no, no, no),
                      rank4_span<double>(TRDMdu, no, no, no, no),
                      rank4_span<double>(TRDMdd, no, no, no, no));

  return E0;
}

/// Functor wraper around `compute_casci_rdms`
template <typename HamGen>
struct CASRDMFunctor {
  template <typename... Args>
  static auto rdms(Args&&... args) {
    return compute_casci_rdms<HamGen>(std::forward<Args>(args)...);
  }
};

}  // namespace macis
