/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <iomanip>
#include <iostream>
#include <macis/hamiltonian_generator/sd_build.hpp>
#include <macis/util/cas.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/wavefunction_io.hpp>

#include "ut_common.hpp"

using macis::NumActive;
using macis::NumInactive;
using macis::NumOrbital;

TEST_CASE("Single Double Build") {
  ROOT_ONLY(MPI_COMM_WORLD);

  auto norb = macis::read_fcidump_norb(hubbard10_fcidump);
  const auto norb2 = norb * norb;
  const auto norb3 = norb2 * norb;
  const size_t nocc = 10;

  std::vector<double> T(norb * norb);
  std::vector<double> V(norb * norb * norb * norb);
  auto E_core = macis::read_fcidump_core(hubbard10_fcidump);
  macis::read_fcidump_1body(hubbard10_fcidump, T.data(), norb);
  macis::read_fcidump_2body(hubbard10_fcidump, V.data(), norb);
  bool just_singles = macis::is_2body_diagonal(hubbard10_fcidump);

  using generator_type = macis::SDBuildHamiltonianGenerator<64>;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
  ham_gen.SetJustSingles(just_singles);
#endif
  const auto hf_det = macis::canonical_hf_determinant<64>(nocc, nocc);

  std::vector<double> eps(norb);
  for(auto p = 0ul; p < norb; ++p) {
    double tmp = 0.;
    for(auto i = 0ul; i < nocc; ++i) {
      tmp += 2. * V[p * (norb + 1) + i * (norb2 + norb3)] -
             V[p * (1 + norb3) + i * (norb + norb2)];
    }
    eps[p] = T[p * (norb + 1)] + tmp;
  }
  const auto EHF = ham_gen.matrix_element(hf_det, hf_det);

  SECTION("HF Energy") { REQUIRE(EHF + E_core == Approx(-0.00)); }

  SECTION("RDM") {
    std::vector<double> ordm(norb * norb, 0.0), trdm(norb3 * norb, 0.0);
    std::vector<std::bitset<64>> dets = {
        macis::canonical_hf_determinant<64>(nocc, nocc)};

    std::vector<double> C = {1.};

    ham_gen.form_rdms(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C.data(),
        macis::matrix_span<double>(ordm.data(), norb, norb),
        macis::rank4_span<double>(trdm.data(), norb, norb, norb, norb));

    auto E_tmp = blas::dot(norb2, ordm.data(), 1, T.data(), 1) +
                 blas::dot(norb3 * norb, trdm.data(), 1, V.data(), 1);
    REQUIRE(E_tmp == Approx(EHF));
  }

  SECTION("CI") {
    size_t nalpha = 5;
    size_t nbeta = 5;
    size_t n_inactive = 0;
    size_t n_active = 10;
    size_t n_virtual = 0;
    std::vector<double> C_local;
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm;
    macis::MCSCFSettings mcscf_settings;
    mcscf_settings.ci_max_subspace = 100;
    // Copy integrals into active subsets
    std::vector<double> T_active(n_active * n_active);
    std::vector<double> V_active(n_active * n_active * n_active * n_active);

    // Compute active-space Hamiltonian and inactive Fock matrix
    std::vector<double> F_inactive(norb2);
    macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                              NumInactive(n_inactive), T.data(), norb, V.data(),
                              norb, F_inactive.data(), norb, T_active.data(),
                              n_active, V_active.data(), n_active);
    auto E_inactive = macis::inactive_energy(NumInactive(n_inactive), T.data(),
                                             norb, F_inactive.data(), norb);
    auto dets = macis::generate_hilbert_space<64>(norb, nalpha, nbeta);
    double E0 = macis::selected_ci_diag(
        dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
        mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
        MACIS_MPI_CODE(MPI_COMM_WORLD, ) true, mcscf_settings.ci_nstates);
    E0 += E_inactive + E_core;
    // Compute RDMs
    ham_gen.form_rdms(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C_local.data(),
        macis::matrix_span<double>(active_ordm.data(), norb, norb),
        macis::rank4_span<double>(active_trdm.data(), norb, norb, norb, norb));
    REQUIRE(E0 == Approx(-2.538061882041e+01));
  }
}
