/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <iomanip>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/mcscf/mcscf.hpp>

#include "ut_common.hpp"

TEST_CASE("MCSCF") {
  ROOT_ONLY(MPI_COMM_WORLD);

  spdlog::null_logger_mt("davidson");
  spdlog::null_logger_mt("ci_solver");
  spdlog::null_logger_mt("diis");
  spdlog::null_logger_mt("mcscf");

  const size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  const size_t norb2 = norb * norb;
  const size_t norb4 = norb2 * norb2;

  using macis::NumActive;
  using macis::NumElectron;
  using macis::NumInactive;
  using macis::NumOrbital;
  using macis::NumVirtual;

  std::vector<double> T(norb2), V(norb4);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  size_t n_inactive = 1;
  size_t n_active = 8;
  size_t n_virtual = norb - n_inactive - n_active;
  NumElectron nalpha(4);

  NumInactive ninact(n_inactive);
  NumActive nact(n_active);
  NumVirtual nvirt(n_virtual);

  size_t na2 = n_active * n_active;
  size_t na4 = na2 * na2;
  std::vector<double> active_ordm(na2), active_trdm(na4);
  macis::MCSCFSettings settings;

  const double ref_E = -76.1114493227;

  SECTION("CASSCF - No Guess - Singlet") {
    auto E = macis::casscf_diis(
        settings, nalpha, nalpha, NumOrbital(norb), ninact, nact, nvirt, E_core,
        T.data(), norb, V.data(), norb, active_ordm.data(), n_active,
        active_trdm.data(),
        n_active MACIS_MPI_CODE(, MPI_COMM_SELF /*b/c root only*/));

    REQUIRE(E == Approx(ref_E).margin(1e-7));
  }

  SECTION("CASSCF - With Guess - Singlet") {
    macis::read_rdms_binary(water_ccpvdz_rdms_fname, n_active,
                            active_ordm.data(), n_active, active_trdm.data(),
                            n_active);
    auto E = macis::casscf_diis(
        settings, nalpha, nalpha, NumOrbital(norb), ninact, nact, nvirt, E_core,
        T.data(), norb, V.data(), norb, active_ordm.data(), n_active,
        active_trdm.data(),
        n_active MACIS_MPI_CODE(, MPI_COMM_SELF /*b/c root only*/));

    REQUIRE(E == Approx(ref_E).margin(1e-7));
  }

  spdlog::drop_all();
}
