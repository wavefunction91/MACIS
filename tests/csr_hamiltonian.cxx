/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/util/fcidump.hpp>

#include "ut_common.hpp"

using wfn_type = macis::wfn_t<64>;
TEMPLATE_TEST_CASE("CSR Hamiltonian", "[ham_gen]",
                   macis::DoubleLoopHamiltonianGenerator<wfn_type>,
                   macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>) {
  ROOT_ONLY(MPI_COMM_WORLD);

  size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb * norb);
  std::vector<double> V(norb * norb * norb * norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = TestType;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
#endif

  // Generate configuration space
  const auto hf_det = wfn_traits::canonical_hf_determinant(nocc, nocc);
  auto dets = macis::generate_cisd_hilbert_space(norb, hf_det);
  std::sort(dets.begin(), dets.end(), wfn_traits::spin_comparator{});

  // Generate CSR Hamiltonian
  auto st = std::chrono::high_resolution_clock::now();
  auto H = macis::make_csr_hamiltonian_block<int32_t>(
      dets.begin(), dets.end(), dets.begin(), dets.end(), ham_gen, 1e-16);
  auto en = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double, std::milli>(en - st).count()
            << std::endl;

// #define GENERATE_TESTS
#ifdef GENERATE_TESTS
  std::string tmp_rowptr_fname = "tmp_rowptr.bin";
  std::string tmp_colind_fname = "tmp_colind.bin";
  std::string tmp_nzval_fname = "tmp_nzval.bin";

  std::ofstream rowptr_file(tmp_rowptr_fname, std::ios::binary);
  rowptr_file.write((char*)H.rowptr().data(),
                    H.rowptr().size() * sizeof(int32_t));
  std::ofstream colind_file(tmp_colind_fname, std::ios::binary);
  colind_file.write((char*)H.colind().data(),
                    H.colind().size() * sizeof(int32_t));
  std::ofstream nzval_file(tmp_nzval_fname, std::ios::binary);
  nzval_file.write((char*)H.nzval().data(), H.nzval().size() * sizeof(double));
#else

  // std::cout << "NEW H " << H.m() << " " << H.nnz() << " " << H.indexing() <<
  // std::endl;

  // Read reference data
  std::vector<int32_t> ref_rowptr(H.rowptr().size()),
      ref_colind(H.colind().size());
  std::vector<double> ref_nzval(H.nzval().size());

  std::ifstream rowptr_file(water_ccpvdz_rowptr_fname, std::ios::binary);
  rowptr_file.read((char*)ref_rowptr.data(),
                   ref_rowptr.size() * sizeof(int32_t));

  std::ifstream colind_file(water_ccpvdz_colind_fname, std::ios::binary);
  colind_file.read((char*)ref_colind.data(),
                   ref_colind.size() * sizeof(int32_t));

  std::ifstream nzval_file(water_ccpvdz_nzval_fname, std::ios::binary);
  nzval_file.read((char*)ref_nzval.data(), ref_nzval.size() * sizeof(double));

  REQUIRE(H.rowptr() == ref_rowptr);
  REQUIRE(H.colind() == ref_colind);

  const size_t nnz = H.nnz();
  for(auto i = 0ul; i < nnz; ++i) {
    REQUIRE(H.nzval()[i] == Approx(ref_nzval[i]));
  }
#endif
}

#ifdef MACIS_ENABLE_MPI
TEMPLATE_TEST_CASE("Distributed CSR Hamiltonian", "[ham_gen]",
                   macis::DoubleLoopHamiltonianGenerator<wfn_type>,
                   macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>) {
  MPI_Barrier(MPI_COMM_WORLD);
  size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb * norb);
  std::vector<double> V(norb * norb * norb * norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = TestType;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
#endif

  // Generate configuration space
  const auto hf_det = wfn_traits::canonical_hf_determinant(nocc, nocc);
  auto dets = macis::generate_cisd_hilbert_space(norb, hf_det);
  std::sort(dets.begin(), dets.end(), wfn_traits::spin_comparator{});

  // Generate Distributed CSR Hamiltonian
  auto H_dist = macis::make_dist_csr_hamiltonian<int32_t>(
      MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, 1e-16);

  // Generate Replicated CSR Hamiltonian
  auto H = macis::make_csr_hamiltonian_block<int32_t>(
      dets.begin(), dets.end(), dets.begin(), dets.end(), ham_gen, 1e-16);

  // Distribute replicated matrix
  decltype(H_dist) H_dist_ref(MPI_COMM_WORLD, H);

  REQUIRE(H_dist.diagonal_tile().rowptr() ==
          H_dist_ref.diagonal_tile().rowptr());
  REQUIRE(H_dist.diagonal_tile().colind() ==
          H_dist_ref.diagonal_tile().colind());

  size_t nnz_local = H_dist.diagonal_tile().nnz();
  for(auto i = 0ul; i < nnz_local; ++i) {
    REQUIRE(H_dist.diagonal_tile().nzval()[i] ==
            Approx(H_dist_ref.diagonal_tile().nzval()[i]));
  }

  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if(mpi_size > 1) {
    REQUIRE(H_dist.off_diagonal_tile().rowptr() ==
            H_dist_ref.off_diagonal_tile().rowptr());
    REQUIRE(H_dist.off_diagonal_tile().colind() ==
            H_dist_ref.off_diagonal_tile().colind());
    nnz_local = H_dist.off_diagonal_tile().nnz();
    for(auto i = 0ul; i < nnz_local; ++i) {
      REQUIRE(H_dist.off_diagonal_tile().nzval()[i] ==
              Approx(H_dist_ref.off_diagonal_tile().nzval()[i]));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif
