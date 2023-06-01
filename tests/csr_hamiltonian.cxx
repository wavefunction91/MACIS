/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include "ut_common.hpp"
#include <macis/fcidump.hpp>
#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>


TEST_CASE("CSR Hamiltonian") {

  ROOT_ONLY(MPI_COMM_WORLD);

  size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  
  using generator_type = macis::DoubleLoopHamiltonianGenerator<64>;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
    macis::matrix_span<double>(T.data(),norb,norb),
    macis::rank4_span<double>(V.data(),norb,norb,norb,norb)
  );
#endif

  // Generate configuration space
  const auto hf_det = macis::canonical_hf_determinant<64>(nocc, nocc);
  auto dets = macis::generate_cisd_hilbert_space( norb, hf_det );

  // Generate CSR Hamiltonian
  auto H = macis::make_csr_hamiltonian_block<int32_t>( dets.begin(), dets.end(),
    dets.begin(), dets.end(), ham_gen, 1e-16 );

  // Read reference data
  std::vector<int32_t> ref_rowptr(H.rowptr().size()), ref_colind(H.colind().size());
  std::vector<double>  ref_nzval(H.nzval().size());

  std::ifstream rowptr_file(water_ccpvdz_rowptr_fname,std::ios::binary);
  rowptr_file.read( (char*)ref_rowptr.data(), ref_rowptr.size()*sizeof(int32_t));

  std::ifstream colind_file(water_ccpvdz_colind_fname,std::ios::binary);
  colind_file.read( (char*)ref_colind.data(), ref_colind.size()*sizeof(int32_t));

  std::ifstream nzval_file(water_ccpvdz_nzval_fname,std::ios::binary);
  nzval_file.read( (char*)ref_nzval.data(), ref_nzval.size()*sizeof(double));
  
  REQUIRE( H.rowptr() == ref_rowptr );
  REQUIRE( H.colind() == ref_colind );

  const size_t nnz = H.nnz();
  for( auto i = 0ul; i < nnz; ++i ) {
    REQUIRE( H.nzval()[i] == Approx(ref_nzval[i]));
  }
}



TEST_CASE("Distributed CSR Hamiltonian") {

  MPI_Barrier(MPI_COMM_WORLD);
  size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  
  using generator_type = macis::DoubleLoopHamiltonianGenerator<64>;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
    macis::matrix_span<double>(T.data(),norb,norb),
    macis::rank4_span<double>(V.data(),norb,norb,norb,norb)
  );
#endif

  // Generate configuration space
  const auto hf_det = macis::canonical_hf_determinant<64>(nocc, nocc);
  auto dets = macis::generate_cisd_hilbert_space( norb, hf_det );

  // Generate Distributed CSR Hamiltonian
  auto H_dist = macis::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, 
    dets.begin(), dets.end(), ham_gen, 1e-16 );

  // Generate Replicated CSR Hamiltonian 
  auto H = macis::make_csr_hamiltonian_block<int32_t>( dets.begin(), dets.end(),
    dets.begin(), dets.end(), ham_gen, 1e-16 );

  // Distribute replicated matrix
  decltype(H_dist) H_dist_ref(MPI_COMM_WORLD, H);

  REQUIRE( H_dist.diagonal_tile().rowptr() == H_dist_ref.diagonal_tile().rowptr() );
  REQUIRE( H_dist.diagonal_tile().colind() == H_dist_ref.diagonal_tile().colind() );

  size_t nnz_local = H_dist.diagonal_tile().nnz();
  for( auto i = 0ul; i < nnz_local; ++i ) {
    REQUIRE( H_dist.diagonal_tile().nzval()[i] == 
             Approx(H_dist_ref.diagonal_tile().nzval()[i] ) );
  }

  int mpi_size; MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if(mpi_size > 1 ) {
    REQUIRE( H_dist.off_diagonal_tile().rowptr() == H_dist_ref.off_diagonal_tile().rowptr() );
    REQUIRE( H_dist.off_diagonal_tile().colind() == H_dist_ref.off_diagonal_tile().colind() );
    nnz_local = H_dist.off_diagonal_tile().nnz();
    for( auto i = 0ul; i < nnz_local; ++i ) {
      REQUIRE( H_dist.off_diagonal_tile().nzval()[i] == 
               Approx(H_dist_ref.off_diagonal_tile().nzval()[i] ) );
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);
}
