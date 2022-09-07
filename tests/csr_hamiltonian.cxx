#include "ut_common.hpp"
#include <asci/fcidump.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>

#define REF_DATA_PREFIX "/home/dbwy/devel/casscf/ASCI-CI/tests/ref_data"
const std::string ref_fcidump = REF_DATA_PREFIX "/h2o.ccpvdz.fci.dat";
const std::string ref_rowptr_fname  = REF_DATA_PREFIX "/h2o.ccpvdz.cisd.rowptr.bin";
const std::string ref_colind_fname  = REF_DATA_PREFIX "/h2o.ccpvdz.cisd.colind.bin";
const std::string ref_nzval_fname   = REF_DATA_PREFIX "/h2o.ccpvdz.cisd.nzval.bin";

TEST_CASE("CSR Hamiltonian") {

  ROOT_ONLY(MPI_COMM_WORLD);

  size_t norb = asci::read_fcidump_norb(ref_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = asci::read_fcidump_core(ref_fcidump);
  asci::read_fcidump_1body(ref_fcidump, T.data(), norb);
  asci::read_fcidump_2body(ref_fcidump, V.data(), norb);

  
  using generator_type = asci::DoubleLoopHamiltonianGenerator<64>;

  generator_type ham_gen(norb, V.data(), T.data());

  // Generate configuration space
  const auto hf_det = asci::canonical_hf_determinant<64>(nocc, nocc);
  auto dets = asci::generate_cisd_hilbert_space( norb, hf_det );

  // Generate CSR Hamiltonian
  auto H = asci::make_csr_hamiltonian_block<int32_t>( dets.begin(), dets.end(),
    dets.begin(), dets.end(), ham_gen, 1e-16 );

  // Read reference data
  std::vector<int32_t> ref_rowptr(H.rowptr().size()), ref_colind(H.colind().size());
  std::vector<double>  ref_nzval(H.nzval().size());

  std::ifstream rowptr_file(ref_rowptr_fname,std::ios::binary);
  rowptr_file.read( (char*)ref_rowptr.data(), ref_rowptr.size()*sizeof(int32_t));

  std::ifstream colind_file(ref_colind_fname,std::ios::binary);
  colind_file.read( (char*)ref_colind.data(), ref_colind.size()*sizeof(int32_t));

  std::ifstream nzval_file(ref_nzval_fname,std::ios::binary);
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
  size_t norb = asci::read_fcidump_norb(ref_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = asci::read_fcidump_core(ref_fcidump);
  asci::read_fcidump_1body(ref_fcidump, T.data(), norb);
  asci::read_fcidump_2body(ref_fcidump, V.data(), norb);

  
  using generator_type = asci::DoubleLoopHamiltonianGenerator<64>;

  generator_type ham_gen(norb, V.data(), T.data());

  // Generate configuration space
  const auto hf_det = asci::canonical_hf_determinant<64>(nocc, nocc);
  auto dets = asci::generate_cisd_hilbert_space( norb, hf_det );

  // Generate Distributed CSR Hamiltonian
  auto H_dist = asci::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, 
    dets.begin(), dets.end(), ham_gen, 1e-16 );

  // Generate Replicated CSR Hamiltonian 
  auto H = asci::make_csr_hamiltonian_block<int32_t>( dets.begin(), dets.end(),
    dets.begin(), dets.end(), ham_gen, 1e-16 );

  // Distribute replicated matrix
  decltype(H_dist) H_dist_ref(MPI_COMM_WORLD, H);

  REQUIRE( H_dist.diagonal_tile().rowptr() == H_dist_ref.diagonal_tile().rowptr() );
  REQUIRE( H_dist.diagonal_tile().colind() == H_dist_ref.diagonal_tile().colind() );

  REQUIRE( H_dist.off_diagonal_tile().rowptr() == H_dist_ref.off_diagonal_tile().rowptr() );
  REQUIRE( H_dist.off_diagonal_tile().colind() == H_dist_ref.off_diagonal_tile().colind() );

  size_t nnz_local = H_dist.diagonal_tile().nnz();
  for( auto i = 0ul; i < nnz_local; ++i ) {
    REQUIRE( H_dist.diagonal_tile().nzval()[i] == 
             Approx(H_dist_ref.diagonal_tile().nzval()[i] ) );
  }

  nnz_local = H_dist.off_diagonal_tile().nnz();
  for( auto i = 0ul; i < nnz_local; ++i ) {
    REQUIRE( H_dist.off_diagonal_tile().nzval()[i] == 
             Approx(H_dist_ref.off_diagonal_tile().nzval()[i] ) );
  }


  MPI_Barrier(MPI_COMM_WORLD);
}
