#include "ut_common.hpp"
#include <asci/fcidump.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/davidson.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <iostream>
#include <iomanip>

TEST_CASE("Davidson") {

  ROOT_ONLY(MPI_COMM_WORLD);

  if(!spdlog::get("davidson")) {
    spdlog::null_logger_mt("davidson");
  }

  size_t norb = asci::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = asci::read_fcidump_core(water_ccpvdz_fcidump);
  asci::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  asci::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  
  using generator_type = asci::DoubleLoopHamiltonianGenerator<64>;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
    asci::matrix_span<double>(T.data(),norb,norb),
    asci::rank4_span<double>(V.data(),norb,norb,norb,norb)
  );
#endif

  // Generate configuration space
  const auto hf_det = asci::canonical_hf_determinant<64>(nocc, nocc);
  auto dets = asci::generate_cisd_hilbert_space( norb, hf_det );
  auto E0_ref = -7.623197835987e+01;

  // Generate CSR Hamiltonian
  auto H = asci::make_csr_hamiltonian_block<int32_t>( dets.begin(), dets.end(),
    dets.begin(), dets.end(), ham_gen, 1e-16 );


  // Obtain lowest eigenvalue
  SECTION("With Vectors") {
    std::vector<double> X(H.n());
    asci::diagonal_guess(H.n(), H, X.data());
    auto D = sparsexx::extract_diagonal_elements(H);
    auto [niter,E0] = asci::davidson(H.n(), 15, asci::SparseMatrixOperator(H), 
      D.data(), 1e-8, X.data());

    REQUIRE( E0 + E_core == Approx(E0_ref) );
    REQUIRE( blas::nrm2(X.size(), X.data(), 1) == Approx(1.0) );
    
    std::vector<double> AX(X.size());
    sparsexx::spblas::gespmbv(1, 1., H, X.data(), H.n(), 0., AX.data(), H.n());
    REQUIRE( blas::dot(X.size(), X.data(), 1, AX.data(), 1) == Approx(E0) );
  }


}



TEST_CASE("Parallel Davidson") {

  if(!spdlog::get("davidson")) {
    auto l = spdlog::null_logger_mt("davidson");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  size_t norb = asci::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = asci::read_fcidump_core(water_ccpvdz_fcidump);
  asci::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  asci::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  
  using generator_type = asci::DoubleLoopHamiltonianGenerator<64>;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
    asci::matrix_span<double>(T.data(),norb,norb),
    asci::rank4_span<double>(V.data(),norb,norb,norb,norb)
  );
#endif

  // Generate configuration space
  const auto hf_det = asci::canonical_hf_determinant<64>(nocc, nocc);
  auto dets = asci::generate_cisd_hilbert_space( norb, hf_det );
  auto E0_ref = -7.623197835987e+01;

  // Generate CSR Hamiltonian
  auto H = asci::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, 
    dets.begin(), dets.end(), ham_gen, 1e-16 );
  auto spmv_info = sparsexx::spblas::generate_spmv_comm_info( H );


  // Obtain lowest eigenvalue
  SECTION("With Vectors") {
    std::vector<double> X_local(H.local_row_extent());
    asci::p_diagonal_guess(X_local.size(), H, X_local.data());
    auto D_local = sparsexx::extract_diagonal_elements(H.diagonal_tile());
    auto [niter,E0] = 
      asci::p_davidson(X_local.size(), 15, asci::SparseMatrixOperator(H), 
        D_local.data(), 1e-8, X_local.data(), MPI_COMM_WORLD);

    REQUIRE( E0 + E_core == Approx(E0_ref) );
    double nrm = blas::dot(X_local.size(), X_local.data(), 1, X_local.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nrm = std::sqrt(nrm);
    REQUIRE( nrm  == Approx(1.0) );
    
    std::vector<double> AX_local(X_local.size());
    sparsexx::spblas::pgespmv(1., H, X_local.data(), 0., AX_local.data(), 
      spmv_info );
    double inner = blas::dot(X_local.size(), AX_local.data(), 1, 
      X_local.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &inner, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    REQUIRE( inner  == Approx(E0) );
  }

  MPI_Barrier(MPI_COMM_WORLD);
  spdlog::drop_all();
}
