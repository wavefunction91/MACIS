/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests H2O 6-31g active space (8e, 5o) Hamiltonian.
 ******************************************/

#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include <iostream>
#include <fstream>
#include "unsupported/Eigen/SparseExtra"

#include <lobpcgxx/lobpcg.hpp>
#include <random>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/util/graph.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/io/read_mm.hpp>
#include "csr_hamiltonian.hpp"

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;

#include <random>

using namespace std;
using namespace cmz::ed;

void gram_schmidt( int64_t N, int64_t K, const double* V_old, int64_t LDV,
  double* V_new ) {

  std::vector<double> inner(K);
  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, 1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K );
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    N, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N );

  blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    K, 1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K );
  blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    N, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N );

  auto nrm = blas::nrm2(N,V_new,1);
  blas::scal( N, 1./nrm, V_new, 1 );
}  


void davidson( int64_t max_m, const sparsexx::csr_matrix<double,int32_t>& A, double tol ) {

  int64_t N = A.n();
  std::cout << "N = " << N << ", MAX_M = " << max_m << std::endl;
  std::vector<double> V(N * (max_m+1)), AV(N * (max_m+1)), C((max_m+1)*(max_m+1)), 
    LAM(max_m+1);

  // Extract diagonal and setup guess
  auto D = extract_diagonal_elements( A );
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance( D.begin(), D_min );
  V[min_idx] = 1.;

  // Prime iterations
  // AV(:,0) = A * V(:,0)
  // V(:,1)  = AV(:,0) - V(:,0) * <V(:,0), AV(:,0)>
  // V(:,1)  = V(:,1) / NORM(V(:,1))
  sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., V.data()+N, N);
  gram_schmidt( N, 1, V.data(), N, V.data()+N );
  //std::cout << blas::nrm2( N, V.data() + N, 1 ) << std::endl;


  for( size_t i = 1; i < max_m; ++i ) {

    // AV(:,i) = A * V(:,i)
    sparsexx::spblas::gespmbv( i+1, 1., A, V.data(), N, 0., AV.data(), N );

    const auto k = i + 1;

    // Rayleigh Ritz
    //blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
    //  k, k, N, 1., V.data(), N, AV.data(), N, 0., C.data(), k );
    //std::cout << "MATRIX" << std::endl;
    //for( auto j = 0; j < k; ++j ) {
    //  for( auto l = 0; l < k; ++l ) std::cout << C[l + j*k] << ", ";
    //  std::cout << std::endl;
    //}
    lobpcgxx::rayleigh_ritz( N, k, V.data(), N, AV.data(), N, LAM.data(),
      C.data(), k);
    //std::cout << "Eigenvectors" << std::endl;
    //for( auto j = 0; j < k; ++j ) {
    //  for( auto l = 0; l < k; ++l ) std::cout << C[l + j*k] << ", ";
    //  std::cout << std::endl;
    //}
    //std::cout << std::endl;

    //std::cout << "LAMBDA ";
    //for( auto j = 0; j < k; ++j ) { std::cout << LAM[j] << ", "; } std::cout << std::endl;
    //std::cout << std::endl;

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    double* R = V.data() + (i+1)*N;
    #if 1
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      N, 1, k, 1., AV.data(), N, C.data(), k, 0., R, N );
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      N, 1, k, -LAM[0], V.data(), N, C.data(), k, 1., R, N );
    #endif

    // Compute residual norm
    auto res_nrm = blas::nrm2( N, R, 1 );
    std::cout << std::scientific << std::setprecision(12);
    std::cout << i << ", " << LAM[0] << ", " << res_nrm << std::endl;
    if( res_nrm < tol ) break;

    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R
    for( auto j = 0; j < N; ++j ) {
      R[j] = - R[j] / (D[j] - LAM[0]);
      //R[j] = - R[j];
    }

    // Project new vector out form old vectors
    std::vector<double> inner(k);
    blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
      k, 1, N, 1., V.data(), N, R, N, 0., inner.data(), k );
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      N, 1, k, -1., V.data(), N, inner.data(), k, 1., R, N );
    blas::gemm( blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
      k, 1, N, 1., V.data(), N, R, N, 0., inner.data(), k );
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      N, 1, k, -1., V.data(), N, inner.data(), k, 1., R, N );

    // Normalize new vector
    auto nrm = blas::nrm2(N, R, 1);
    blas::scal( N, 1./nrm, R, 1 );

  } // Davidson iterations

}


int main( int argn, char* argv[] )
{

  MPI_Init(NULL,NULL);
  const auto world_rank = sparsexx::detail::get_mpi_rank(MPI_COMM_WORLD);
  const auto world_size = sparsexx::detail::get_mpi_size(MPI_COMM_WORLD);

  if( argn != 2 ) {
    cout << "Usage: " << argv[0] << " <Input-File>" << endl;
    return 0;
  }  
  try {

  string in_file = argv[1];
  Input_t input;
  ReadInput(in_file, input);

  uint64_t Norbs = getParam<int>( input, "norbs" );
  uint64_t Nups  = getParam<int>( input, "nups"  );
  uint64_t Ndos  = getParam<int>( input, "ndos"  );
  uint64_t Norbseff  = getParam<int>( input, "norbseff"  );
  bool print = true;
  string fcidump = getParam<string>( input, "fcidump_file" );

  if(Norbseff < Nups) Norbseff = Nups;
  if(Norbseff < Ndos) Norbseff = Ndos;
  cout << "Using effective norbs space " << Norbseff << endl; 

  intgrls::integrals ints(Norbs, fcidump);

  FermionHamil Hop(ints);
  #if 0
  //Lets test hartree-fock
  uint64_t u1 = 37793167;
  uint64_t d1 = 37793167;
  u1 = (1 << Nups)-1;
  d1 = (1 << Ndos)-1;
  uint64_t st =   (d1 << Norbs) + u1;
  slater_det hello =  slater_det( st, Norbs, Nups, Ndos ) ;
  double nE =  Hop.GetHmatel(hello,hello);
  cout << std::setprecision(16) << "E0 = " << nE + ints.core_energy << endl;
  //exit(0);

  //SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );
  #endif



  // Build configuration space
  SetSlaterDets stts = BuildShiftHilbertSpace( Norbs, Norbseff, Nups, Ndos );
  const size_t ndets = stts.size();
  if(!world_rank) std::cout << "NDETS = " << ndets << std::endl;
  

#if 0
  // Form the Hamiltonian in distributed memory
  MPI_Barrier(MPI_COMM_WORLD);
  auto dist_h_start = clock_type::now();

  auto H_dist = make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD,
    stts.begin(), stts.end(), Hop, ints, 1e-9 );

  MPI_Barrier(MPI_COMM_WORLD);
  auto dist_h_end = clock_type::now();

  duration_type dist_h_dur = dist_h_end - dist_h_start;
  if(!world_rank)  
    std::cout << "Distributed H Construction took " << dist_h_dur.count() 
              << " ms" << std::endl;

  const bool do_serial_check = false;
  if( do_serial_check ) {
    auto H_dist_ref = make_dist_csr_hamiltonian_bcast<int32_t>(
      MPI_COMM_WORLD, stts.begin(), stts.end(), Hop, ints, 1e-9 );

    std::cout << std::boolalpha << 
      ( H_dist.diagonal_tile() == H_dist_ref.diagonal_tile() ) << std::endl;
    std::cout << std::boolalpha << 
      ( H_dist.off_diagonal_tile() == H_dist_ref.off_diagonal_tile() ) << std::endl;
  }
#endif


#if 0
  //std::minstd_rand0 gen;
  //std::uniform_real_distribution<> dist(-1,1);

  // Generate V (replicated)
  std::vector<double> V(ndets);
  //std::generate( V.begin(), V.end(), [&](){ return dist(gen); } );
  std::fill( V.begin(), V.end(), 1./ndets );
  //std::cout << "LOCAL NNZ = " << H_dist.nnz() << std::endl;
  size_t local_mf = H_dist.mem_footprint();
  std::cout << "LOCAL MF = " << (local_mf/(1024.*1024.*1024.)) << " GB" << std::endl;


  // Copy local parts of V into "distributed" V
  auto [local_rs, local_re] = H_dist.row_bounds(world_rank);
  const auto local_extent = H_dist.local_row_extent();
  std::vector<double> V_dist( local_extent ), AV_dist( local_extent );
  for( auto i = local_rs; i < local_re; ++i ) {
    V_dist[i - local_rs] = V[i];
  }

  // Generate SPMV info
  auto spmv_info = sparsexx::spblas::generate_spmv_comm_info( H_dist );

  // Do matvec
  MPI_Barrier(MPI_COMM_WORLD);
  auto pspmbv_start = clock_type::now();

  sparsexx::spblas::pgespmv( 1., H_dist, V_dist.data(), 
    0., AV_dist.data(), spmv_info );

  MPI_Barrier(MPI_COMM_WORLD);
  auto pspmbv_end = clock_type::now();
  duration_type pspmv_dur = pspmbv_end - pspmbv_start;

  if(!world_rank) std::cout << "PSPMV took " << pspmv_dur.count() << " ms" 
    << std::endl;
#endif


  #if 0
  auto H = make_csr_hamiltonian( stts, Hop, ints, 1e-9 );
  {
  std::ofstream H_file( "H_cr2_" + std::to_string(ndets) + ".bin", std::ios::binary );
  size_t nnz = H.nnz();
  H_file.write( (const char*)&ndets, sizeof(size_t) );
  H_file.write( (const char*)&nnz,   sizeof(size_t) );
  H_file.write( (const char*)H.rowptr().data(), (ndets+1) * sizeof(int32_t) );
  H_file.write( (const char*)H.colind().data(), nnz * sizeof(int32_t) );
  H_file.write( (const char*)H.nzval().data(),  nnz * sizeof(double) );
  }

  lobpcgxx::operator_action_type<double> HamOp = 
    [&]( int64_t n , int64_t k , const double* x , int64_t ldx ,
         double* y , int64_t ldy ) -> void {

      sparsexx::spblas::gespmbv( k, 1., H, x, ldx, 0., y, ldy );

    };

  lobpcgxx::lobpcg_settings settings;
  settings.conv_tol = 1e-6;
  settings.maxiter  = 2000;
  settings.print_iter = true;
  lobpcgxx::lobpcg_operator<double> lob_op( HamOp );

  int64_t K = 4;
  int64_t N = H.n();
  std::vector<double> X0( N * K );

  // Random vectors 
  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&](){ return dist(gen); };
  std::generate( X0.begin(), X0.end(), rand_gen );
  lobpcgxx::cholqr( N, K, X0.data(), N ); // Orthogonalize

  std::vector<double> lam(K), res(K);
  lobpcgxx::lobpcg( settings, N, K, K, lob_op, lam.data(), X0.data(), N,
    res.data() );

  auto E0 = lam[0];
  auto psi0 = Eigen::Map<Eigen::VectorXd>( X0.data(), N );

  cout << std::scientific << std::setprecision(16);
  cout << "Ground state energy: " << E0 + ints.core_energy << endl;


  #else

  auto start = clock_type::now(); 
  #if 1
  auto H = make_csr_hamiltonian( stts, Hop, ints, 1e-9 );
  #else
  auto H = sparsexx::read_mm< sparsexx::csr_matrix<double,int> >(
    "/global/cfs/cdirs/m1027/dbwy/ASCI-CI/external/sparsexx/SiNa/SiNa.mtx" 
    //"/global/cfs/cdirs/m1027/dbwy/ASCI-CI/external/sparsexx/Ga41As41H72/Ga41As41H72.mtx"
  );
  #endif
  duration_type H_dur = clock_type::now() - start;
  std::cout << "H duration = " << H_dur.count() << std::endl;

  auto N = H.m();
  std::cout << "NNZ = " << H.nnz() << std::endl;
  #if 0
  std::vector<double> H_dense(N*N);
  sparsexx::convert_to_dense( H, H_dense.data(), N );
  double max_diff = 0.;
  for( auto i = 0; i < N; ++i )
  for( auto j = i; j < N; ++j ) {
    max_diff = std::max( max_diff, std::abs(H_dense[i+j*N] - H_dense[j+i*N]) );
  }
  std::cout << "MAX DIFF = " << max_diff << std::endl;

  //for( auto i = 0; i < N; ++i ) {
  //  double sum = 0.;
  //  for( auto j = 0; j < N; ++j ) sum += std::abs(H_dense[j + i*N]);
  //  auto diag = H_dense[i*(N+1)];
  //  sum -= std::abs(diag);
  //  std::cout << i << ", " << std::abs(diag) << ", " << sum << std::endl;
  //}
  #endif

  // Davidson

  const size_t max_m = 500;
  davidson(max_m, H, 1e-8);

  #endif

  }
  catch(const char *s)
  {
    cout << "Exception occurred!! Code: " << s << endl;
  }
  catch(string s)
  {
    cout << "Exception occurred!! Code: " << s << endl;
  }

  MPI_Finalize();
  return 0;
}

