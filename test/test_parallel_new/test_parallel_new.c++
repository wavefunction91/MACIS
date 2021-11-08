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
#include "csr_hamiltonian.hpp"

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;

#include <random>

using namespace std;
using namespace cmz::ed;



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


  #if 0
  auto H = make_csr_hamiltonian( stts, Hop, ints, 1e-9 );

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

