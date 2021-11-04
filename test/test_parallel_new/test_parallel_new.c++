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
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include "csr_hamiltonian.hpp"

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;


using namespace std;
using namespace cmz::ed;



int main( int argn, char* argv[] )
{

  MPI_Init(NULL,NULL);
  const auto world_rank = sparsexx::detail::get_mpi_rank(MPI_COMM_WORLD);
  const auto world_size = sparsexx::detail::get_mpi_size(MPI_COMM_WORLD);

  if( argn != 2 )
  {
    cout << "Usage: " << argv[0] << " <Input-File>" << endl;
    return 0;
  }  
  try
  {
    string in_file = argv[1];
    Input_t input;
    ReadInput(in_file, input);

    uint64_t Norbs = getParam<int>( input, "norbs" );
    uint64_t Nups  = getParam<int>( input, "nups"  );
    uint64_t Ndos  = getParam<int>( input, "ndos"  );
    uint64_t Norbseff  = getParam<int>( input, "norbseff"  );
    bool print = true;
    string fcidump = getParam<string>( input, "fcidump_file" );

//    if( Norbs > 16 )
//      throw( "cmz::ed is not ready for more than 16 orbitals!" );
 //   if( Nups > Norbs || Ndos > Norbs )
 //     throw( "Nups or Ndos cannot be larger than Norbs!" );
    if(Norbseff < Nups) Norbseff = Nups;
    if(Norbseff < Ndos) Norbseff = Ndos;
    cout << "Using effective norbs space " << Norbseff << endl; 

    intgrls::integrals ints(Norbs, fcidump);

    FermionHamil Hop(ints);
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



    // Build configuration space
    SetSlaterDets stts = BuildShiftHilbertSpace( Norbs, Norbseff, Nups, Ndos );
    

    // Form the Hamiltonian in distributed memory
    MPI_Barrier(MPI_COMM_WORLD);
    auto dist_h_start = clock_type::now();

    auto H_dist = make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD,
      stts.begin(), stts.end(), Hop, ints, 1e-9 );

    MPI_Barrier(MPI_COMM_WORLD);
    auto dist_h_end = clock_type::now();

    duration_type dist_h_dur = dist_h_end - dist_h_start;
    if(!world_rank)  
      std::cout << "Dist H Duration " << dist_h_dur.count() << std::endl;


#if 0 
    // Form the Hamiltonian on the root rank and broadcast
    sparsexx::csr_matrix<double,int32_t> H;
    if(!world_rank) { 
      // Build Hamiltonian on root rank

      cout << "Building Hamiltonian Matrix (Serial)" << endl;
      auto new_hmat_st = clock_type::now();
      H = make_csr_hamiltonian( stts, Hop, ints, 1e-09 );
      auto new_hmat_en = clock_type::now();
      std::chrono::duration<double,std::milli> new_hmat_dur = 
        new_hmat_en - new_hmat_st;

      // Broadcast data
      size_t nnz = H.nnz();
      sparsexx::detail::mpi_bcast( &nnz, 1, 0, MPI_COMM_WORLD );

      sparsexx::detail::mpi_bcast( H.rowptr(), 0, MPI_COMM_WORLD );
      sparsexx::detail::mpi_bcast( H.colind(), 0, MPI_COMM_WORLD );
      sparsexx::detail::mpi_bcast( H.nzval(),  0, MPI_COMM_WORLD );

      std::cout << "HAM N = " << H.n() << std::endl;
      std::cout << "NEW HMAT NNZ = " << H.nnz() << std::endl;

      std::cout << "Serial H Construction Duration " << new_hmat_dur.count() << std::endl;
    } else {
      // Recieve data from root rank
      size_t nnz;
      sparsexx::detail::mpi_bcast( &nnz, 1, 0, MPI_COMM_WORLD );

      size_t n = stts.size();
      H = sparsexx::csr_matrix<double,int32_t>(n,n,nnz,0);

      sparsexx::detail::mpi_bcast( H.rowptr(), 0, MPI_COMM_WORLD );
      sparsexx::detail::mpi_bcast( H.colind(), 0, MPI_COMM_WORLD );
      sparsexx::detail::mpi_bcast( H.nzval(),  0, MPI_COMM_WORLD );
    }

    // Hamiltonian reordering
    const bool do_reorder = false;
    if( do_reorder ) {
      if(world_rank == 0){
        int npart = std::max(2l,world_size);
        auto kway_part_begin = clock_type::now();
        auto part = sparsexx::kway_partition( npart, H );
        auto kway_part_end = clock_type::now();

        std::vector<int32_t> mat_perm;
        std::tie( mat_perm, std::ignore ) = sparsexx::perm_from_part( npart, part );

        auto permute_begin = clock_type::now();
        H = sparsexx::permute_rows_cols( H, mat_perm, mat_perm );
        auto permute_end = clock_type::now();

        duration_type kway_part_dur = kway_part_end - kway_part_begin;
        duration_type permute_dur   = permute_end - permute_begin;

        std::cout << "KWAY PART DUR = " << kway_part_dur.count() << std::endl;
        std::cout << "PERMUTE DUR   = " << permute_dur.count() << std::endl;
      }

      // Broadcast reordered matrix
      if( world_size > 1 ) {
        sparsexx::detail::mpi_bcast( H.rowptr(), 0, MPI_COMM_WORLD );
        sparsexx::detail::mpi_bcast( H.colind(), 0, MPI_COMM_WORLD );
        sparsexx::detail::mpi_bcast( H.nzval(),  0, MPI_COMM_WORLD );
      }
    }

    // Distribute the matrix from replicated data
    sparsexx::dist_sparse_matrix< sparsexx::csr_matrix<double, int32_t> >
      H_dist_ref( MPI_COMM_WORLD, H ); 
#else
  auto H_dist_ref = make_dist_csr_hamiltonian_bcast<int32_t>(
    MPI_COMM_WORLD, stts.begin(), stts.end(), Hop, ints, 1e-9 );
#endif



   auto check_eq = [](const auto& A, const auto& B) {
     return (A.colind() == B.colind()) and
            (A.rowptr() == B.rowptr()) and
            (A.nzval()  == B.nzval() );
   };


   std::cout << std::boolalpha << 
     check_eq( H_dist.diagonal_tile(), H_dist_ref.diagonal_tile() ) << std::endl;
   std::cout << std::boolalpha << 
     check_eq( H_dist.off_diagonal_tile(), H_dist_ref.off_diagonal_tile() ) << std::endl;

   std::cout << H_dist.diagonal_tile().nnz() + H_dist.off_diagonal_tile().nnz() << std::endl;


   #if 0

    cout << "Computing Ground State..." << endl;

    double E0;
    VectorXd psi0;


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

    E0 = lam[0];
    psi0 = Eigen::Map<Eigen::VectorXd>( X0.data(), N );

    std::cout << std::scientific << std::setprecision(5);

    cout << std::setprecision(16);
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

