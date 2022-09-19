#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>

#include <lobpcgxx/lobpcg.hpp>
#include "cmz_ed/lobpcg_call.h++"
#include "dbwy/asci_body.hpp"

template <typename T>
int test() {

  const int64_t n = 1000 ;
  const int64_t k = 20 ;

  // matrix A -> 2D Laplacian
  std::vector<T> A( n*n, 0. );
  A[0] = 2.0 ;
  for ( int64_t i = 1; i < n; ++i ) {
    A[ (i-1)*n + i ] = -1.0 ;
    A[ i*n + i ] = 2.0 ;
    A[ i*n + (i-1) ] = -1.0 ;
  }



  // Aop
  lobpcgxx::operator_action_type<T> Aop = 
    [&]( int64_t n , int64_t k , const T* x , int64_t ldx ,
         T* y , int64_t ldy ) -> void {

      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                  n , k , n , T(1.0) , A.data() , n , x , ldx , T(0.0) , y , ldy ) ;
    };


  // starting vectors
  std::vector<T> X( n * k );

  // Structured vectors
  //for ( int64_t i = 0; i < n/2; ++i ) {
  //  X[2*i] = 1.0 / std::sqrt(n/2) ;
  //  X[2*i+n+1] = 1.0 / std::sqrt(n/2) ;
  //}

  // Random vectors 
  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&](){ return dist(gen); };
  std::generate( X.begin(), X.end(), rand_gen );
  lobpcgxx::cholqr( n, k, X.data(), n ); // Orthogonalize



  lobpcgxx::lobpcg_settings settings;
  settings.conv_tol = 1e-6;
  settings.maxiter  = 2000;
  lobpcgxx::lobpcg_operator<T> lob_op( Aop );



  std::vector<T> lam(k), res(k);
  lobpcgxx::lobpcg( settings, n , k , k, lob_op, lam.data(), X.data() , n , res.data() ); 

  std::cout << std::scientific << std::setprecision(10) << std::endl ;
  for ( int64_t i = 0; i < k; ++i ) {
    std::cout << "  lam[" << i << "] = " << lam[i]
              << ", res[" << i << "] = " << res[i]
              << std::endl ;
  }

  return 0 ;

}

template <typename T>
int test2() {

  const int64_t n = 4 ;
  const int64_t k = 4 ;

  // matrix A -> 2D Laplacian
  std::vector<T> A( n*n, 0. );
  A[1 + 0 * n] = -1.;
  A[2 + 0 * n] = -1.;
  A[0 + 1 * n] = -1.;
  A[3 + 1 * n] = -1.;
  A[0 + 2 * n] = -1.;
  A[3 + 2 * n] = -1.;
  A[1 + 3 * n] = -1.;
  A[2 + 3 * n] = -1.;

  // Aop
  lobpcgxx::operator_action_type<T> Aop = 
    [&]( int64_t n , int64_t k , const T* x , int64_t ldx ,
         T* y , int64_t ldy ) -> void {

      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                  n , k , n , T(1.0) , A.data() , n , x , ldx , T(0.0) , y , ldy ) ;
    };


  // starting vectors
  std::vector<T> X( n * k );

  // Structured vectors
  //for ( int64_t i = 0; i < n/2; ++i ) {
  //  X[2*i] = 1.0 / std::sqrt(n/2) ;
  //  X[2*i+n+1] = 1.0 / std::sqrt(n/2) ;
  //}

  // Random vectors 
  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&](){ return dist(gen); };
  std::generate( X.begin(), X.end(), rand_gen );
  lobpcgxx::cholqr( n, k, X.data(), n ); // Orthogonalize


  lobpcgxx::lobpcg_settings settings;
  settings.conv_tol = 1e-6;
  settings.maxiter  = 2000;
  lobpcgxx::lobpcg_operator<T> lob_op( Aop );



  std::vector<T> lam(k), res(k);
  lobpcgxx::lobpcg( settings, n , k , k, lob_op, lam.data(), X.data() , n , res.data() ); 

  std::cout << std::scientific << std::setprecision(10) << std::endl ;
  for ( int64_t i = 0; i < k; ++i ) {
    std::cout << "  lam[" << i << "] = " << lam[i]
              << ", res[" << i << "] = " << res[i]
              << std::endl ;
  }

  return 0 ;

}


int test3( std::string in_file ) {

  // Read Input
  constexpr size_t nbits = 16;
  cmz::ed::Input_t input;
  cmz::ed::ReadInput( in_file, input );
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  if( norb > nbits / 2 )
    throw std::runtime_error("Asked for too many orbitals in test3. Accepts up to 8 spatial orbitals.");
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );
  size_t nstates = cmz::ed::getParam<int>( input, "nstates" );
  int nLobpcgIts = cmz::ed::getParam<int>( input, "maxLobpcgIts" );

  std::vector<double> evals;
  std::vector<double> X;
  size_t dimH = 0;
  // Hamiltonian Matrix Element Generator
  MPI_Init(NULL,NULL);
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  if(world_size != 1) 
    throw "NO MPI"; // Disable MPI for now
  { // MPI Scope
    // Read in the integrals 
    cmz::ed::intgrls::integrals ints(norb, fcidump);
    dbwy::DoubleLoopHamiltonianGenerator<nbits> 
      ham_gen( norb, ints.u.data(), ints.t.data() );
    auto dets = dbwy::generate_full_hilbert_space<nbits>( norb, nalpha, nbeta );
    dimH = dets.size();
    auto mat  = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, 1.E-10 );
    cmz::ed::LobpcgGS( mat, dimH, nstates, evals, X, nLobpcgIts );
    

    MPI_Finalize();
  }// MPI scope
 
  cout << "From LOBPCG to get the first 2 eigenstates, with 2 bands" << endl;
  for( int i = 0; i < evals.size(); i++ )
  {
    cout << "  * E[" << i << "] = " << evals[i] << endl;
    cout << "  * v[" << i << "] = ";
    for( int j = 0; j < dimH; j++ )
      cout << X[j + i * dimH] << ", ";
    cout << endl << endl;
  }
  return 0 ;
}

int main( int argn, char *argv[] ) {

  if( argn != 2 )
  {
    std::cout << "Usage: " << argv[0] << " <input-file>" << std::endl;
    return 1 ;
  }

  // double
  std::cout << "\n+++ TEST for double +++\n\n" ;
  test< double >() ;
  try
  {
    test2< double >() ;
  }
  catch( std::runtime_error e )
  {
    std::cout << "Runtime error in test2: " << e.what() << std::endl;
  }
  try
  {
    test3( argv[1] ) ;
  }
  catch( std::runtime_error e )
  {
    std::cout << "Runtime error in test3: " << e.what() << std::endl;
  }
  catch( std::exception e )
  {
    std::cout << "Caught std::exception!" << std::endl;
    std::cout << e.what() << std::endl;
  }
  catch( std::string s )
  {
    std::cout << "Caught std::string!" << std::endl;
    std::cout << s << std::endl;
  }
  catch( char *c )
  {
    std::cout << "Caught char*!" << std::endl;
    std::cout << c << std::endl;
  }
  catch( char const *c )
  {
    std::cout << "Caught char const*!" << std::endl;
    std::cout << c << std::endl;
  }

  // float
  //std::cout << "\n+++ TEST for float +++\n\n" ;
  //test< float >() ;

  // successful
  std::cout << ">> end test lobpcg <<" << std::endl ;

  return 0 ;

}

