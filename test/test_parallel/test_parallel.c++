/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests H2O 6-31g active space (8e, 5o) Hamiltonian.
 ******************************************/

#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include<iostream>
#include<fstream>
#include "unsupported/Eigen/SparseExtra"

#include <lobpcgxx/lobpcg.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <random>

using namespace std;
using namespace cmz::ed;


template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const SetSlaterDets& stts,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh = 1e-9
);


int main( int argn, char* argv[] )
{
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

    unsigned short Norbs = getParam<int>( input, "norbs" );
    unsigned short Nups  = getParam<int>( input, "nups"  );
    unsigned short Ndos  = getParam<int>( input, "ndos"  );
    bool print = true;
    string fcidump = getParam<string>( input, "fcidump_file" );

    if( Norbs > 16 )
      throw( "cmz::ed is not ready for more than 16 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );

    intgrls::integrals ints(Norbs, fcidump);

    FermionHamil Hop(ints);


    auto now = [](){ return std::chrono::high_resolution_clock::now(); };
    using duration = std::chrono::duration<double,std::milli>;

    // New code
    cout << "Building Hamiltonian matrix (new)" << endl;
    auto hbuild_new_st = now();
    auto Hcsr = make_csr_hamiltonian<int32_t>( stts, Hop, ints, 1e-9 );
    auto hbuild_new_en = now();

    std::cout << "NNZ = " << Hcsr.nnz() << std::endl;
    std::cout << "H Build New Duration = " << duration( hbuild_new_en - hbuild_new_st).count() << " ms" << std::endl;

    // Old code
    cout << "Building Hamiltonian matrix (old)" << endl;
    auto hbuild_old_st = now();
    SpMatD Hmat = GetHmat( &Hop, stts, print );
    auto hbuild_old_en = now();
    std::cout << "H Build Old Duration = " << duration( hbuild_old_en - hbuild_old_st).count() << " ms" <<  std::endl;


    double E0;
    VectorXd psi0;

    auto eigensolver_st = now();

    lobpcgxx::operator_action_type<double> HamOp = 
      [&]( int64_t n , int64_t k , const double* x , int64_t ldx ,
           double* y , int64_t ldy ) -> void {
        sparsexx::spblas::gespmbv( k, 1., Hcsr, x, ldx, 0., y, ldy );
      };

    lobpcgxx::lobpcg_settings settings;
    settings.conv_tol = 1e-6;
    settings.maxiter  = 2000;
    settings.print_iter = true;
    lobpcgxx::lobpcg_operator<double> lob_op( HamOp );

    int64_t K = 4;
    int64_t N = Hmat.rows();
    std::vector<double> X0( N * K );

    // Random vectors 
    std::default_random_engine gen;
    std::normal_distribution<> dist(0., 1.);
    auto rand_gen = [&](){ return dist(gen); };
    std::generate( X0.begin(), X0.end(), rand_gen );
    lobpcgxx::cholqr( N, K, X0.data(), N ); // Orthogonalize

    std::vector<double> lam(K), res(K);
    lobpcgxx::lobpcg( settings, N, K, 1, lob_op, lam.data(), X0.data(), N,
      res.data() );

    E0 = lam[0];
    psi0 = Eigen::Map<Eigen::VectorXd>( X0.data(), N );

    auto eigensolver_en = now();
    cout << "Eigensolver Duration = " << duration(eigensolver_en-eigensolver_st).count() << " ms" << std::endl;
    std::cout << std::scientific << std::setprecision(5);
    cout << "Ground state energy: " << E0 + ints.core_energy << endl;

 
  }
  catch(const char *s)
  {
    cout << "Exception occurred!! Code: " << s << endl;
  }
  catch(string s)
  {
    cout << "Exception occurred!! Code: " << s << endl;
  }
  return 0;
}





template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const SetSlaterDets& stts,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh
) {


  // Form CSR adjacency
  std::vector<index_t> colind, rowptr;
  std::vector<double>  nzval;

  const auto ndets = stts.size();
  rowptr.reserve( ndets + 1 );
  rowptr.push_back(0);

  index_t i = 0;
  for( const auto& det : stts ) {

    // Form all single/double excitations from current det
    // XXX: This is proabably too much work, no?
    auto sd_exes = det.GetSinglesAndDoubles( &ints );
    const auto nsd_det = sd_exes.size();

    // Initialize memory for adjacency row
    std::vector<index_t> colind_local; colind_local.reserve( nsd_det + 1 );
    std::vector<double>  nzval_local;  nzval_local .reserve( nsd_det + 1 );

    // Initialize adjacency row with diagonal element
    colind_local.push_back(i++);
    nzval_local .push_back( Hop.GetHmatel( det, det ) );

    // Loop over singles and doubles
    for( const auto& ex_det : sd_exes ) {
      // Attempt to locate excited determinant in full determinant list
      auto it = stts.find( ex_det );

      // If ex_det in list and ( det | H | ex_det) > thresh, append to adjacency
      if( it != stts.end() ) {
        const auto h_el = Hop.GetHmatel(det, *it);
        if( std::abs( h_el ) > H_thresh ) {
          colind_local.push_back( std::distance( stts.begin(), it ) );
          nzval_local .push_back( h_el );
        }
      }
    } // End loop over excited determinants

    // Sort column indicies selected for adjacency row
    const size_t nnz_col = colind_local.size();
    std::vector<index_t> idx( nnz_col );
    std::iota( idx.begin(), idx.end(), 0 );
    std::sort( idx.begin(), idx.end(),
      [&]( auto _i, auto _j ) { return colind_local[_i] < colind_local[_j]; }
    );

    // Place into permanent storage 
    for( auto j = 0; j < nnz_col; ++j ) {
      colind.push_back( colind_local[idx[j]] );
      nzval. push_back( nzval_local[idx[j]]  );
    }

    // Update next rowptr
    rowptr.push_back( rowptr.back() + nnz_col );

  } // End loop over all determinants 


  // Move resources into CSR matrix
  const auto nnz = colind.size();
  sparsexx::csr_matrix<double, index_t> H( ndets, ndets, nnz, 0 );
  H.colind() = std::move(colind);
  H.rowptr() = std::move(rowptr);
  H.nzval()  = std::move(nzval);

  return H;

}

template 
sparsexx::csr_matrix<double,int32_t> make_csr_hamiltonian<int32_t>(
  const SetSlaterDets&, const FermionHamil&, const intgrls::integrals&,
  const double H_thresh
);

template 
sparsexx::csr_matrix<double,int64_t> make_csr_hamiltonian<int64_t>(
  const SetSlaterDets&, const FermionHamil&, const intgrls::integrals&,
  const double H_thresh
);


