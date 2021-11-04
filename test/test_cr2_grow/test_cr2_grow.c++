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

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;


using namespace std;
using namespace cmz::ed;

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

#if 0
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
#else

#if 0
  index_t i = 0;
  for( auto bra : stts ) {
    size_t nnz_col = 0;

    index_t j = 0;
    for( auto ket : stts ) {
      if( std::popcount( bra.GetState() ^ ket.GetState() ) <= 4 ) {
        const auto h_el = Hop.GetHmatel( bra, ket );
        if( std::abs(h_el) > H_thresh ) {
          nnz_col++;
          colind.emplace_back(j);
          nzval.emplace_back( h_el );
        }
      }
      j++;
    }

    rowptr.push_back( rowptr.back() + nnz_col );
    i++;
  }
#else

  const double res_fraction = 0.20;

  std::vector<slater_det> stts_vec( stts.begin(), stts.end() );
  std::vector< std::vector<index_t> > colind_by_row( ndets );
  std::vector< std::vector<double>  > nzval_by_row ( ndets );
  for( auto& v : colind_by_row ) v.reserve( ndets * res_fraction );
  for( auto& v : nzval_by_row )  v.reserve( ndets * res_fraction );

  #pragma omp parallel for
  for( index_t i = 0; i < ndets; ++i ) {
    for( index_t j = 0; j < ndets; ++j ) 
    if( std::popcount( stts_vec[i].GetState() ^ stts_vec[j].GetState() <= 4 ) ) {
      const auto h_el = Hop.GetHmatel( stts_vec[i], stts_vec[j] );
      if( std::abs(h_el) > H_thresh ) {
        colind_by_row[i].emplace_back(j);
        nzval_by_row[i].emplace_back( h_el );
      }
    }
  }

  std::vector<size_t> row_counts( ndets );
  std::transform( colind_by_row.begin(), colind_by_row.end(), row_counts.begin(),
    [](const auto& v){ return v.size(); } );
  const size_t _nnz = std::accumulate( row_counts.begin(), row_counts.end(), 0ul );

  std::exclusive_scan( row_counts.begin(), row_counts.end(), rowptr.begin(), 0);
  rowptr[ndets] = rowptr[ndets-1] + row_counts[ndets-1];

  colind.reserve( _nnz );
  nzval .reserve( _nnz );
  for( auto& v : colind_by_row ) colind.insert(colind.end(), v.begin(), v.end());
  for( auto& v : nzval_by_row )  nzval .insert(nzval.end(),  v.begin(), v.end());

#endif
#endif


  // Move resources into CSR matrix
  const auto nnz = colind.size();
  sparsexx::csr_matrix<double, index_t> H( ndets, ndets, nnz, 0 );
  H.colind() = std::move(colind);
  H.rowptr() = std::move(rowptr);
  H.nzval()  = std::move(nzval);

  return H;

}


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




    SetSlaterDets stts = BuildShiftHilbertSpace( Norbs, Norbseff, Nups, Ndos );

    cout << "Building Hamiltonian matrix (old)" << endl;
    auto old_hmat_st = clock_type::now();
    //SpMatD Hmat = GetHmat( &Hop, stts, print );
    auto old_hmat_en = clock_type::now();
    std::chrono::duration<double,std::milli> old_hmat_dur = 
      old_hmat_en - old_hmat_st;

    cout << "Building Hamiltonian matrix (new)" << endl;
    auto new_hmat_st = clock_type::now();
    #if 1
    auto Hmat_csr = make_csr_hamiltonian( stts, Hop, ints, 1e-09 );
    #else
    sparsexx::csr_matrix<double,int32_t> 
      Hmat_csr( Hmat.rows(), Hmat.cols(), Hmat.nonZeros(), 0 );
    Hmat_csr.rowptr()[0] = 0;
    for( auto i = 0, inz = 0; i < Hmat.rows(); ++i ) {
      size_t nnz_col = 0;
      for( auto j = 0; j < Hmat.cols(); ++j ) 
      if( std::abs(Hmat.coeff(i,j)) > 1e-9 ) { 
        Hmat_csr.colind()[inz] = j;
        Hmat_csr.nzval() [inz] = Hmat.coeff(i,j);
        nnz_col++; inz++;
      }
      Hmat_csr.rowptr()[i+1] = Hmat_csr.rowptr()[i] + nnz_col; 
    }
    #endif
    auto new_hmat_en = clock_type::now();
    std::chrono::duration<double,std::milli> new_hmat_dur = 
      new_hmat_en - new_hmat_st;

    std::cout << "HAM N = " << Hmat_csr.n() << std::endl;
    std::cout << "NEW HMAT NNZ = " << Hmat_csr.nnz() << std::endl;

    std::cout << "Durations " << old_hmat_dur.count() << ", " 
              << new_hmat_dur.count() << std::endl;
    

    // Hamiltonian graph partitioning
    if(1){
      int npart = 4;
      auto kway_part_begin = clock_type::now();
      auto part = sparsexx::kway_partition( npart, Hmat_csr );
      auto kway_part_end = clock_type::now();

      std::vector<int32_t> mat_perm;
      std::tie( mat_perm, std::ignore ) = sparsexx::perm_from_part( npart, part );

      auto permute_begin = clock_type::now();
      Hmat_csr = sparsexx::permute_rows_cols( Hmat_csr, mat_perm, mat_perm );
      auto permute_end = clock_type::now();

      duration_type kway_part_dur = kway_part_end - kway_part_begin;
      duration_type permute_dur   = permute_end - permute_begin;

      std::cout << "KWAY PART DUR = " << kway_part_dur.count() << std::endl;
      std::cout << "PERMUTE DUR   = " << permute_dur.count() << std::endl;
    }


    //return 0;

    cout << "Computing Ground State..." << endl;

    double E0;
    VectorXd psi0;


#if 0
    SpMatDOp Hwrap( Hmat );
    GetGS( Hwrap, E0, psi0, input );
#else
    lobpcgxx::operator_action_type<double> HamOp = 
      [&]( int64_t n , int64_t k , const double* x , int64_t ldx ,
           double* y , int64_t ldy ) -> void {

        #if 0
        Eigen::Map<const Eigen::MatrixXd> xmap(x,ldx,k); 
        Eigen::Map<Eigen::MatrixXd>       ymap(y,ldy,k);
        ymap.block(0,0,n,k).noalias() = Hmat * xmap;
        #else
        sparsexx::spblas::gespmbv( k, 1., Hmat_csr, x, ldx, 0., y, ldy );
        #endif

      };
    lobpcgxx::lobpcg_settings settings;
    settings.conv_tol = 1e-6;
    settings.maxiter  = 2000;
    settings.print_iter = true;
    lobpcgxx::lobpcg_operator<double> lob_op( HamOp );

    int64_t K = 4;
    int64_t N = Hmat_csr.n();
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
#endif

    cout << std::setprecision(16);
    cout << "Ground state energy: " << E0 + ints.core_energy << endl;

#if 0
    cout << "Building rdms!!" << endl;

    rdm::rdms rdms( Norbs, psi0, stts );

    cout << "Testing energy with rdms..." << endl;
   
    double E0_rdm = MyInnProd( ints.t, rdms.rdm1 ) + MyInnProd( ints.u, rdms.rdm2 ) + ints.get_core_energy();

    cout << "E0 = " << E0 + ints.core_energy << ", E0_rdm = " << E0_rdm << endl;
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
  return 0;
}

