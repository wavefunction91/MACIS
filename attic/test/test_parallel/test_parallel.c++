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
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/util/submatrix.hpp>
#include <random>

#include <mpi.h>

using namespace std;
using namespace cmz::ed;


template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian(
  const SetSlaterDets& stts,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh = 1e-9
);

template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  SetSlaterDets::iterator bra_begin,
  SetSlaterDets::iterator bra_end,
  SetSlaterDets::iterator ket_begin,
  SetSlaterDets::iterator ket_end,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh = 1e-9
);


int main( int argn, char* argv[] )
{
  MPI_Init(NULL,NULL);

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
    bool print = true;
    string fcidump = getParam<string>( input, "fcidump_file" );

    if( Norbs > 16 )
      throw( "cmz::ed is not ready for more than 16 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );

    intgrls::integrals ints(Norbs, fcidump);

    FermionHamil Hop(ints);

    // MPI World info
    int world_rank, world_size;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    //world_rank = 0;
    //world_size = 1;

    auto now = [](){ return std::chrono::high_resolution_clock::now(); };
    using duration = std::chrono::duration<double,std::milli>;

    using index_t = int32_t;
    sparsexx::csr_matrix<double,index_t> Hcsr;

    // New Hamiltonian build
    if(world_rank == 0 ) {
      cout << "Building Hamiltonian matrix (new)" << endl;
      std::cout << "NDETS = " << stts.size() << std::endl;
      auto hbuild_new_st = now();
      Hcsr = make_csr_hamiltonian<int32_t>( stts, Hop, ints, 1e-9 );
      auto hbuild_new_en = now();
      std::cout << "N = " << Hcsr.m() << std::endl;
      std::cout << "NNZ = " << Hcsr.nnz() << std::endl;
      std::cout << "H Build New Duration = " << 
        duration( hbuild_new_en - hbuild_new_st).count() << " ms" << std::endl;
    }


    // Form 1D block-row tiling
    const auto ndet = stts.size();
    int64_t nrow_per_rank = ndet / world_size;
    std::vector<index_t> row_tiling( world_size + 1 );
    for( auto i = 0; i < world_size; ++i ) row_tiling[i] = i * nrow_per_rank;
    row_tiling.back() = ndet;

    std::vector<index_t> col_tiling = {0, ndet};


    sparsexx::dist_csr_matrix<double,index_t> 
      dist_H( MPI_COMM_WORLD, ndet, ndet, row_tiling, col_tiling );

    for( auto& [tile_index, local_tile] : dist_H ) {

      local_tile.local_matrix = make_csr_hamiltonian_block<int32_t>(
        std::next(stts.begin(), local_tile.global_row_extent.first),
        std::next(stts.begin(), local_tile.global_row_extent.second),
        std::next(stts.begin(), local_tile.global_col_extent.first),
        std::next(stts.begin(), local_tile.global_col_extent.second),
        Hop, ints, 1e-9
      );
    
      //std::cout << world_rank << "; " << 
      //  local_tile.global_row_extent.first << ", " <<
      //  local_tile.global_row_extent.second << std::endl;
    }

    if( world_size == 1 ) {
      auto& H = dist_H.begin()->second.local_matrix;

      cout << std::boolalpha << (H.colind() == Hcsr.colind()) << ", "
           << (H.rowptr() == Hcsr.rowptr()) << ", "
           << (H.nzval() == Hcsr.nzval()) << std::endl;
    }


    size_t K = 1;
    std::vector<double> V( ndet * K );


    // Random vectors 
    std::default_random_engine gen;
    std::normal_distribution<> dist(0., 1.);
    auto rand_gen = [&](){ return dist(gen); };
    std::generate( V.begin(), V.end(), rand_gen );
    MPI_Bcast( V.data(), V.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD );

    std::vector<double> AV(ndet * K), AV_dist(ndet * K);

    // Serial SPMBV
    if(world_rank == 0) {
      auto spmv_st = now();
      sparsexx::spblas::gespmbv( K, 1., Hcsr, V.data(), ndet, 0., 
        AV.data(), ndet );
      auto spmv_en = now();
      std::cout << "Serial SPMBV = " << duration(spmv_en-spmv_st).count() 
        << " ms" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Distributed SPMBV
    {
      auto spmv_st = now();
      #if 0
      sparsexx::spblas::pgespmbv_grv( K, 1., dist_H, V.data(), ndet, 0., 
        AV_dist.data(), ndet );
      #else

      std::vector<int> sendcounts(world_size), 
        senddisp(row_tiling.begin(), row_tiling.end()-1);
      for( auto i = 0 ; i < world_size; ++i )
        sendcounts[i] = row_tiling[i+1] - row_tiling[i];

      std::vector<double> V_local( sendcounts[world_rank] ),
                          AV_local( sendcounts[world_rank] );

      // Scatter V to V_local
      MPI_Scatterv( V.data(), sendcounts.data(), senddisp.data(),
        MPI_DOUBLE, V_local.data(), sendcounts[world_rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD );

      // Do matvec
      sparsexx::spblas::pgespmv_rdv( 1., dist_H, V_local.data(), 0., 
        AV_local.data() );

      // Gather AV to AV_dist
      MPI_Gatherv( AV_local.data(), sendcounts[world_rank], MPI_DOUBLE,
        AV_dist.data(), sendcounts.data(), senddisp.data(), MPI_DOUBLE, 
        0, MPI_COMM_WORLD );

      #endif
      MPI_Barrier(MPI_COMM_WORLD);
      auto spmv_en = now();
      if( world_rank == 0 )
      std::cout << "Distributed SPMBV = " << duration(spmv_en-spmv_st).count() 
        << " ms" << std::endl;
    }

    if( world_rank == 0 ) {
      for( auto i = 0; i < AV.size(); ++i ) {
        AV[i] = std::abs(AV[i] - AV_dist[i]);
      }
      std::cout << "MAX DIFF = " << *std::max_element(AV.begin(),AV.end()) 
        << std::endl;
    }



#if 0
    // Old Hamiltonian build
    cout << "Building Hamiltonian matrix (old)" << endl;
    auto hbuild_old_st = now();
    SpMatD Hmat = GetHmat( &Hop, stts, print );
    auto hbuild_old_en = now();
    std::cout << "H Build Old Duration = " << duration( hbuild_old_en - hbuild_old_st).count() << " ms" <<  std::endl;
#endif


#if 0 // Eigensolver
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


  //MPI_Finalize();
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
#endif


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




template <typename index_t = int32_t>
sparsexx::csr_matrix<double,index_t> make_csr_hamiltonian_block(
  SetSlaterDets::iterator bra_begin,
  SetSlaterDets::iterator bra_end,
  SetSlaterDets::iterator ket_begin,
  SetSlaterDets::iterator ket_end,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh 
) {

  // Form CSR adjacency
  std::vector<index_t> colind, rowptr;
  std::vector<double>  nzval;

  const auto ndets_bra = std::distance(bra_begin,bra_end);
  const auto ndets_ket = std::distance(ket_begin,ket_end);
  rowptr.reserve( ndets_bra + 1 );
  rowptr.push_back(0);

#if 0
  index_t i = 0;
  for( auto bra_it = bra_begin; bra_it != bra_end; ++bra_it ) {
    
    const auto& bra_det = *bra_it;
    // Form all single/double excitations from current det
    // XXX: This is proabably too much work, no?
    auto sd_exes = bra_det.GetSinglesAndDoubles( &ints );
    const auto nsd_det = sd_exes.size();

    // Initialize memory for adjacency row
    std::vector<index_t> colind_local; colind_local.reserve( nsd_det + 1 );
    std::vector<double>  nzval_local;  nzval_local .reserve( nsd_det + 1 );

    // Initialize adjacency row with diagonal element if contained in bra
    if( auto it = std::find( ket_begin, ket_end, bra_det ); it != ket_end ) {
      colind_local.push_back(std::distance(ket_begin,it));
      nzval_local .push_back( Hop.GetHmatel( bra_det, bra_det ) );
    }
    ++i; // Increment row index

    // Loop over singles and doubles
    for( const auto& ex_det : sd_exes ) {
      // Attempt to locate excited determinant in full determinant list
      auto it = std::find( ket_begin, ket_end, ex_det );

      // If ex_det in list and ( det | H | ex_det) > thresh, append to adjacency
      if( it != ket_end ) {
        const auto& ket_det = *it;
        const auto h_el = Hop.GetHmatel(bra_det, ket_det);
        if( std::abs( h_el ) > H_thresh ) {
          colind_local.push_back( std::distance( ket_begin, it ) );
          nzval_local .push_back( h_el );
        }
      }
    } // End loop over excited determinants


    // TODO add diagonal element if not present?

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

  index_t i = 0;
  for( auto bra_it = bra_begin; bra_it != bra_end; ++bra_it ) {
    const auto& bra = *bra_it;
    size_t nnz_col = 0;
    index_t j = 0;
    for( auto ket_it = ket_begin; ket_it != ket_end; ++ket_it ) {
      const auto& ket = *ket_it;
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

#endif

  // Move resources into CSR matrix
  const auto nnz = colind.size();
  sparsexx::csr_matrix<double, index_t> H( ndets_bra, ndets_ket, nnz, 0 );
  H.colind() = std::move(colind);
  H.rowptr() = std::move(rowptr);
  H.nzval()  = std::move(nzval);

  return H;

}

template
sparsexx::csr_matrix<double,int32_t> make_csr_hamiltonian_block<int32_t>(
  SetSlaterDets::iterator bra_begin,
  SetSlaterDets::iterator bra_end,
  SetSlaterDets::iterator ket_begin,
  SetSlaterDets::iterator ket_end,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh 
);


template
sparsexx::csr_matrix<double,int64_t> make_csr_hamiltonian_block<int64_t>(
  SetSlaterDets::iterator bra_begin,
  SetSlaterDets::iterator bra_end,
  SetSlaterDets::iterator ket_begin,
  SetSlaterDets::iterator ket_end,
  const FermionHamil&  Hop,
  const intgrls::integrals& ints,
  const double H_thresh 
);
