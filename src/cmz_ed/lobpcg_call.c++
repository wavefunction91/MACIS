#include "cmz_ed/lobpcg_call.h++"
#include <random>

namespace cmz
{
  namespace ed
  {

    void LobpcgGS(
      const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> > &H, 
      size_t dimH,
      size_t nstates,
      std::vector<double> &evals,
      std::vector<double> &X,
      int maxIts, 
      double tol,
      bool print)
    {
      // Run LOBPCG to get the first few eigenstates of a given
      // Hamiltonian
      evals.clear(); evals.resize( nstates, 0. );
      X.clear(); X.resize( dimH * nstates, 0. );
      auto spmv_info = sparsexx::spblas::generate_spmv_comm_info( H );
      // Hop
      lobpcgxx::operator_action_type<double> Hop = 
        [&]( int64_t n , int64_t k , const double* x , int64_t ldx ,
             double* y , int64_t ldy ) -> void {

          for( int ik = 0; ik < k; ik++)
            sparsexx::spblas::pgespmv( 1., H, x+ik*n, 
                                       0., y+ik*n, spmv_info );
        };

      // Random vectors 
      std::default_random_engine gen;
      std::normal_distribution<> dist(0., 1.);
      auto rand_gen = [&](){ return dist(gen); };
      std::generate( X.begin(), X.end(), rand_gen );
      lobpcgxx::cholqr( dimH, nstates, X.data(), dimH ); // Orthogonalize

      lobpcgxx::lobpcg_settings settings;
      settings.conv_tol = tol;
      settings.maxiter  = maxIts;
      lobpcgxx::lobpcg_operator<double> lob_op( Hop );
    
    
    
      std::vector<double> res(nstates);

      try
      {
        lobpcgxx::lobpcg( settings, dimH , nstates , nstates, 
                          lob_op, evals.data(), X.data(), dimH, 
	  		  res.data() ); 
      }
      catch( std::runtime_error e )
      {
	std::cout << "Runtime error during lobpcg: " << e.what() << std::endl;
      }
    
      if( print )
      {
        std::cout << std::scientific << std::setprecision(10) << std::endl ;
        for ( int64_t i = 0; i < nstates; ++i ) {
          std::cout << "  evals[" << i << "] = " << evals[i]
                    << ",   res[" << i << "] = " << res[i]
                    << std::endl ;
        }
      }

    }
  }// name ed
}// name cmz
