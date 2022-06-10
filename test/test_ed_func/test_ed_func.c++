#include <mpi.h>
#include <iostream>
#include "dbwy/asci_body.hpp"

int main( int argc, char* argv[] ) {


  try
  {

    if( argc != 2 ) {
      std::cout << "Must Specify Input" << std::endl;
      return 1;
    }
    using clock_type = std::chrono::high_resolution_clock;
    using duration_type = std::chrono::duration<double>;
    // Read Input
    std::string in_file = argv[1];
    cmz::ed::Input_t input;
    cmz::ed::ReadInput( in_file, input );
    size_t norb = cmz::ed::getParam<int>( input, "norbs" );
    MPI_Init(NULL,NULL);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    if(world_size != 1) 
      throw "NO MPI"; // Disable MPI for now
    { // MPI Scope
  
      double E0_ed, E0_asci;
      std::vector<double> ordm_ed, ordm_asci;
      auto ed_st = clock_type::now();
      if( norb < 16 )
        std::tie( E0_ed, ordm_ed ) = dbwy::run_ed_w_1rdm<32>( input );
      else if( norb < 32 )
        std::tie( E0_ed, ordm_ed ) = dbwy::run_ed_w_1rdm<64>( input );
      else
        std::tie( E0_ed, ordm_ed ) = dbwy::run_ed_w_1rdm<128>( input );
      auto ed_en = clock_type::now();

      if(world_rank == 0) 
      {
        std::cout << "After running ED: E0 = " << E0_ed << std::endl;
        std::cout << "Took: " << duration_type( ed_en - ed_st ).count() << " s " << std::endl;
        std::cout << "Writing 1-rdm to file" << std::endl;
        std::ofstream ofile( "1rdm_ed.dat", std::ios::out );

        auto w = std::setw(15);
        //ofile.precision(15);

        for( int i = 0; i < norb; i++ )
        {
          for( int j = 0; j < norb; j++ )
            ofile << scientific << w << ordm_ed[ i + norb * j ];
          ofile << std::endl;
        }
        ofile.close();
      }
 
      // Try running ASCI again, to check for MPI bugs
      if(world_rank == 0)
        std::cout << "Running ASCI instead!" << std::endl;
      auto asci_st = clock_type::now();
      if( norb < 16 )
        std::tie( E0_asci, ordm_asci ) = dbwy::run_asci_w_1rdm<32>( input );
      else if( norb < 32 )
        std::tie( E0_asci, ordm_asci ) = dbwy::run_asci_w_1rdm<64>( input );
      else
        std::tie( E0_asci, ordm_asci ) = dbwy::run_asci_w_1rdm<128>( input );
      auto asci_en = clock_type::now();
      if(world_rank == 0) 
      {
        std::cout << "After running ASCI: E0 = " << E0_asci << std::endl;
        std::cout << "Took: " << duration_type( asci_en - asci_st ).count() << " s " << std::endl;

        std::cout << "Energy error in ASCI: " << abs(E0_asci - E0_ed) << std::endl;
        //for( int i = 0; i < norb; i++ )
        //{
        //  if( i == 1 || i == 2 )
        //    continue;
        //  double tmp = ordm_asci[ 1 + norb * i ];
        //  ordm_asci[ 1 + norb * i ] = ordm_asci[ 2 + norb * i ];
        //  ordm_asci[ 2 + norb * i ] = tmp;
        //  tmp = ordm_asci[ i + norb * 1 ];
        //  ordm_asci[ i + norb * 1 ] = ordm_asci[ i + norb * 2 ];
        //  ordm_asci[ i + norb * 2 ] = tmp;
        //}
        //double tmp = ordm_asci[ 1 + norb * 1 ];
        //ordm_asci[ 1 + norb * 1 ] = ordm_asci[ 2 + norb * 2 ];
        //ordm_asci[ 2 + norb * 2 ] = tmp;
        //tmp = ordm_asci[ 1 + norb * 2 ];
        //ordm_asci[ 1 + norb * 2 ] = ordm_asci[ 2 + norb * 1 ];
        //ordm_asci[ 2 + norb * 1 ] = tmp;
        double rdm_err = 0.;
        for( int i = 0; i < ordm_asci.size(); i++ )
          rdm_err += std::pow( ordm_asci[i] - ordm_ed[i],2);
        rdm_err = std::sqrt(rdm_err);
        std::cout << "1-RDM error in ASCI: " << rdm_err << ", relative-error matrix: " << std::endl;
        auto w = std::setw(15);
        for( int i = 0; i < norb; i++ )
        {
          for( int j = 0; j < norb; j++ )
            std::cout << w << std::scientific << abs( ordm_asci[j + norb * i] - ordm_ed[j + norb * i] ) / abs( ordm_ed[j + norb * i ] );
          std::cout << std::endl;
        }
        std::ofstream ofile( "1rdm_asci.dat", std::ios::out );

        w = std::setw(15);
        //ofile.precision(15);

        for( int i = 0; i < norb; i++ )
        {
          for( int j = 0; j < norb; j++ )
            ofile << scientific << w << ordm_asci[ i + norb * j ];
          ofile << std::endl;
        }
        ofile.close();
      }
      MPI_Finalize();
    }// MPI scope
  
  }// try scope
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


}
