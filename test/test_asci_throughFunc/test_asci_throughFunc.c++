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
  
      double E0;
      std::vector<double> ordm;
      if( norb < 16 )
        std::tie( E0, ordm ) = dbwy::run_asci_w_1rdm<32>( input );
      else if( norb < 32 )
        std::tie( E0, ordm ) = dbwy::run_asci_w_1rdm<64>( input );
      else
        std::tie( E0, ordm ) = dbwy::run_asci_w_1rdm<128>( input );

      if(world_rank == 0) 
      {
        std::cout << "After running ASCI: E0 = " << E0 << std::endl;
        std::cout << "Writing 1-rdm to file" << std::endl;
        std::ofstream ofile( "1rdm.dat", std::ios::out );

        auto w = std::setw(25);
        ofile.precision(15);

        for( int i = 0; i < norb; i++ )
        {
          for( int j = 0; j < norb; j++ )
            ofile << scientific << w << ordm[ i + norb * j ];
          ofile << std::endl;
        }
        ofile.close();
      }
 
      // Try running ASCI again, to check for MPI bugs
      if(world_rank == 0)
        std::cout << "Running ASCI a second time!" << std::endl;
      if( norb < 16 )
        std::tie( E0, ordm ) = dbwy::run_asci_w_1rdm<32>( input );
      else if( norb < 32 )
        std::tie( E0, ordm ) = dbwy::run_asci_w_1rdm<64>( input );
      else
        std::tie( E0, ordm ) = dbwy::run_asci_w_1rdm<128>( input );
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
