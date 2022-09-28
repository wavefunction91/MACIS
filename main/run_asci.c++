#include <mpi.h>
#include <iostream>
#include "dbwy/asci_body.hpp"
#include "cmz_ed/freq_grids.h++"

using namespace std;

int main( int argc, char* argv[] ) {


  cout << argv[0] << " ..Code implementing an ASCI selected CI implementation" 
            << endl;

  if( argc != 2 ) {
    cout << "Usage: " << argv[0] << " <Input-File>" << endl;
    return 1;
  }
  try
  {

    // Read Input
    std::string in_file = argv[1];
    cmz::ed::Input_t input;
    cmz::ed::ReadInput( in_file, input );
    size_t norb = cmz::ed::getParam<int>( input, "norbs" );
    bool ed_mode = false;
    try{ ed_mode = cmz::ed::getParam<bool>( input, "asci_ed_mode" ); } catch(...){ }
    if( ed_mode && norb > 14 )
      throw( std::runtime_error( "Error in run_asci_impsolv! Asked for too many orbitals in ED mode. Max is 14" ) );
    MPI_Init(NULL,NULL);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    if(world_size != 1) 
      throw "NO MPI"; // Disable MPI for now
    { // MPI Scope
  
      double E0;
      if( ed_mode )
        E0 = dbwy::run_ed<32>( input );
      else if( norb < 2 )
        E0 = dbwy::run_asci<4>( input );
      else if( norb < 4 )
        E0 = dbwy::run_asci<8>( input );
      else if( norb < 8 )
        E0 = dbwy::run_asci<16>( input );
      else if( norb < 16 )
        E0 = dbwy::run_asci<32>( input );
      else if( norb < 32 )
        E0 = dbwy::run_asci<64>( input );
      else
        E0 = dbwy::run_asci<128>( input );

      if(world_rank == 0) 
        std::cout << "After running ASCI: E0 = " << std::scientific << std::setprecision(15) << E0 << std::endl;
 
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

