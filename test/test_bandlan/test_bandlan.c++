#include <mpi.h>
#include <iostream>
#include "cmz_ed/bandlan.h++"
#include "dbwy/asci_body.hpp"

using namespace std;

int main( int argn, char *argv[] )
{
  try
  {
    if( argn != 2 )
    {
      std::cout << "Usage: " << argv[0] << " <input-file> " << std::endl;
      throw( runtime_error( "Call error!" ) );
    }

    vector<vector<double> > qs(4, vector<double>(4,0.)), bandH;
    qs[0][0] = 1.; qs[0][1] = 0.; qs[0][2] = 0.; qs[0][3] = 0.; 
    qs[1][0] = 0.; qs[1][1] = 1.; qs[1][2] = 0.; qs[1][3] = 0.; 
    qs[2][0] = 0.; qs[2][1] = 0.; qs[2][2] = 1.; qs[2][3] = 0.; 
    qs[3][0] = 0.; qs[3][1] = 0.; qs[3][2] = 0.; qs[3][3] = 1.; 
    int nLanIts = 4;
    constexpr size_t nbits = 8;
    std::vector<double> evals;
    std::vector<double> X;
    
    // Read Input
    std::string in_file = argv[1];
    cmz::ed::Input_t input;
    cmz::ed::ReadInput( in_file, input );
    size_t norb = cmz::ed::getParam<int>( input, "norbs" );
    size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
    size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
    string fcidump = 
      cmz::ed::getParam<std::string>( input, "fcidump_file" );
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
      auto mat  = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, 1.E-10 );
      cmz::ed::MyBandLan<double>(mat, qs, bandH, nLanIts);
      

      MPI_Finalize();
    }// MPI scope

    cout << "bandH : " << endl;
    for(int i = 0; i < bandH.size(); i++){
      for(int j = 0; j < bandH[i].size(); j++) cout << bandH[i][j] << "  ";
      cout << endl;
    }

    vector<double> eigvals;
    vector<vector<double> > eigvecs;

    cmz::ed::GetEigsysBand(bandH, 2, eigvals, eigvecs);

    cout << "Eigenvalues: [";
    for(int i = 0; i < eigvals.size(); i++) cout << eigvals[i] << ", ";
    cout << "]" << endl;

    cout << "Eigenvectors: " << endl;
    for(int i = 0; i < eigvecs.size(); i++){
      cout << "v[" << i << "] = [";
      for(int j = 0; j < eigvecs[i].size(); j++) cout << eigvecs[i][j] << ", ";
      cout << "]" << endl;
    }
  }
  catch( std::runtime_error e )
  {
    std::cout << "Caught std::runtime_error!" << std::endl;
    std::cout << e.what() << std::endl;
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
}
