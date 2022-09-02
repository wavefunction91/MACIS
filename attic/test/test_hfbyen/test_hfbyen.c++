/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests the full GF computation on a Hubbard dimer 
 ******************************************/

#include "cmz_ed/integrals.h++"
#include "cmz_ed/lanczos.h++"
#include "dbwy/gf.h++"
#include <iostream>
#include <fstream>
#include <math.h>
#include "unsupported/Eigen/SparseExtra"
#include "dbwy/asci_body.hpp"

using namespace std;
using namespace cmz::ed;

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

    if( Norbs > 4 )
      throw( "This test is not meant for more than 4 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    constexpr size_t nbits = 8;
    double h_el_tol = 1.E-6;
    MPI_Init(NULL,NULL);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    if(world_size != 1) 
      throw "NO MPI"; // Disable MPI for now
    { // MPI Scope
      intgrls::integrals ints(Norbs, fcidump);
      vector<double> orb_ens( Norbs );
      for( int i = 0; i < Norbs; i++ )
        orb_ens[i] = ints.get( i, i );

      bitset<nbits> hf = dbwy::canonical_hf_determinant<nbits>( Nups, Ndos, orb_ens );

      cout << "Non-interacting HF:" << endl;
      cout << dbwy::to_canonical_string( hf ) << endl;

      MPI_Finalize();
    }// MPI scope
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
