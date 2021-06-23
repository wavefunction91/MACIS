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

    if( Norbs > 16 )
      throw( "cmz::ed is not ready for more than 16 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );

    intgrls::integrals ints(Norbs, fcidump);

    FermionHamil Hop(ints);

    cout << "Building Hamiltonian matrix" << endl;
    SpMatD Hmat = GetHmat( &Hop, stts, print );

    cout << "Computing Ground State..." << endl;

    double E0;
    VectorXd psi0;

    SpMatDOp Hwrap( Hmat );

    GetGS( Hwrap, E0, psi0, input );

    cout << "Ground state energy: " << E0 + ints.core_energy << endl;

    cout << "Building rdms!!" << endl;

    rdm::rdms rdms( Norbs, psi0, stts );

    cout << "Testing energy with rdms..." << endl;
   
    double E0_rdm = MyInnProd( ints.t, rdms.rdm1 ) + MyInnProd( ints.u, rdms.rdm2 ) + ints.get_core_energy();

    cout << "E0 = " << E0 + ints.core_energy << ", E0_rdm = " << E0_rdm << endl;
 
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
