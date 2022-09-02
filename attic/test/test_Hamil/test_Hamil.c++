/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests Fermionic Hamiltonian class.
 ******************************************/

#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include<iostream>

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

    for( SetSlaterDets_It ket = stts.begin(); ket != stts.end(); ket++ )
    {
      for( SetSlaterDets_It bra = stts.begin(); bra != stts.end(); bra++ )
      {
        cout << " --- " << bra->ToStrBra() << " H " << ket->ToStr() << " = " << Hop.GetHmatel( *bra, *ket ) <<endl;
      }
    }

    cout << "Checking GetSinglesAndDoubles: " << endl;
    for( SetSlaterDets_It ket = stts.begin(); ket != stts.end(); ket++ )
    {
      cout << " --- " << ket->ToStr() << ": " << endl;
      std::vector<slater_det> excs = ket->GetSinglesAndDoubles();
      for( auto exc : excs )
        cout << " --- ---- " << exc.ToStr() << endl;
    }

    std::vector<std::pair<size_t, size_t> > pairs = Hop.GetHpairs( stts );
  
    cout << "Checking FermionHamil::GetHpairs:" << endl;
    for( auto p : pairs )
      cout << "<" << p.first << "|" << p.second << ">" << endl;

    cout << "Checking GetHmat" << endl;
    SpMatD Hmat = GetHmat( &Hop, stts, print );

    cout << "H = " << endl << Eigen::MatrixXd( Hmat ) << endl;

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
