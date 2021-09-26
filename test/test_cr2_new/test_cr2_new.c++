/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests H2O 6-31g active space (8e, 5o) Hamiltonian.
 ******************************************/
#include "cmz_ed/slaternorm.h++"
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

//    if( Norbs > 16 )
 //     throw( "cmz::ed is not ready for more than 16 orbitals!" );
  //  if( Nups > Norbs || Ndos > Norbs )
 //     throw( "Nups or Ndos cannot be larger than Norbs!" );

   //SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );
 
    SetSlaterDets stts;
    uint64_t u1 = 37793167;
    uint64_t d1 = 37793167;
    u1 = (1LL << Nups)-1;
    d1 = (1LL << Ndos)-1;
    vector<uint64_t> alpha ;
    vector<uint64_t> beta ;
    std::vector<uint64_t> talph,tbeta;
    alpha.push_back(u1);
    beta.push_back(u1);

    uint64_t st = (d1 << Norbs) + u1;
    std::bitset<32> st1;
    //stts.insert( slater_det( st, Norbs, Nups, Ndos ) );
    slater_det hello =  slater_det( st, Norbs, Nups, Ndos ) ;
    //slater_det_l testing =  slater_det_l( st1, Norbs, Nups, Ndos ) ;
    slater_det bye =  slater_det( 0, 16, Nups, Ndos ) ;


    intgrls::integrals ints(Norbs, fcidump);

    FermionHamil Hop(ints);
    double nE =  Hop.GetHmatel(hello,hello);

    cout << "norm here" <<endl;
    cout << std::setprecision(16) << "E0 = " << nE + ints.core_energy << endl;
   
    int occ = 12;
    //singlesdoubles(alpha[0],beta[0],occ,talph,tbeta);

    //cout << talph[1] << " " << talph[2] << " " << tbeta[1] << " " << tbeta[2] << endl;

    return(1);   
    cout << "Building Hamiltonian matrix" << endl;
    SpMatD Hmat = GetHmat( &Hop, stts, print );

    cout << "Computing Ground State..." << endl;

    double E0;
    VectorXd psi0;

    SpMatDOp Hwrap( Hmat );

    GetGS( Hwrap, E0, psi0, input );

    cout << std::setprecision(9) << "Ground state energy: " << E0 + ints.core_energy << endl;

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
