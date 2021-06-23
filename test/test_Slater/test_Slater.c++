/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests Slater determinant class.
 ******************************************/

#include "cmz_ed/slaterdet.h++"
#include<iostream>

using namespace std;
using namespace cmz::ed;

int main( int argn, char* argv[] )
{
  if( argn != 4 )
  {
    cout << "Usage: " << argv[0] << " <Norbs>  <Nup>  <Ndown>" << endl;
    return 0;
  }  
  try
  {
    unsigned short Norbs = atoi( argv[1] );
    unsigned short Nups  = atoi( argv[2] );
    unsigned short Ndos  = atoi( argv[3] );

    if( Norbs > 16 )
      throw( "cmz::ed is not ready for more than 16 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );

    cout << "Full Hilbert space has size: " << stts.size() << endl;
    for( SetSlaterDets_It it = stts.begin(); it != stts.end(); it++ )
    {
      cout << " --- " << it->ToStr() << endl;
      vector<unsigned short> occs_up = it->GetOccOrbsUp();
      vector<unsigned short> occs_do = it->GetOccOrbsDo();
      cout << " ------> Up Orbitals Occupied  : ";
      for( auto o : occs_up ) cout << o << ", ";
      cout << endl << " ------> Down Orbitals Occupied: ";
      for( auto o : occs_do ) cout << o << ", ";
      cout << endl;
    }
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
