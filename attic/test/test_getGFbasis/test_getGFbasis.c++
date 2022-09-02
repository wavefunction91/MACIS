/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests routine to build GF space 
 ******************************************/

#include "dbwy/gf.h++"
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
    size_t    trunc_size = getParam<int>( input, "trunc_size" );
    int           tot_SD = getParam<int>( input, "tot_SD" );
    double     GFsThresh = getParam<double>( input, "GFseedThresh" );
    double      asThresh = getParam<double>( input, "asThresh" );
    bool print = true;

    if( Norbs > 4 )
      throw( "This test can be run with at most 4 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    constexpr size_t nbits = 8;
    vector<bitset<nbits> > dets;
    dets = dbwy::generate_full_hilbert_space<nbits>( Norbs, Nups, Ndos );

    vector<bitset<nbits> > old_basis( dets.begin(), dets.end() );
    vector<bitset<nbits> > new_basis;

    VectorXd wfn = VectorXd::Ones(dets.size());
    VecD occs( Norbs, 0.5 );

    cout << "Old basis: " << endl;
    for(int i = 0; i < old_basis.size(); i++)
      cout << "## " << dbwy::to_canonical_string( old_basis[i] ) << endl;

    bool sp_up = true;
    // Spin up
    cout << " SPIN UP " << endl;
    for( int orb = 0; orb < Norbs; orb++ )
    {

      get_GF_basis_AS_1El<nbits>( orb, sp_up, true, wfn, old_basis, new_basis, occs, input );
 
      cout << "##############################################" << endl << "New basis: " << endl;
      cout << "## Adding particle in orbital " << orb << endl;
      for(int i = 0; i < new_basis.size(); i++)
        cout << "## " << dbwy::to_canonical_string( new_basis[i] ) << endl;
      break;
    }
    sp_up = false;
    // Spin down
    cout << " SPIN DOWN " << endl;
    for( int orb = 0; orb < Norbs; orb++ )
    {

      break;
      get_GF_basis_AS_1El<nbits>( orb, sp_up, true, wfn, old_basis, new_basis, occs, input );
 
      cout << "##############################################" << endl << "New basis: " << endl;
      cout << "## Adding particle in orbital " << orb << endl;
      for(int i = 0; i < new_basis.size(); i++)
        cout << "## " << dbwy::to_canonical_string( new_basis[i] ) << endl;

    }
    sp_up = true;
    // Spin up
    cout << " SPIN UP " << endl;
    for( int orb = 0; orb < Norbs; orb++ )
    {

      get_GF_basis_AS_1El<nbits>( orb, sp_up, false, wfn, old_basis, new_basis, occs, input );
 
      cout << "##############################################" << endl << "New basis: " << endl;
      cout << "## Removing particle in orbital " << orb << endl;
      for(int i = 0; i < new_basis.size(); i++)
        cout << "## " << dbwy::to_canonical_string( new_basis[i] ) << endl;
      break;
    }
    sp_up = false;
    // Spin down
    cout << " SPIN DOWN " << endl;
    for( int orb = 0; orb < Norbs; orb++ )
    {

      break;
      get_GF_basis_AS_1El<nbits>( orb, sp_up, false, wfn, old_basis, new_basis, occs, input );
 
      cout << "##############################################" << endl << "New basis: " << endl;
      cout << "## Removing particle in orbital " << orb << endl;
      for(int i = 0; i < new_basis.size(); i++)
        cout << "## " << dbwy::to_canonical_string( new_basis[i] ) << endl;
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

