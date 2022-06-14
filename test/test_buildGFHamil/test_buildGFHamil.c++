/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests GF Hamiltonian building routine, on Hubbard dimer. 
 ******************************************/

#include "cmz_ed/integrals.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include "dbwy/gf.h++"
#include<iostream>
#include<fstream>
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
      throw( "This test is not meant for more than 8 orbitals!" );
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
      vector<bitset<nbits> > dets;
      dets = dbwy::generate_full_hilbert_space<nbits>( Norbs, Nups, Ndos );

      cout << "Ground state Basis:" << endl;
      for( const auto det : dets )
         cout << "## " << dbwy::to_canonical_string( det ) << endl;

      intgrls::integrals ints(Norbs, fcidump);

      // Hamiltonian Matrix Element Generator
      dbwy::DoubleLoopHamiltonianGenerator<nbits> 
        ham_gen( Norbs, ints.u.data(), ints.t.data() );

      cout << "Building Hamiltonian matrix" << endl;
      auto H = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, h_el_tol );

      cout << "Hamiltonian matrix in Ground state space: " << endl;

      cout << "Computing Ground State..." << endl;

      double E0;
      Eigen::VectorXd psi0 = Eigen::VectorXd::Zero( dets.size() );

      E0 = p_davidson( 100, H, 1.E-8, psi0.data() );

      cout << "Ground state energy: " << E0 + ints.core_energy << endl;
      cout << "Ground state wave function:" << endl;
      for(int i = 0; i < psi0.size();i++)
        cout << psi0(i) << endl;

      cout << "Building rdms!!" << endl;

      vector<double> ordm( Norbs*Norbs, 0. );
      ham_gen.form_rdms( dets.begin(), dets.end(), 
                         dets.begin(), dets.end(),
                         psi0.data(), ordm.data() );

      VecD occs(Norbs, 0.);
      for( int i = 0; i < Norbs; i++ )
        occs[i] += ordm[i + i * Norbs] / 2.;
      cout << "Occs: ";
      for( const auto oc : occs )
        cout << oc << ", ";
      cout << endl;

      vector<bitset<nbits> > dets_add, dets_sub;

      // PARTICLE SECTOR
      VecInt GF_orbs = {0, 1};
      vector<bool> is_up = {false, false};
      //vector<bool> is_up = {true, true};
      bool is_part = true;
      get_GF_basis_AS_1El<nbits>( GF_orbs[0], is_up[0], is_part, psi0, dets, dets_add, occs, input );
      VecInt todelete;

      cout << "Particle state Basis:" << endl;
      for( const auto det : dets_add )
         cout << "## " << dbwy::to_canonical_string( det ) << endl;

      auto Hgf_add = BuildHamil4GF<nbits>( ham_gen, dets_add, MPI_COMM_WORLD, h_el_tol, true );

      cout << "Effective Hamiltonian matrix in Particle space: " << endl;

      vector<VecD> Part_wfns4Lan = BuildWfn4Lanczos<nbits>( psi0, GF_orbs, is_up, dets, dets_add, is_part, todelete );

      cout << "Wave functions for particle sector: " << endl;

      for(int iii = 0; iii < GF_orbs.size(); iii++)
      {
        cout << "## Orbital " << GF_orbs[iii] << endl;
        for( const double el : Part_wfns4Lan[iii] )
          cout << "## " << el << endl;
      }
      cout << endl;    

      // HOLE SECTOR
      is_part = false;
      get_GF_basis_AS_1El<nbits>( GF_orbs[0], is_up[0], is_part, psi0, dets, dets_sub, occs, input );

      cout << "Hole state Basis:" << endl;
      for( const auto det : dets_sub )
         cout << "## " << dbwy::to_canonical_string( det ) << endl;

      auto Hgf_sub = BuildHamil4GF<nbits>( ham_gen, dets_sub, MPI_COMM_WORLD, h_el_tol, true );

      cout << "Effective Hamiltonian matrix in Hole space: " << endl;

      vector<VecD> Hole_wfns4Lan = BuildWfn4Lanczos<nbits>( psi0, GF_orbs, is_up, dets, dets_sub, is_part, todelete );

      cout << "Wave functions for hole sector: " << endl;

      for(int iii = 0; iii < GF_orbs.size(); iii++)
      {
        cout << "## Orbital " << GF_orbs[iii] << endl;
        for( const double el : Hole_wfns4Lan[iii] )
          cout << "## " << el << endl;
      }
      cout << endl;    
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
