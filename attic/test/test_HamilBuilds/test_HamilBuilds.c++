/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests Hamiltonian building routines, on Hubbard dimer. 
 ******************************************/

#include "cmz_ed/integrals.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"
#include "dbwy/gf.h++"
#include<iostream>
#include<fstream>
#include "unsupported/Eigen/SparseExtra"
#include "dbwy/asci_body.hpp"
#include "dbwy/sd_build.hpp"

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

      cout << "Building Hamiltonian matrix with double loop" << endl;
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
        occs[i] = ordm[i + i * Norbs] / 2.;
      cout << "Occs: ";
      for( const auto oc : occs )
        cout << oc << ", ";
      cout << endl;

      // Hamiltonian Matrix Element Generator
      dbwy::SDBuildHamiltonianGenerator<nbits> 
        ham_gen2( Norbs, ints.u.data(), ints.t.data() );

      cout << "Building Hamiltonian matrix with SD search" << endl;
      auto H2 = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen2, h_el_tol );

      cout << "Hamiltonian matrix in Ground state space: " << endl;

      cout << "Computing Ground State..." << endl;

      E0 = p_davidson( 100, H2, 1.E-8, psi0.data() );

      cout << "Ground state energy: " << E0 + ints.core_energy << endl;
      cout << "Ground state wave function:" << endl;
      for(int i = 0; i < psi0.size();i++)
        cout << psi0(i) << endl;

      cout << "Building rdms!!" << endl;

      for( int i = 0; i < ordm.size(); i++ )
        ordm[i] *= 0.;
      ham_gen2.form_rdms( dets.begin(), dets.end(), 
                          dets.begin(), dets.end(),
                          psi0.data(), ordm.data() );

      for( int i = 0; i < Norbs; i++ )
        occs[i] = ordm[i + i * Norbs] / 2.;
      cout << "Occs: ";
      for( const auto oc : occs )
        cout << oc << ", ";
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
