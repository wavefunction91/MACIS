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

Eigen::MatrixXd GetNatOrbRot( const vector<double> &ordm, const int norb )
{
  Eigen::MatrixXd rdm( norb, norb );
  for( int i = 0; i < norb; i++ )
    for( int j = 0; j < norb; j++ )
      rdm(i,j) = ordm[j + norb * i];

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver( rdm );

  return solver.eigenvectors();
}

Eigen::MatrixXd GetEigVals( const vector<double> &ordm, const int norb )
{
  Eigen::MatrixXd rdm( norb, norb );
  for( int i = 0; i < norb; i++ )
    for( int j = 0; j < norb; j++ )
      rdm(i,j) = ordm[j + norb * i];

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver( rdm );

  return solver.eigenvalues();
}

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

    if( Norbs > 12 )
      throw( "This test is not meant for more than 12 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    constexpr size_t nbits = 32;
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
      if( dets.size() <= 10 )
        for( const auto det : dets )
           cout << "## " << dbwy::to_canonical_string( det ) << endl;

      bool just_singles = false;
      intgrls::integrals ints(Norbs, fcidump, just_singles);
      double U = ints.getChem(0, 0, 0, 0);

      cout << "####################################################" << endl;
      // Hamiltonian Matrix Element Generator
      dbwy::SDBuildHamiltonianGenerator<nbits> 
        ham_gen( Norbs, ints.u.data(), ints.t.data() );
      ham_gen.SetJustSingles( just_singles );

      cout << "Building Hamiltonian matrix" << endl;
      auto H = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, h_el_tol );

      cout << "Hamiltonian matrix in Ground state space: " << endl;

      cout << "Computing Ground State..." << endl;

      double E0;
      Eigen::VectorXd psi0 = Eigen::VectorXd::Zero( dets.size() );

      E0 = p_davidson( 100, H, 1.E-8, psi0.data() );

      cout << "Ground state energy: " << E0 + ints.core_energy << endl;
      if( dets.size() <= 10 )
      {
        cout << "Ground state wave function:" << endl;
        for(int i = 0; i < psi0.size();i++)
          cout << psi0(i) << endl;
      }

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
      cout << "RDM-Evals: " << GetEigVals( ordm, Norbs ).transpose() << endl;
      cout << "RDM:" << endl;
      for( int i = 0; i < Norbs; i++ )
      {
        for( int j = 0; j < Norbs; j++ )
          cout << ordm[j + i * Norbs] << "  ";
        cout << endl;
      }

      cout << "####################################################" << endl;
      // Rotate with ham_gen routine.
      cout << "Rotate to natural orbitals with ham_gen routine" << endl; 
      ham_gen.rotate_hamiltonian_ordm( ordm.data() );
      ham_gen.SetJustSingles( false );

      cout << "Building Hamiltonian matrix" << endl;
      auto H2 = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, h_el_tol );

      cout << "Hamiltonian matrix in Ground state space: " << endl;

      cout << "Computing Ground State..." << endl;

      double E0_2;
      psi0 = Eigen::VectorXd::Zero( dets.size() );

      E0_2 = p_davidson( 100, H2, 1.E-8, psi0.data() );

      cout << "Ground state energy: " << E0_2 + ints.core_energy << endl;
      if( dets.size() <= 10 )
      {
        cout << "Ground state wave function:" << endl;
        for(int i = 0; i < psi0.size();i++)
          cout << psi0(i) << endl;
      }

      cout << "Building rdms!!" << endl;

      vector<double> ordm2( Norbs*Norbs, 0. );
      ham_gen.form_rdms( dets.begin(), dets.end(), 
                         dets.begin(), dets.end(),
                         psi0.data(), ordm2.data() );

      for( int i = 0; i < Norbs; i++ )
        occs[i] = ordm2[i + i * Norbs] / 2.;
      cout << "Occs: ";
      for( const auto oc : occs )
        cout << oc << ", ";
      cout << endl;
      cout << "RDM-Evals: " << GetEigVals( ordm2, Norbs ).transpose() << endl;
      cout << "RDM:" << endl;
      for( int i = 0; i < Norbs; i++ )
      {
        for( int j = 0; j < Norbs; j++ )
          cout << ordm2[j + i * Norbs] << "  ";
        cout << endl;
      }

      cout << "####################################################" << endl;
      // Rotate with ints routine.
      cout << "Rotate to natural orbitals with ints routine" << endl; 
      Eigen::MatrixXd nat_orb_rot = GetNatOrbRot( ordm, Norbs );
      intgrls::integrals ints2(Norbs, fcidump);
      ints2.rotate_orbitals( nat_orb_rot );
      dbwy::SDBuildHamiltonianGenerator<nbits> 
        ham_gen2( Norbs, ints2.u.data(), ints2.t.data() );
      ham_gen2.SetJustSingles( false );

      cout << "Building Hamiltonian matrix" << endl;
      auto H3 = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen2, h_el_tol );

      cout << "Hamiltonian matrix in Ground state space: " << endl;

      cout << "Computing Ground State..." << endl;

      double E0_3;
      psi0 = Eigen::VectorXd::Zero( dets.size() );

      E0_3 = p_davidson( 100, H3, 1.E-8, psi0.data() );

      cout << "Ground state energy: " << E0_3 + ints.core_energy << endl;
      if( dets.size() <= 10 )
      {
        cout << "Ground state wave function:" << endl;
        for(int i = 0; i < psi0.size();i++)
          cout << psi0(i) << endl;
      }

      cout << "Building rdms!!" << endl;

      vector<double> ordm3( Norbs*Norbs, 0. );
      ham_gen2.form_rdms( dets.begin(), dets.end(), 
                          dets.begin(), dets.end(),
                          psi0.data(), ordm3.data() );

      for( int i = 0; i < Norbs; i++ )
        occs[i] = ordm3[i + i * Norbs] / 2.;
      cout << "Occs: ";
      for( const auto oc : occs )
        cout << oc << ", ";
      cout << endl;
      cout << "RDM-Evals: " << GetEigVals( ordm3, Norbs ).transpose() << endl;
      cout << "RDM:" << endl;
      for( int i = 0; i < Norbs; i++ )
      {
        for( int j = 0; j < Norbs; j++ )
          cout << ordm3[j + i * Norbs] << "  ";
        cout << endl;
      }

      double Eerr_hrot = std::abs( E0 - E0_2 );
      double Eerr_irot = std::abs( E0 - E0_3 );
      double RDMeverr_hrot = ( GetEigVals( ordm, Norbs ) - GetEigVals( ordm2, Norbs ) ).norm();
      double RDMeverr_irot = ( GetEigVals( ordm, Norbs ) - GetEigVals( ordm3, Norbs ) ).norm();

      cout << "Error in the energy" << endl;
      cout << "    * For ham_gen.rotate_hamiltonian_ordm: " << Eerr_hrot << endl;
      cout << "    * For ints.rotate_orbitals           : " << Eerr_irot << endl << endl;

      
      cout << "Error in the RDM eigenvalues" << endl;
      cout << "    * For ham_gen.rotate_hamiltonian_ordm: " << RDMeverr_hrot << endl;
      cout << "    * For ints.rotate_orbitals           : " << RDMeverr_irot << endl;

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
