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

template<class Iterable>
complex<double> MyTrapz( const Iterable &f, const double dx )
{
  // Implements simple trapezoidal rule
  complex<double> res( 0.,0. );

  for( typename Iterable::const_iterator it = f.begin() + 1; it != f.end(); it++ )
    res += (*it + *(it-1)) / 2. * dx;

  return res;
}

CompD HubDimerGF( CompD w, int i, int j, double U )
{
  // Analytical Green's function for the Hubbard dimer.
  CompD res(0., 0.);
  CompD I(0., 1.);

  double c = sqrt(16. + U * U);
  double a = sqrt(2. * ( 1. + 16. / ( (c - U) * (c - U) ) ));
  double sign = ( (i-j) % 2 == 1)? -1. : 1.;
  
  res = sign *( std::pow( (1. + 4. / (c - U)), 2) / (w - (c / 2. - 1.) )
               + sign * std::pow( (1. - 4. / (c - U)), 2) / (w - (c / 2. + 1.) ) );
  res +=        std::pow( (1. + 4. / (c - U)), 2) / (w + (c / 2. - 1.) )
              + sign * std::pow( (1. - 4. / (c - U)), 2) / (w + (c / 2. + 1.) );

  return res / 2. / a / a;
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
      vector<bitset<nbits> > dets;
      dets = dbwy::generate_full_hilbert_space<nbits>( Norbs, Nups, Ndos );

      cout << "Ground state Basis:" << endl;
      for( const auto det : dets )
         cout << "## " << dbwy::to_canonical_string( det ) << endl;

      intgrls::integrals ints(Norbs, fcidump);
      double U = ints.getChem(0, 0, 0, 0);

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

      // GREEN'S FUNCTION CALCULATION
      std::cout << "About to compute GF for wave function:" << std::endl;
      auto w = std::setw(20);
      for( int idet = 0; idet < dets.size(); idet++ )
         std::cout << psi0(idet) << w << dbwy::to_canonical_string( dets[idet] ) << std::endl; 
      std::cout << std::endl;
      
      size_t nws  = getParam<int>( input, "nws" );
      double wmin = getParam<double>( input, "wmin" );
      double wmax = getParam<double>( input, "wmax" );
      double eta  = getParam<double>( input,  "broad" );
      VecCompD ws(nws);
      for( int iw = 0; iw < nws; iw++ )
        ws[iw] = CompD( wmin + double(iw) * (wmax - wmin) / (double(nws - 1)) , eta );

      // PARTICLE SECTOR
      bool is_part = true;
      vector<vector<vector<complex<double> > > > GF_add;
      RunGFCalc<nbits>( GF_add, psi0, ham_gen, dets, E0, is_part, ws, occs, input );

      // HOLE SECTOR
      is_part = false;
      vector<vector<vector<complex<double> > > > GF_sub;
      RunGFCalc<nbits>( GF_sub, psi0, ham_gen, dets, E0, is_part, ws, occs, input );

      // CHECK SUM RULE
      vector<double > trGF(nws);
      for( size_t iw = 0; iw < nws; iw++ )
        for( size_t iorb = 0; iorb < GF_add[iw].size();  iorb++ )
          trGF[iw] += -1. / M_PI * imag(GF_add[iw][iorb][iorb] + GF_sub[iw][iorb][iorb]);

      cout << "SUM RULE CHECK: " << real( MyTrapz( trGF, real(ws[1] - ws[0]) ) ) << " == " << GF_add[0].size() << "?" << endl;

      // Compare to ED result
      double avg_err = 0.;
      for( size_t iw = 0; iw < nws; iw++ )
        for( size_t iorb1 = 0; iorb1 < GF_add[iw].size(); iorb1++ )
          for( size_t iorb2 = 0; iorb2 < GF_add[iw][0].size(); iorb2++ )
            avg_err = std::abs( GF_add[iw][iorb1][iorb2] + GF_sub[iw][iorb1][iorb2] - HubDimerGF( ws[iw], iorb1, iorb2, U ) );

      avg_err = avg_err / double(nws) / double(GF_add[0].size()) / double(GF_add[0].size());
      cout << "AVG GF ERROR COMPARED TO ANALYTICAL RESULT: " << avg_err << endl;

      //ofstream ofile("gf_exact.dat", ios::out);
      //ofile.precision(15);
      //auto w = setw(25);
      //
      //for( size_t iw = 0; iw < nws; iw++ )
      //{
      //  ofile << scientific << real(ws[iw]);
      //  for( size_t iorb1 = 0; iorb1 < GF_add[iw].size(); iorb1++ )
      //    for( size_t iorb2 = 0; iorb2 < GF_add[iw][0].size(); iorb2++ )
      //    {
      //      CompD GF = HubDimerGF( ws[iw], iorb1, iorb2, U);
      //      ofile << scientific << w << real(GF) << w << imag(GF);
      //    }
      //  ofile << endl;
      //}    
   
      //ofile.close();
      //ofile.open("gf_numeric.dat", ios::out);
      //ofile.precision(15);
      //
      //for( size_t iw = 0; iw < nws; iw++ )
      //{
      //  ofile << scientific << real(ws[iw]);
      //  for( size_t iorb1 = 0; iorb1 < GF_add[iw].size(); iorb1++ )
      //    for( size_t iorb2 = 0; iorb2 < GF_add[iw][0].size(); iorb2++ )
      //    {
      //      CompD GF = GF_add[iw][iorb1][iorb2] + GF_sub[iw][iorb1][iorb2];
      //      ofile << scientific << w << real(GF) << w << imag(GF);
      //    }
      //  ofile << endl;
      //}    
   
      //ofile.close();

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
