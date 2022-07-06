/***Written by Carlos Mejuto Zaera***/

/***Note***********************************
 * Tests the fast diagonal routines
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

    if( Norbs > 32 )
      throw( "This test is not meant for more than 32 orbitals!" );
    if( Nups > Norbs || Ndos > Norbs )
      throw( "Nups or Ndos cannot be larger than Norbs!" );

    constexpr size_t nbits = 64;
    double h_el_tol = 1.E-6;
    MPI_Init(NULL,NULL);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    if(world_size != 1) 
      throw "NO MPI"; // Disable MPI for now
    { // MPI Scope
      intgrls::integrals ints(Norbs, fcidump);
      vector<double> orb_ens( Norbs, 0. );
      for( int i = 0; i < Norbs; i++ )
        orb_ens[i] = ints.get( i, i );

      bitset<nbits>  hf_det = dbwy::canonical_hf_determinant<nbits>( Nups, Ndos, orb_ens );

      // Hamiltonian Matrix Element Generator
      dbwy::DoubleLoopHamiltonianGenerator<nbits> 
        ham_gen( Norbs, ints.u.data(), ints.t.data() );

      const double hf_en = ham_gen.matrix_element(hf_det, hf_det); 

      cout << "'HF' determinant: ";
      cout << "## " << dbwy::to_canonical_string( hf_det ) << endl;
      cout << "With energy: " << hf_en << endl;

      // Separate out into alpha/beta components 
      bitset<nbits/2> hf_alpha = dbwy::truncate_bitset<nbits/2>(hf_det);
      bitset<nbits/2> hf_beta  = dbwy::truncate_bitset<nbits/2>(hf_det >> (nbits/2));
      
      // Get occupied indices
      vector<uint32_t> hf_occ_alpha, hf_occ_beta;
      dbwy::bits_to_indices( hf_alpha, hf_occ_alpha );
      dbwy::bits_to_indices(  hf_beta,  hf_occ_beta );
                      
      cout << "Building singles and doubles" << endl;
      vector<bitset<nbits> > singles, doubles;
      dbwy::generate_singles_doubles_spin( Norbs, hf_det, singles, doubles );

      cout << "Checking single fast diagonal energies!" << endl;
      double avg_err = 0.;
      for( const auto sdet : singles )
      {
        bitset<nbits/2> ket_alpha = dbwy::truncate_bitset<nbits/2>( sdet );
        bitset<nbits/2> ket_beta  = dbwy::truncate_bitset<nbits/2>( sdet >> (nbits/2));

        bitset<nbits> ex_total = hf_det ^ sdet;

        bitset<nbits/2> ex_alpha = dbwy::truncate_bitset<nbits/2>( ex_total );
        bitset<nbits/2> ex_beta  = dbwy::truncate_bitset<nbits/2>( ex_total >> (nbits/2) );

        // Compute Matrix Element explicitly
        const auto h_el = ham_gen.matrix_element( sdet, sdet ); 
        // Compute Matrix Element with fast diagonal routine
        double h_elf = 0.;
        if( ex_alpha.count() == 2 )
        {
          auto [o1, v1, sign] = dbwy::single_excitation_sign_indices( ket_alpha, hf_alpha, ex_alpha );
          h_elf = ham_gen.fast_diag_single( hf_occ_alpha, hf_occ_beta, o1, v1, hf_en );
        }
        else if( ex_beta.count() == 2 )
        {
          auto [o1, v1, sign] = dbwy::single_excitation_sign_indices( ket_beta, hf_beta, ex_beta );
          h_elf = ham_gen.fast_diag_single( hf_occ_beta, hf_occ_alpha, o1, v1, hf_en );
        }
        else
          throw( std::runtime_error("Error in singles loop! One determinant is not a single!") );
        avg_err += abs( h_elf - h_el );
                          
      }
      cout << "Average error in singles energies: " << avg_err / double( singles.size() ) << endl;

      cout << "Checking double fast diagonal energies!" << endl;
      avg_err = 0.;
      for( const auto ddet : doubles )
      {
        bitset<nbits/2> ket_alpha = dbwy::truncate_bitset<nbits/2>( ddet );
        bitset<nbits/2> ket_beta  = dbwy::truncate_bitset<nbits/2>( ddet >> (nbits/2));

        bitset<nbits> ex_total = hf_det ^ ddet;

        bitset<nbits/2> ex_alpha = dbwy::truncate_bitset<nbits/2>( ex_total );
        bitset<nbits/2> ex_beta  = dbwy::truncate_bitset<nbits/2>( ex_total >> (nbits/2) );

        // Compute Matrix Element explicitly
        const auto h_el = ham_gen.matrix_element( ddet, ddet ); 
        // Compute Matrix Element with fast diagonal routine
        double h_elf = 0.;
        if( ex_alpha.count() == 4 )
        {
          auto [o1, v1, o2, v2, sign] = dbwy::doubles_sign_indices( ket_alpha, hf_alpha, ex_alpha );
          h_elf = ham_gen.fast_diag_ss_double( hf_occ_alpha, hf_occ_beta, o1, o2, v1, v2, hf_en );
        }
        else if( ex_beta.count() == 4 )
        {
          auto [o1, v1, o2, v2, sign] = dbwy::doubles_sign_indices( ket_beta, hf_beta, ex_beta );
          h_elf = ham_gen.fast_diag_ss_double( hf_occ_beta, hf_occ_alpha, o1, o2, v1, v2, hf_en );
        }
        else if( ex_alpha.count() == 2 && ex_beta.count() == 2 )
        {
          auto [ou,vu,sign_u] = 
            dbwy::single_excitation_sign_indices( ket_alpha, hf_alpha, ex_alpha );
          auto [od,vd,sign_d] = 
            dbwy::single_excitation_sign_indices( ket_beta,  hf_beta,  ex_beta  );
          h_elf = ham_gen.fast_diag_os_double( hf_occ_alpha, hf_occ_beta, ou, od, vu, vd, hf_en );
        }
        else
          throw( std::runtime_error("Error in singles loop! One determinant is not a single!") );
        avg_err += abs( h_elf - h_el );
                          
      }
      cout << "Average error in doubles energies: " << avg_err / double( doubles.size() ) << endl;

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
