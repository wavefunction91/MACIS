#include <mpi.h>
#include <iostream>
#include "cmz_ed/utils.h++"

#include <bitset>

template <size_t N>
std::vector<std::bitset<N>> build_combs( uint64_t nbits, uint64_t nset ) {

  std::vector<bool> v(nbits, false);
  std::fill_n( v.begin(), nset, true );
  std::vector<std::bitset<N>> store;

  do {

    std::bitset<N> temp = 0ul;
    std::bitset<N> one  = 1ul;
    for( uint64_t i = 0; i < nbits; ++i )
    if( v[i] ) {
      temp = temp | (one << i);
    }
    store.emplace_back(temp);

  } while ( std::prev_permutation( v.begin(), v.end() ) );

  return store;

}

template <size_t N>
std::vector<std::bitset<N>> build_hilbert_space(
  size_t norbs, size_t nalpha, size_t nbeta
) {

  // Get all alpha and beta combs
  auto alpha_dets = build_combs<N>( norbs, nalpha );
  auto beta_dets  = build_combs<N>( norbs, nbeta  );

  std::vector<std::bitset<N>> states;
  states.reserve( alpha_dets.size() * beta_dets.size() );
  for( auto alpha_det : alpha_dets )
  for( auto beta_det  : beta_dets  ) {
    std::bitset<N> state = alpha_det | (beta_det << (N/2));
    states.emplace_back( state );
  }

  return states;
 
}

int main( int argc, char* argv[] ) {

  MPI_Init(NULL,NULL);

  if( argc != 2 ) {
    std::cout << "Must Specify Input" << std::endl;
    return 1;
  }

  // Read Input
  std::string in_file = argv[1];
  cmz::ed::Input_t input;
  cmz::ed::ReadInput( in_file, input );

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  size_t norb_eff = 
    cmz::ed::getParam<int>( input, "norbseff"  );
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );


  // Read-In FCI file
  size_t norb2 = norb * norb;
  size_t norb3 = norb * norb2;
  size_t norb4 = norb * norb3;
  std::vector<double> V_pqrs(norb4), T_pq(norb2);
  double core_energy = 0.;
  {
    std::ifstream ifile(fcidump);
    std::string line;
    while( std::getline( ifile, line ) ) {
      // Split the line
      std::stringstream ss(line);

      // Read the line
      uint64_t p,q,r,s;
      double matel;
      ss >> p >> q >> r >> s >> matel;
      if(!ss) throw "Bad Read";

      if( p > norb or q > norb or r > norb or s > norb )
        throw "Bad Orbital Index";


      // Zero Matrix Element
      if( std::abs(matel) < 
        std::numeric_limits<double>::epsilon() ) continue;

      // Read Core Energy
      if( p == 0 and q == 0 and r == 0 and s == 0 ) {
        core_energy = matel;
      } else if( r == 0 and s == 0 ) {
        T_pq[(p-1) + (q-1)*norb] = matel;
      } else {
        V_pqrs[(p-1) + (q-1)*norb + (r-1)*norb2 + (s-1)*norb3] = matel;
      }
    }
  }

  
  auto dets = build_hilbert_space<64>( norb_eff, nalpha, nbeta );

  MPI_Finalize();

}
