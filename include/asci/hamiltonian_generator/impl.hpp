#pragma once
#include <asci/hamiltonian_generator.hpp>

namespace asci {

template <size_t N>
HamiltonianGenerator<N>::HamiltonianGenerator( size_t no, double* V, double* T ) :
  norb_(no), norb2_(no*no), norb3_(no*no*no),
  V_pqrs_(V), T_pq_(T) {

  generate_integral_intermediates(no, V_pqrs_);

}


template <size_t N>
void HamiltonianGenerator<N>::generate_integral_intermediates(
  size_t no, const double* V) {

  size_t no2 = no  * no;
  size_t no3 = no2 * no;
  size_t no4 = no3 * no;

  // G(i,j,k,l) = V(i,j,k,l) - V(i,l,k,j)
  G_pqrs_ = std::vector<double>( V, V + no4 );
  for( auto i = 0ul; i < no; ++i )
  for( auto j = 0ul; j < no; ++j )
  for( auto k = 0ul; k < no; ++k )
  for( auto l = 0ul; l < no; ++l ) {
    G_pqrs_[i + j*no + k*no2 + l*no3] -= V[i + l*no + k*no2 + j*no3];
  }

  // G_red(i,j,k) = G(i,j,k,k) = G(k,k,i,j)
  // V_red(i,j,k) = V(i,j,k,k) = V(k,k,i,j)
  G_red_.resize(no3);
  V_red_.resize(no3);
  for( auto j = 0ul; j < no; ++j ) 
  for( auto i = 0ul; i < no; ++i )
  for( auto k = 0ul; k < no; ++k ) {
    G_red_[k + i*no + j*no2 ] = G_pqrs_[k*(no+1) + i*no2 + j*no3];
    V_red_[k + i*no + j*no2 ] = V      [k*(no+1) + i*no2 + j*no3];
  }

  // G2_red(i,j) = 0.5 * G(i,i,j,j)
  // V2_red(i,j) = V(i,i,j,j)
  G2_red_.resize(no2);
  V2_red_.resize(no2);
  for( auto j = 0ul; j < no; ++j ) 
  for( auto i = 0ul; i < no; ++i ) {
    G2_red_[i + j*no] = 0.5 * G_pqrs_[i*(no+1) + j*(no2+no3)];
    V2_red_[i + j*no] = V[i*(no+1) + j*(no2+no3)];
  }


}

}


#include <asci/hamiltonian_generator/matrix_elements.hpp>
#include <asci/hamiltonian_generator/rdms.hpp>
#include <asci/hamiltonian_generator/fast_diagonals.hpp>
