#pragma once
#include <asci/hamiltonian_generator.hpp>

namespace asci {

template <size_t N>
HamiltonianGenerator<N>::HamiltonianGenerator( 
  matrix_span<double> T, rank4_span_t V ) :
  norb_(T.extent(0)), norb2_(norb_*norb_), 
  norb3_(norb2_*norb_), T_pq_(T), V_pqrs_(V) { 

  generate_integral_intermediates(V_pqrs_); 

}


template <size_t N>
void HamiltonianGenerator<N>::
  generate_integral_intermediates(rank4_span_t V) {

  if(V.extent(0) != norb_ or V.extent(1) != norb_ or
     V.extent(2) != norb_ or V.extent(3) != norb_) 
    throw std::runtime_error("V has incorrect dimensions");

  size_t no  = norb_;
  size_t no2 = no  * no;
  size_t no3 = no2 * no;
  size_t no4 = no3 * no;

  // G(i,j,k,l) = V(i,j,k,l) - V(i,l,k,j)
  G_pqrs_data_ = std::vector<double>( begin(V), end(V) );
  G_pqrs_ = 
    rank4_span_t(G_pqrs_data_.data(),no,no,no,no);
  for( auto i = 0ul; i < no; ++i )
  for( auto j = 0ul; j < no; ++j )
  for( auto k = 0ul; k < no; ++k )
  for( auto l = 0ul; l < no; ++l ) {
    G_pqrs_(i,j,k,l) -= V(i,l,k,j);
  }

  // G_red(i,j,k) = G(i,j,k,k) = G(k,k,i,j)
  // V_red(i,j,k) = V(i,j,k,k) = V(k,k,i,j)
  G_red_data_.resize(no3);
  V_red_data_.resize(no3);
  G_red_ = rank3_span_t(G_red_data_.data(),no,no,no);
  V_red_ = rank3_span_t(V_red_data_.data(),no,no,no);
  for( auto j = 0ul; j < no; ++j ) 
  for( auto i = 0ul; i < no; ++i )
  for( auto k = 0ul; k < no; ++k ) {
    G_red_(k,i,j) = G_pqrs_(k,k,i,j);
    V_red_(k,i,j) = V(k,k,i,j);
  }

  // G2_red(i,j) = 0.5 * G(i,i,j,j)
  // V2_red(i,j) = V(i,i,j,j)
  G2_red_data_.resize(no2);
  V2_red_data_.resize(no2);
  G2_red_ = matrix_span<double>(G2_red_data_.data(),no,no);
  V2_red_ = matrix_span<double>(V2_red_data_.data(),no,no);
  for( auto j = 0ul; j < no; ++j ) 
  for( auto i = 0ul; i < no; ++i ) {
    G2_red_(i,j) = 0.5 * G_pqrs_(i,i,j,j);
    V2_red_(i,j) = V(i,i,j,j);
  }


}

}


#include <asci/hamiltonian_generator/matrix_elements.hpp>
#include <asci/hamiltonian_generator/rdms.hpp>
#include <asci/hamiltonian_generator/fast_diagonals.hpp>
