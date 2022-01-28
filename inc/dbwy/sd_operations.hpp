#pragma once
#include "bitset_operations.hpp"

namespace dbwy {

template <size_t N>
void generate_residues( std::bitset<N> state, std::vector<std::bitset<N>>& res ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  auto occ_alpha = bits_to_indices(state_alpha, occ_alpha);
  const int nalpha = occ_alpha.size();

  auto occ_beta = bits_to_indices(state_beta, occ_beta);
  const int nbeta  = occ_beta.size();

  std::bitset<N> state_alpha_full = expand_bitset<N>(state_alpha);
  std::bitset<N> state_beta_full  = expand_bitset<N>(state_beta); 
  state_beta_full = state_beta_full << (N/2);


  std::bitset<N/2> one = 1ul;

  // Double alpha
  for( auto i = 0;   i < nalpha; ++i ) 
  for( auto j = i+1; j < nalpha; ++j ) {
    auto mask = (one << occ_alpha[i]) | (one << occ_alpha[j]);
    std::bitset<N> _r = expand_bitset<N>(state_alpha & ~mask);
    res.emplace_back( _r | state_beta_full );
  }

  // Double beta
  for( auto i = 0;   i < nbeta; ++i ) 
  for( auto j = i+1; j < nbeta; ++j ) {
    auto mask = (one << occ_beta[i]) | (one << occ_beta[j]);
    std::bitset<N> _r = expand_bitset<N>(state_beta & ~mask) << (N/2);
    res.emplace_back( _r | state_alpha_full );
  }

  // Mixed
  for( auto i = 0; i < nalpha; ++i) 
  for( auto j = 0; j < nbeta;  ++j) {
    std::bitset<N> mask = expand_bitset<N>(one << occ_alpha[i]);
    mask = mask | (expand_bitset<N>(one << occ_beta[j]) << (N/2));
    res.emplace_back( state & ~mask );
  }

}

}
