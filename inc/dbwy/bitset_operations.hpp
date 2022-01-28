#pragma once
#include <bitset>
#include <cassert>

namespace dbwy {

template <size_t N, size_t M = N>
std::bitset<M> full_mask() {
  static_assert( M >= N, "M < N" );
  std::bitset<M> mask(0ul);
  return (~mask) >> (M-N);
}

template <size_t N>
std::bitset<N> full_mask(size_t i) {
  assert( i <= N );
  std::bitset<N> mask(0ul);
  return (~mask) >> (N-i);
}
  
template <size_t N>
uint32_t ffs( std::bitset<N> bits ) {

  if constexpr (N <= 32)      return ffsl( bits.to_ulong() );
  else if constexpr (N <= 64) return ffsll( bits.to_ullong() );
  else {
    uint32_t ind = 0;
    for( ind = 0; ind < N; ++ind )
    if( bits[ind] ) return (ind+1);
    return ind;
  }
  abort();

}

template <size_t N>
void bits_to_indices( std::bitset<N> bits, std::vector<uint32_t>& indices ) {
  indices.clear();
  for( auto i = 0; i < N; ++i )
  if( bits[i] ) indices.emplace_back(i);
}

template <size_t N>
std::vector<uint32_t> bits_to_indices( std::bitset<N> bits ) {
  std::vector<uint32_t> indices;
  bits_to_indices( bits, indices );
  return indices;
}


template <size_t N, size_t M>
inline std::bitset<N> truncate_bitset( std::bitset<M> bits ) {
  static_assert( M >= N, "M < N" );
  if constexpr ( M == N ) return bits;
  
  const auto mask = full_mask<N,M>();
  if constexpr ( N <= 32 ) {
    return ( bits & mask ).to_ulong();
  } else if constexpr ( N <= 64 ) {
    return ( bits & mask ).to_ullong();
  } else {
    std::bitset<N> trunc_bits = 0;
    for( size_t i = 0; i < N; ++i ) 
    if( bits[i] ) trunc_bits[i] = 1; 
    return trunc_bits;
  }
}

template <size_t N, size_t M>
inline std::bitset<N> expand_bitset( std::bitset<M> bits ) {
  static_assert( N >= M, "N < M" );
  if constexpr (M == N) return bits;

  if constexpr ( M <= 32 ) {
    return bits.to_ulong();
  } else if constexpr ( M <= 64 ) {
    return bits.to_ullong();
  } else {
    std::bitset<N> exp_bits = 0;
    for( size_t i = 0; i < M; ++i )
    if( bits[i] ) exp_bits[i] = 1;
    return exp_bits;
  }
}

template <size_t N>
bool bitset_less( std::bitset<N> x, std::bitset<N> y ) {
  static_assert( N <= 64, "N > 64");
  if constexpr (N <= 32) return x.to_ulong() < y.to_ulong();
  else if constexpr (N <= 64) return x.to_ullong() < y.to_ullong();
  abort();
}


}
