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
  #if 1
  else if constexpr ( N%64 == 0 ) {
    const auto mask = full_mask<64,N>();
    constexpr auto nchunks = N/64;
    #pragma GCC unroll 2
    for( size_t i = 0; i < nchunks; ++i ) {
      const uint64_t chunk = ((bits >> (i*64)) & mask).to_ullong();
      if( chunk ) return ffsll(chunk);
    }
    return 0;
  }
  #endif
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
  if constexpr (N <= 32) return x.to_ulong() < y.to_ulong();
  else if constexpr (N <= 64) return x.to_ullong() < y.to_ullong();
  else if constexpr (N == 128) {
    typedef unsigned __int128 uint128_t;
    auto _x = reinterpret_cast<uint128_t*>(&x);
    auto _y = reinterpret_cast<uint128_t*>(&y);
    return *_x < *_y;
  } 
  else {
    for (int i = N-1; i >= 0; i--) {
      if (x[i] ^ y[i]) return y[i];
    }
    return false;
  }
  abort();
}


template <size_t N>
struct bitset_less_comparator {
  bool operator()( std::bitset<N> x, std::bitset<N> y ) const {
    return bitset_less(x,y);
  }
};
}
