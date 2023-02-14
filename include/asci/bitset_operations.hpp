#pragma once
#include <cassert>
#include <iostream>
#include <bit>
#include <climits>
#include <asci/types.hpp>

namespace asci {

inline auto clz( unsigned int i ) {
  return __builtin_clz(i);
}
inline auto clz( unsigned long int i ) {
  return __builtin_clzl(i);
}
inline auto clz( unsigned long long int i ) {
  return __builtin_clzll(i);
}
template <typename Integral>
std::enable_if_t<
  std::is_integral_v<Integral> and !std::is_signed_v<Integral>, 
  unsigned
> fls(Integral i) { 
  return CHAR_BIT * sizeof(Integral) - clz(i) - 1;
}


template <size_t N>
unsigned long long fast_to_ullong(const std::bitset<N>& bits) {
  // Low words
  if constexpr (N == 64 or N == 128)  return *reinterpret_cast<const uint64_t*>(&bits);
  if constexpr (N == 32)  return *reinterpret_cast<const uint32_t*>(&bits);
  return bits.to_ullong();
}

template <size_t N>
unsigned long fast_to_ulong(const std::bitset<N>& bits) {
  // Low words
  if constexpr (N == 32 or N == 64 or N == 128) return *reinterpret_cast<const uint32_t*>(&bits);
  return bits.to_ulong();
}

template <size_t N>
uint128_t to_uint128( std::bitset<N> bits ) {
  static_assert( N <= 128, "N > 128");
  if constexpr (N == 128) {
    auto _x = reinterpret_cast<uint128_t*>(&bits);
    return *_x;
  } else {
    return fast_to_ullong(bits);
  }
}

template <size_t N, size_t M = N>
std::bitset<M> full_mask() {
  static_assert( M >= N, "M < N" );
  std::bitset<M> mask(0ul);
  if constexpr (N == M/2) {
    if constexpr ( N == 64 ) {
      reinterpret_cast<uint64_t*>(&mask)[0] = UINT64_MAX;
    } else if constexpr ( N == 32 ) {
      reinterpret_cast<uint32_t*>(&mask)[0] = UINT32_MAX;
    } else mask = (~mask) >> (M-N);
    return mask;
  } else return (~mask) >> (M-N);
}

template <size_t N>
std::bitset<N> full_mask(size_t i) {
  assert( i <= N );
  std::bitset<N> mask(0ul);
  return (~mask) >> (N-i);
}
  
template <size_t N>
uint32_t ffs( std::bitset<N> bits ) {

  if constexpr (N <= 32)      return ffsl ( fast_to_ulong (bits) );
  else if constexpr (N <= 64) return ffsll( fast_to_ullong(bits) );
  else if constexpr (N <= 128) {
    //if(bits.any()) {
    //  return std::countr_zero( to_uint128(bits) ) + 1;
    //} else { return 0; }
    auto as_words = reinterpret_cast<uint64_t*>(&bits);
    if(as_words[0]) return ffsll(as_words[0]);
    else            return ffsll(as_words[1]) + 64;
  }
  else {
    uint32_t ind = 0;
    for( ind = 0; ind < N; ++ind )
    if( bits[ind] ) return (ind+1);
    return ind;
  }
  abort();

}

template <size_t N>
uint32_t fls( std::bitset<N> bits ) {
  if constexpr (N <= 32)      return fls( fast_to_ulong (bits) );
  else if constexpr (N <= 64) return fls( fast_to_ullong(bits) );
  else if constexpr (N <= 128) {
    auto as_words = reinterpret_cast<uint64_t*>(&bits);
    if(as_words[1]) return fls(as_words[1]) + 64;
    else            return fls(as_words[0]);
  }
  else {
    uint32_t ind = 0;
    for( ind = N-1; ind >= 0; ind-- )
    if( bits[ind] ) return ind;
    return ind;
  }
  abort();

}

template <size_t N>
void bits_to_indices( std::bitset<N> bits, std::vector<uint32_t>& indices ) {
  indices.clear();
#if 0
  for( auto i = 0ul; i < N; ++i )
  if( bits[i] ) indices.emplace_back(i);
#else
  auto c = bits.count();
  indices.resize(c);
  if( ! c ) return;
  for(int i = 0; i < c; ++i) {
    const auto ind = ffs(bits) - 1;
    bits.flip(ind);
    indices[i] = ind;
  }
#endif
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
inline std::bitset<N/2> bitset_lo_word( std::bitset<N> bits ) {
  static_assert( N == 128 or N == 64, "Not Supported");
  if constexpr (N == 128) {
    return std::bitset<64>(reinterpret_cast<uint64_t*>(&bits)[0]);
  }
  if constexpr (N == 64) {
    return std::bitset<32>(reinterpret_cast<uint32_t*>(&bits)[0]);
  }
}

template <size_t N>
inline std::bitset<N/2> bitset_hi_word( std::bitset<N> bits ) {
  static_assert( N == 128 or N == 64, "Not Supported");
  if constexpr (N == 128) {
    return std::bitset<64>(reinterpret_cast<uint64_t*>(&bits)[1]);
  }
  if constexpr (N == 64) {
    return std::bitset<32>(reinterpret_cast<uint32_t*>(&bits)[1]);
  }
}

template <size_t N>
bool bitset_less( std::bitset<N> x, std::bitset<N> y ) {
  if constexpr (N <= 32)      return fast_to_ulong (x) < fast_to_ulong (y);
  else if constexpr (N <= 64) return fast_to_ullong(x) < fast_to_ullong(y);
  else if constexpr (N == 128) {
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
