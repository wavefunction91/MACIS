#pragma once
#include <bitset>
#include <bit>
#include <vector>
#include <strings.h>

namespace asci {

template <size_t N>
using wavefunction_iterator_t = typename std::vector< std::bitset<N> >::iterator;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
using uint128_t = unsigned __int128;
#pragma GCC diagnostic pop

}
