#pragma once
#include <bitset>
#include <bit>
#include <vector>

namespace asci {

template <size_t N>
using wavefunction_iterator_t = typename std::vector< std::bitset<N> >::iterator;

using uint128_t = unsigned __int128;

}
