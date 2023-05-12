#pragma once
#include <bitset>
#include <bit>
#include <vector>
#include <strings.h>
#include <mdspan/mdspan.hpp>

namespace macis {

namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

template <typename T, size_t rank>
using col_major_span = 
  Kokkos::mdspan<T, Kokkos::dextents<size_t,rank>, 
    Kokkos::layout_left>;

template <typename T>
using matrix_span = col_major_span<T,2>;

template <typename T>
using rank3_span = col_major_span<T,3>;

template <typename T>
using rank4_span = col_major_span<T,4>;

template <typename T>
auto begin(T&& s){
  return s.data_handle();
}

template <typename T>
auto end(T&& s){
  return begin(s) + s.size();
}

template <size_t N>
using wfn_t = std::bitset<N>;

template <size_t N>
using wavefunction_iterator_t = 
  typename std::vector< std::bitset<N> >::iterator;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
using uint128_t = unsigned __int128;
#pragma GCC diagnostic pop



template <typename T, typename ParameterType>
class NamedType {

public:

  constexpr explicit NamedType() : value_() { }
  constexpr explicit NamedType(T const& value) : value_(value) {}
  constexpr explicit NamedType(T&& value) : value_(std::move(value)) {}

  constexpr NamedType( const NamedType& other ) : value_(other.get()) { }
  constexpr NamedType( NamedType&& other ) noexcept : 
    value_(std::move(other.get())) { };

  constexpr NamedType& operator=( const NamedType& other ) {
    value_ = other.get();
    return *this;
  }
  constexpr NamedType& operator=( NamedType&& other ) noexcept {
    value_ = std::move(other.get());
    return *this;
  }

  constexpr T& get() { return value_; }
  constexpr T const& get() const {return value_; }

private:

  T value_;

};

using NumElectron = NamedType<size_t, struct nelec_type>;
using NumOrbital  = NamedType<size_t, struct norb_type>;
using NumActive   = NamedType<size_t, struct nactive_type>;
using NumInactive = NamedType<size_t, struct ninactive_type>;
using NumVirtual  = NamedType<size_t, struct nvirtual_type>;

}
