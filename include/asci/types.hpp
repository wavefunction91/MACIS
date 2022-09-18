#pragma once
#include <bitset>
#include <bit>
#include <vector>
#include <strings.h>
#include <experimental/mdspan>

namespace asci {

namespace stdex = std::experimental;

template <typename T, size_t rank>
using col_major_span = 
  stdex::mdspan<T, stdex::dextents<size_t,rank>, 
    stdex::layout_left>;

template <size_t N>
using wavefunction_iterator_t = typename std::vector< std::bitset<N> >::iterator;

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
