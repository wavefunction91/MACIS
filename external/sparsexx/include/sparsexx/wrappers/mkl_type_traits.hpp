#pragma once
#if SPARSEXX_ENABLE_MKL

#include "mkl_types.hpp"
#include <sparsexx/matrix_types/type_traits.hpp>
#include <type_traits>
#include <complex>

namespace sparsexx::detail::mkl {

template <typename T>
struct is_mkl_type_supported : public std::false_type {};

template<>
struct is_mkl_type_supported< float > : public std::true_type {};
template<>
struct is_mkl_type_supported< double > : public std::true_type {};
template<>
struct is_mkl_type_supported< std::complex<float> > : public std::true_type {};
template<>
struct is_mkl_type_supported< std::complex<double> > : public std::true_type {};

template <typename T>
inline constexpr bool is_mkl_type_supported_v = is_mkl_type_supported<T>::value;

template <typename T, typename U = void>
struct enable_if_mkl_type_supported : 
  public std::enable_if< is_mkl_type_supported_v<T>, U > {

  using type = typename std::enable_if< is_mkl_type_supported_v<T>, U >::type;

};

template <typename T, typename U = void>
using enable_if_mkl_type_supported_t = typename enable_if_mkl_type_supported<T,U>::type;


template <typename SpMatType, typename = void>
struct is_mkl_compatible : public std::false_type {};

template <typename SpMatType>
struct is_mkl_compatible< SpMatType,
  std::enable_if_t< 
    std::is_same_v< typename SpMatType::index_type, int_type > and
    is_mkl_type_supported_v< typename SpMatType::value_type > and
    ( // Supported matrix types
      is_csr_matrix_v<SpMatType>
      or is_coo_matrix_v<SpMatType>
      // or is_csc_matrix_v<SpMatType>
    )
    >
  > : public std::true_type {};

template <typename SpMatType>
inline constexpr bool is_mkl_compatible_v = is_mkl_compatible<SpMatType>::value; 

template <typename SpMatType, typename U = void>
struct enable_if_mkl_compatible : 
  public std::enable_if< is_mkl_compatible_v<SpMatType>, U > {

  using type = typename std::enable_if< is_mkl_compatible_v<SpMatType>, U >::type;

};

template <typename SpMatType, typename U = void>
using enable_if_mkl_compatible_t = typename enable_if_mkl_compatible<SpMatType,U>::type;

}
#endif
