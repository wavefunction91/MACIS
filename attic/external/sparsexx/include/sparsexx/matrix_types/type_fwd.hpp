#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace sparsexx {


template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
class csr_matrix;

template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
class csc_matrix;

template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
class coo_matrix;

}
