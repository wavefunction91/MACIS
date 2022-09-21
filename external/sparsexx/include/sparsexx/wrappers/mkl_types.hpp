#pragma once
#include <sparsexx/sparsexx_config.hpp>
#if SPARSEXX_ENABLE_MKL

#if __has_include(<mkl.h>)
  #include <mkl_spblas.h>
  #include <mkl_dss.h>
#else
  #error "FATAL: Cannot use sparsexx MKL bindings without MKL!"
#endif

namespace sparsexx::detail::mkl {
  using int_type = MKL_INT;
}
#endif
