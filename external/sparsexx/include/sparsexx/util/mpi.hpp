#pragma once

#include <mpi.h>

namespace sparsexx::detail {


static inline int64_t get_mpi_rank( MPI_Comm c ) {
  int rank;
  MPI_Comm_rank( c, &rank );
  return rank;
}


static inline int64_t get_mpi_size( MPI_Comm c ) {
  int size;
  MPI_Comm_size( c, &size );
  return size;
}


template <typename T>
struct mpi_data;

template <>
struct mpi_data<double> {
  inline static constexpr auto type = MPI_DOUBLE;
};

template <typename T>
inline static constexpr auto mpi_data_t = mpi_data<double>::type;

}
