#pragma once

#include <mpi.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <typeinfo>

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

#define REGISTER_MPI_STATIC_TYPE(TYPE, MPI_TYPE)\
template <>                                     \
struct mpi_data<TYPE>{                          \
  inline static constexpr auto type = MPI_TYPE; \
};

REGISTER_MPI_STATIC_TYPE( double,   MPI_DOUBLE   )
REGISTER_MPI_STATIC_TYPE( int64_t,  MPI_INT64_T  )
REGISTER_MPI_STATIC_TYPE( uint64_t, MPI_UINT64_T )
REGISTER_MPI_STATIC_TYPE( int,      MPI_INT  )
REGISTER_MPI_STATIC_TYPE( unsigned, MPI_UNSIGNED )

#undef REGISTER_MPI_STATIC_TYPE

template <typename T>
inline static constexpr auto mpi_data_t = mpi_data<T>::type;

template <typename T>
std::vector<T> mpi_allgather( const std::vector<T>& data, MPI_Comm comm ) {
  const size_t count = data.size();
  const auto   comm_size = get_mpi_size(comm);
  std::vector<T> gathered_data( count * comm_size );

  MPI_Allgather( data.data(), count, mpi_data_t<T>, gathered_data.data(),
    count, mpi_data_t<T>, comm );

  return gathered_data;
}

template <typename T>
MPI_Request mpi_irecv( T* data, size_t count, int source_rank, int tag,
  MPI_Comm comm ) {

  MPI_Request req;
  MPI_Irecv( data, count, mpi_data_t<T>, source_rank, tag, comm, &req );
  return req;

}

template <typename T>
MPI_Request mpi_irecv( std::vector<T>& data, int source_rank, int tag, 
  MPI_Comm comm ) {
  return mpi_irecv( data.data(), data.size(), source_rank, tag, comm );
}

template <typename T>
MPI_Request mpi_isend( const T* data, size_t count, int dest_rank, int tag,
  MPI_Comm comm ) {

  MPI_Request req;
  MPI_Isend( data, count, mpi_data_t<T>, dest_rank, tag, comm, &req );
  return req;

}

template <typename T>
MPI_Request mpi_isend( const std::vector<T>& data, int dest_rank, int tag, 
  MPI_Comm comm ) {
  return mpi_isend( data.data(), data.size(), dest_rank, tag, comm );
}


inline void mpi_waitall_ignore_status( std::vector<MPI_Request>& requests ) {
  MPI_Waitall( requests.size(), requests.data(), MPI_STATUSES_IGNORE );
}

}
