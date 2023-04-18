#pragma once
#include <mpi.h>
#include <memory>
#include <iostream>
#include <bitset>

namespace asci {

namespace detail {
struct mpi_datatype_impl {
  MPI_Datatype dtype;
  mpi_datatype_impl() = delete;
  mpi_datatype_impl(MPI_Datatype d) : dtype(d) {}

  virtual ~mpi_datatype_impl() noexcept = default;
};

struct managed_mpi_datatype_impl : public mpi_datatype_impl {
  template <typename... Args>
  managed_mpi_datatype_impl(Args&&... args) :
    mpi_datatype_impl( std::forward<Args>(args)... ) {}

  ~managed_mpi_datatype_impl() noexcept {
    MPI_Type_free( &dtype );
  }
};
}

inline int comm_rank(MPI_Comm comm) {
  int rank; MPI_Comm_rank(comm, &rank);
  return rank;
}

inline int comm_size(MPI_Comm comm) {
  int size; MPI_Comm_size(comm, &size);
  return size;
}


class mpi_datatype {
public:
  using pimpl_type = detail::mpi_datatype_impl;
  using pimpl_pointer_type = std::unique_ptr<pimpl_type>;
  mpi_datatype( pimpl_pointer_type&& p ) : pimpl_(std::move(p)) {}

  inline operator MPI_Datatype() const { return pimpl_->dtype; }
private:
  pimpl_pointer_type pimpl_;
};

template <typename... Args>
inline mpi_datatype make_managed_mpi_datatype(Args&&... args) {
  return mpi_datatype( 
    std::make_unique<detail::managed_mpi_datatype_impl>(std::forward<Args>(args)...)
  );
}

template <typename... Args>
inline mpi_datatype make_mpi_datatype(Args&&... args) {
  return mpi_datatype( 
    std::make_unique<detail::mpi_datatype_impl>(std::forward<Args>(args)...)
  );
}


template <typename T>
struct mpi_traits;

#define REGISTER_MPI_TYPE(T, TYPE) \
template <> \
struct mpi_traits<T> { \
  using type = T; \
  inline static mpi_datatype datatype() { return make_mpi_datatype(TYPE); }\
};


REGISTER_MPI_TYPE(char,   MPI_CHAR    );
REGISTER_MPI_TYPE(int,    MPI_INT     );
REGISTER_MPI_TYPE(double, MPI_DOUBLE  );
REGISTER_MPI_TYPE(float,  MPI_FLOAT   );
REGISTER_MPI_TYPE(size_t, MPI_UINT64_T);

#undef REGISTER_MPI_TYPE

template <typename T> 
mpi_datatype make_contiguous_mpi_datatype(int n) {
  auto dtype = mpi_traits<T>::datatype();
  MPI_Datatype contig_dtype;
  MPI_Type_contiguous( n, dtype, &contig_dtype );
  MPI_Type_commit( &contig_dtype );
  return make_managed_mpi_datatype( contig_dtype );
}


template <typename T>
void allreduce( const T* send, T* recv, size_t count, MPI_Op op, 
  MPI_Comm comm ) {

  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  for(int i = 0; i < nchunk; ++i) {
    MPI_Allreduce( send + i*intmax, recv + i*intmax, intmax, 
      dtype, op, comm );
  }

  int nrem = count % intmax;
  if(nrem) {
    MPI_Allreduce( send + nchunk*intmax, recv + nchunk*intmax, nrem,
      dtype, op, comm);
  }

}

template <typename T>
void allreduce( T* recv, size_t count, MPI_Op op, MPI_Comm comm ) {

  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  for(int i = 0; i < nchunk; ++i) {
    MPI_Allreduce( MPI_IN_PLACE, recv + i*intmax, intmax, 
      dtype, op, comm );
  }

  int nrem = count % intmax;
  if(nrem) {
    MPI_Allreduce( MPI_IN_PLACE, recv + nchunk*intmax, nrem,
      dtype, op, comm);
  }

}

template <typename T>
T allreduce( const T& d, MPI_Op op, MPI_Comm comm ) {
  T r;
  allreduce( &d, &r, 1, op, comm );
  return r;
}

template <typename T>
void bcast( T* buffer, size_t count, int root, MPI_Comm comm ) {

  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  for(int i = 0; i < nchunk; ++i) {
    MPI_Bcast( buffer + i*intmax, intmax, dtype, root, comm );
  }

  int nrem = count % intmax;
  if(nrem) {
    MPI_Bcast( buffer + nchunk*intmax, nrem, dtype, root, comm );
  }

}




// Generate MPI types
template <size_t N>
struct mpi_traits<std::bitset<N>> {
  using type = std::bitset<N>;
  inline static mpi_datatype datatype() { 
    return make_contiguous_mpi_datatype<char>(sizeof(type));
  }
};


}
