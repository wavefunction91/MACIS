#pragma once
#include <mpi.h>
#include <memory>

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

template <typename T> mpi_datatype mpi_dtype();
#define REGISTER_MPI_TYPE(T, TYPE) \
template <> inline mpi_datatype mpi_dtype<T>(){ \
  return make_mpi_datatype(TYPE); \
}

REGISTER_MPI_TYPE(int,    MPI_INT   );
REGISTER_MPI_TYPE(double, MPI_DOUBLE);

#undef REGISTER_MPI_TYPE

template <typename T> 
mpi_datatype make_contiguous_mpi_datatype(int n) {
  auto dtype = mpi_dtype<T>();
  MPI_Datatype contig_dtype;
  MPI_Type_contiguous( n, dtype, &contig_dtype );
  MPI_Type_commit( &contig_dtype );
  return make_managed_mpi_datatype( contig_dtype );
}

}
