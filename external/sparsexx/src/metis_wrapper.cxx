#include <sparsexx/util/graph.hpp>
#include <sparsexx/sparsexx_config.hpp>

#ifdef SPARSEXX_ENABLE_METIS
#include <metis.h>
#endif

namespace sparsexx::detail {

template <typename IndexType>
void metis_kway_partitioning(int64_t _nvert, int64_t _npart, IndexType* _xadj, 
  IndexType* _adjncy, std::vector<IndexType>& _part) {

#ifdef SPARSEXX_ENABLE_METIS
  idx_t nvert    = _nvert;
  idx_t nweights = 1;
  idx_t nparts   = _npart;
  idx_t obj;

  std::vector<idx_t> xadj_local, adjncy_local, part_local;
  idx_t *xadj, *adjncy, *part;
  if constexpr (std::is_same_v<IndexType, idx_t>) {
    xadj   = _xadj;
    adjncy = _adjncy;
    part   = _part.data();
  } else {
    throw std::runtime_error("METIS + Index Mismatch NYI");
  }

  METIS_PartGraphKway( &nvert, &ndeights, xadj, adjncy, NULL, NULL, NULL
    &nparts, NULL, NULL, NULL, &obj, part );
#else
  throw std::runtime_error("METIS Not Enabled");
#endif

}

#define IMPL(T) \
template \
void metis_kway_partitioning(int64_t _nvert, int64_t _npart, T* _xadj,\
  T* _adjncy, std::vector<T>& _part);

IMPL(int32_t);
IMPL(int64_t);
  

}
