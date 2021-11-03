#include <sparsexx/util/mpi.hpp>
#include <sparsexx/util/graph.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/io/read_mm.hpp>

#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <random>

template <typename T>
auto operator-(const std::vector<T>& a, const std::vector<T>& b ) {
  if( a.size() != b.size() ) throw std::runtime_error("");
  std::vector<T> c(a.size());
  for( auto i = 0; i < a.size(); ++i ) {
    c[i] = std::abs(a[i] - b[i]);
  }
  return c;
}

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;

#include <functional>

template <typename T>
std::size_t hash_combine( std::size_t seed, const T& x ) {
  return seed ^ (std::hash<T>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename T, typename... Args>
std::size_t hash_combine( std::size_t seed, const T& x, Args&&... args ) {
  return hash_combine( hash_combine(seed,x), std::forward<Args>(args)... );
}

template <typename T>
struct std::hash<std::vector<T>> {
  std::size_t operator()( const std::vector<T>& v ) const noexcept {
    std::size_t seed = v.size();
    for( auto i : v ) seed = hash_combine(seed, i); 
    return seed;
  }
};

template <typename T, typename I>
struct std::hash< sparsexx::csr_matrix<T,I> > {
  std::size_t operator()( const sparsexx::csr_matrix<T,I>& A ) const noexcept {
    return hash_combine( A.m(), A.n(), A.nnz(), A.rowptr(), A.colind(), A.nzval() );
  };
};


int main(int argc, char** argv) {
  MPI_Init( &argc, &argv );
  auto world_size = sparsexx::detail::get_mpi_size( MPI_COMM_WORLD );
  auto world_rank = sparsexx::detail::get_mpi_rank( MPI_COMM_WORLD );
  {
  assert( argc == 2 );
  using spmat_type = sparsexx::csr_matrix<double, int32_t>;
  #if 1
  if(world_rank == 0) std::cout << "READING MATRIX" << std::endl;
  auto A = sparsexx::read_mm<spmat_type>( std::string( argv[1] ) );
  if(world_rank == 0) std::cout << "DONE READING MATRIX" << std::endl;
  #else
  // [ x x 0 0 x x 0 0 x ]
  // [ x x 0 0 x x 0 0 x ]
  // [ 0 0 x x 0 0 x x x ]
  // [ 0 0 x x 0 0 x x x ]
  // [ x x 0 0 x x 0 0 0 ]
  // [ x x 0 0 x x 0 0 0 ]
  // [ 0 0 x x 0 0 x x 0 ]
  // [ 0 0 x x 0 0 x x 0 ]
  // [ x x x x 0 0 0 0 x ]
  spmat_type A(9,9,41,0);
  A.rowptr() = { 0, 5, 10, 15, 20, 24, 28, 32, 36, 41 };
  A.colind() = {
    0, 1, 4, 5, 8,
    0, 1, 4, 5, 8,
    2, 3, 6, 7, 8,
    2, 3, 6, 7, 8,
    0, 1, 4, 5,
    0, 1, 4, 5,
    2, 3, 6, 7,
    2, 3, 6, 7,
    1, 2, 3, 4, 8 
  };

  //std::fill(A.nzval().begin(), A.nzval().end(), 1. );
  std::iota(A.nzval().begin(), A.nzval().end(), 1. );
  /*
  A.nzval() = {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9, 9 
  };
  */
  #endif
  const int N = A.m();

  const bool do_reorder = true;
  std::vector<int32_t> mat_perm;
  if( do_reorder ) {

    int nparts = std::max(2l,world_size);
    // Reorder on root rank
    if( world_rank == 0 ) {
      // Partition the graph of A (assumes symmetric)
      auto kway_part_begin = clock_type::now(); 
      auto part = sparsexx::kway_partition( nparts, A );
      auto kway_part_end = clock_type::now(); 

      // Form permutation from partition
      std::tie( mat_perm, std::ignore)  = sparsexx::perm_from_part( nparts, part );

      // Permute rows/cols of A 
      // A(I,P(J)) = A(P(I),J)
      auto permute_begin = clock_type::now(); 
      A = sparsexx::permute_rows_cols( A, mat_perm, mat_perm );
      auto permute_end = clock_type::now(); 

      duration_type kway_part_dur = kway_part_end - kway_part_begin;
      duration_type permute_dur   = permute_end - permute_begin;

      std::cout << "KWAY PART DUR = " << kway_part_dur.count() << std::endl;
      std::cout << "PERMUTE DUR   = " << permute_dur.count() << std::endl;
    } else {
      mat_perm.resize(N);
    }

    // Broadcast reordered data
    if( world_size > 1 ) {
      using index_type = sparsexx::detail::index_type_t<spmat_type>;
      using value_type = sparsexx::detail::value_type_t<spmat_type>;
      MPI_Bcast( A.rowptr().data(), A.rowptr().size() * sizeof(index_type),
                 MPI_BYTE, 0, MPI_COMM_WORLD );
      MPI_Bcast( A.colind().data(), A.colind().size() * sizeof(index_type),
                 MPI_BYTE, 0, MPI_COMM_WORLD );
      MPI_Bcast( A.nzval().data(), A.nzval().size() * sizeof(value_type),
                 MPI_BYTE, 0, MPI_COMM_WORLD );

      MPI_Bcast( mat_perm.data(), mat_perm.size() * sizeof(int32_t),
                 MPI_BYTE, 0, MPI_COMM_WORLD );
    }
  }

  // Get distributed matrix 
  sparsexx::dist_sparse_matrix<spmat_type> A_dist( MPI_COMM_WORLD, A );
  auto spmv_info = sparsexx::spblas::generate_spmv_comm_info( A_dist );

  size_t comm_vol = spmv_info.communication_volume();
  if( world_rank == 0 ) std::cout << "COMM VOLUME = " << comm_vol << std::endl;

  


  // Serial SPMV
  std::vector<double> V(N), AV(N);
  //std::iota( V.begin(), V.end(), 0 );
  //for( auto& x : V ) x *= 0.01;
  std::minstd_rand gen;
  std::uniform_real_distribution<> dist(-1,1);
  std::generate( V.begin(), V.end(), [&](){ return dist(gen); } );
  if( mat_perm.size() ) {
    sparsexx::permute_vector( N, V.data(), mat_perm.data(),
      sparsexx::PermuteDirection::Backward );
  }

  auto spmv_st = clock_type::now();
  sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N );
  auto spmv_en = clock_type::now();
  if( mat_perm.size() ) {
    sparsexx::permute_vector( N, AV.data(), mat_perm.data(), 
      sparsexx::PermuteDirection::Forward );
  }

  // Parallel SPMV
  std::vector<double> V_dist ( A_dist.local_row_extent() ),
                      AV_dist( A_dist.local_row_extent() );
  auto [dist_row_st, dist_row_en] = A_dist.row_bounds( world_rank );
  for( auto i = dist_row_st; i < dist_row_en; ++i ) {
    V_dist[i-dist_row_st] = V[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto pspmv_st = clock_type::now();

  sparsexx::spblas::pgespmv( 1., A_dist, V_dist.data(), 0., AV_dist.data(), 
    spmv_info );

  MPI_Barrier(MPI_COMM_WORLD);
  auto pspmv_en = clock_type::now();

  if( !world_rank ) {
    duration_type spmv_dur  = spmv_en - spmv_st;
    duration_type pspmv_dur = pspmv_en - pspmv_st;
    std::cout << "SERIAL   = " << spmv_dur.count() << std::endl;
    std::cout << "PARALLEL = " << pspmv_dur.count() << std::endl;
  }

  // Compare results
  std::vector<double> AV_dist_combine(N);
  size_t n_per_rank = N / world_size;
  MPI_Allgather( AV_dist.data(), n_per_rank, MPI_DOUBLE, AV_dist_combine.data(),
    n_per_rank, MPI_DOUBLE, MPI_COMM_WORLD );
  if( N % world_size and world_size > 1 ) {
    if( world_rank == (world_size-1) ) {
      std::copy_n( AV_dist.data() + n_per_rank, N % world_size, 
        AV_dist_combine.data() + world_size*n_per_rank );
    }
    MPI_Bcast( AV_dist_combine.data() + world_size*n_per_rank, N%world_size,
      MPI_DOUBLE, world_size-1, MPI_COMM_WORLD );
  }
  if( mat_perm.size() ) {
    sparsexx::permute_vector( N, AV_dist_combine.data(), mat_perm.data(), 
      sparsexx::PermuteDirection::Forward );
  }

  double max_diff = 0.;
  double nrm = 0.;
  for( auto i = 0; i < N; ++i ) {
    auto diff = std::abs(AV_dist_combine[i] - AV[i]);
    max_diff = std::max( max_diff, diff );
    nrm += diff*diff;
  }
  nrm = std::sqrt(nrm);

  if( !world_rank ) {
    std::cout << "MAX DIFF = " << max_diff << std::endl;
    std::cout << "NRM DIFF = " << nrm << std::endl;
  }
  

  }
  MPI_Finalize();
}
