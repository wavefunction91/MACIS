#include "ut_common.hpp"
#include <asci/util/topk_parallel.hpp>

TEST_CASE("TopK - POD") {

  // MPI Info
  const auto world_size = asci::comm_size(MPI_COMM_WORLD);
  const auto world_rank = asci::comm_rank(MPI_COMM_WORLD);

  constexpr int Chunk = 10;
  int N, K;
  SECTION("Chunk = N = K") {
    K = Chunk;
    N = K;
  }

  SECTION("Chunk = K, N > K, N % K == 0") {
    K = Chunk;
    N = 2*K;
  }

  SECTION("Chunk = K, N > K, N % K != 0") {
    K = Chunk;
    N = K + Chunk - 1;
  }

  SECTION("Chunk = K, N < K") {
    if(world_size == 1) return;
    K = Chunk;
    N = K - 2;
  }

  SECTION("Chunk < K, K % Chunk == 0") {
    K = 2*Chunk;
    N = K;
  }
  SECTION("Chunk < K, K % Chunk != 0") {
    K = Chunk + Chunk/2;
    N = K;
  }
  SECTION("Chunk > K") {
    K = Chunk/2;
    N = K;
  }

  std::greater<int> comparator;

  std::vector<int> data_local(N), topk(K);

  // Generate Random Permutation of MPI Ranks
  std::vector<int> rank_shuffle(world_size);
  std::iota(rank_shuffle.begin(), rank_shuffle.end(), 0);

  std::random_device r;
  int seed = r();
  MPI_Bcast( &seed, 1, MPI_INT, 0, MPI_COMM_WORLD );
  std::default_random_engine g(seed);
  std::shuffle(rank_shuffle.begin(), rank_shuffle.end(), g);

  // Generate Unique Global Data
  std::iota( data_local.begin(), data_local.end(), world_rank * N );

  // Do TOP-K reduction
  asci::topk_allreduce<Chunk>(data_local.data(), 
    data_local.data() + data_local.size(), K, topk.data(),
    comparator, MPI_COMM_WORLD);

  // Sort topk elements
  std::sort(topk.begin(), topk.end(), comparator);

  // Check elements
  for(int i = 0; i < K; ++i) {
    REQUIRE(topk[i] == world_size*N - i - 1);
  }

}

struct my_ranking_pair {
  int id;
  double score;
};

namespace asci {

template <> 
struct mpi_traits<my_ranking_pair> {
  using type = my_ranking_pair;
  inline static mpi_datatype datatype() {
  
    type dummy;
  
    int lengths[2] = {1,1};
    MPI_Aint displacements[2];
    MPI_Aint base_address;
    MPI_Get_address(&dummy,       &base_address);
    MPI_Get_address(&dummy.id,    displacements + 0);
    MPI_Get_address(&dummy.score, displacements + 1);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Datatype custom_type;
    MPI_Type_create_struct( 2, lengths, displacements, types, &custom_type );
    MPI_Type_commit( &custom_type );
  
    return make_managed_mpi_datatype( custom_type );
  
  }
};

}

bool id_comparator( my_ranking_pair a, my_ranking_pair b) {
  return a.id < b.id;
}

struct score_comparator {
  using type = my_ranking_pair;
  constexpr bool operator()( const type& a, const type& b) const {
    return a.score > b.score;
  }
};

TEST_CASE("TopK - Metadata") {

  // MPI Info
  const auto world_size = asci::comm_size(MPI_COMM_WORLD);
  const auto world_rank = asci::comm_rank(MPI_COMM_WORLD);

  constexpr int Chunk = 10;
  int N = 100;
  int K = world_size;

  // Generate Local Data
  std::vector<my_ranking_pair> data_local(N);
  for(int i = 0; i < N; ++i) {
    data_local[i] = my_ranking_pair{ world_rank*N + i, world_rank*N + i };
  }

  for(int i = 0; i < N/2; ++i) {
    data_local[i].score = 0;
  }

  std::vector<my_ranking_pair> topk(K);
  asci::topk_allreduce<Chunk>(data_local.data(), 
    data_local.data() + data_local.size(), K, topk.data(),
    score_comparator(), MPI_COMM_WORLD);

  std::sort(topk.begin(), topk.end(), score_comparator());

  for( auto i = 0; i < K; ++i ) {
    REQUIRE( topk[i].score == double(world_size*N - i - 1) );
  }
  
}

