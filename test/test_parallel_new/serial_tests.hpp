#pragma once

#include "csr_hamiltonian.hpp"
#include "davidson.hpp"
#include <chrono>

void serial_davidson_test( const SetSlaterDets& stts, const FermionHamil& Hop,
  const intgrls::integrals& ints, double H_tol, int max_m, double eig_tol ) {

#if 0
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  auto H_st = clock_type::now();
  auto H  = make_csr_hamiltonian( stts, Hop, ints, H_tol );
  auto H_en = clock_type::now();

  auto eig_st = clock_type::now();
  auto E0 = davidson(max_m, H, eig_tol);
  auto eig_en = clock_type::now();

  std::cout << "E0  = " << E0 << std::endl;
  std::cout << "NNZ = " << H.nnz() << std::endl;
  std::cout << "H Construction = " << duration_type(H_en-H_st).count() << " ms\n";
  std::cout << "Davidson       = " << duration_type(eig_en-eig_st).count() 
            << " ms" << std::endl;
#endif

}

void parallel_davidson_test( const SetSlaterDets& stts, const FermionHamil& Hop,
  const intgrls::integrals& ints, double H_tol, int max_m, double eig_tol ) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  MPI_Barrier(MPI_COMM_WORLD);
  auto H_st = clock_type::now();

  auto H  = make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, 
    stts.begin(), stts.end(), Hop, ints, H_tol );

  MPI_Barrier(MPI_COMM_WORLD);
  auto H_en = clock_type::now();

  MPI_Barrier(MPI_COMM_WORLD);
  auto eig_st = clock_type::now();

  auto E0 = p_davidson(max_m, H, eig_tol);

  MPI_Barrier(MPI_COMM_WORLD);
  auto eig_en = clock_type::now();

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if( world_rank == 0 ) {
    std::cout << "E0  = " << E0 << std::endl;
    std::cout << "NNZ = " << H.nnz() << std::endl;
    std::cout << "H Construction = " << duration_type(H_en-H_st).count() << " ms\n";
    std::cout << "Davidson       = " << duration_type(eig_en-eig_st).count() 
              << " ms" << std::endl;
  }

}

void parallel_generation_test( const SetSlaterDets& stts, const FermionHamil& Hop,
  const intgrls::integrals& ints, double H_tol ) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  MPI_Barrier(MPI_COMM_WORLD);
  auto H_st = clock_type::now();

  auto H  = make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD, 
    stts.begin(), stts.end(), Hop, ints, H_tol );

  MPI_Barrier(MPI_COMM_WORLD);
  auto H_en = clock_type::now();

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<size_t> global_mf( world_size ), global_nnz(world_size);

  size_t local_mf = H.mem_footprint();
  MPI_Allgather( &local_mf, sizeof(size_t), MPI_BYTE, 
    global_mf.data(), sizeof(size_t), MPI_BYTE, 
    MPI_COMM_WORLD );

  size_t local_nnz = H.nnz();
  MPI_Allgather( &local_nnz, sizeof(size_t), MPI_BYTE, 
    global_nnz.data(), sizeof(size_t), MPI_BYTE, 
    MPI_COMM_WORLD );


  if(world_rank == 0) {

    double avg_mf = std::accumulate( global_mf.begin(), global_mf.end(), 0ul );
    avg_mf = avg_mf / world_size;
    avg_mf = avg_mf / 1024. / 1024. / 1024.; // GB

    size_t nnz =  std::accumulate( global_nnz.begin(), global_nnz.end(), 0 );

    std::cout << "Hamiltonian Statistics" << std::endl;
    std::cout << "  Duration = " << duration_type(H_en-H_st).count() << " ms" << std::endl;
    std::cout << "  NNZ      = " << nnz << std::endl;
    std::cout << "  Avg MF   = " << avg_mf << " GB" << std::endl;

  }
}
