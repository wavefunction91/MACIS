/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#define CATCH_CONFIG_RUNNER
#include <macis/util/mpi.hpp>

#include "catch2/catch.hpp"

int main(int argc, char* argv[]) {
#ifdef MACIS_ENABLE_MPI
  MPI_Init(&argc, &argv);
#endif
  int result = Catch::Session().run(argc, argv);
#ifdef MACIS_ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return result;
}
