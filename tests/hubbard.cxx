/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */



#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <macis/model/hubbard.hpp>

#include "ut_common.hpp"

TEST_CASE("Hubbard") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t nsites = 4;
  const size_t nsites2 = nsites * nsites;
  const size_t nsites3 = nsites2 * nsites;
  const double t = 1.0, U = 4.0;
  std::vector<double> T, V;

  SECTION("1D") {
    macis::hubbard_1d(nsites, t, U, T, V);

    // Check two-body term
    for(int p = 0; p < nsites; ++p)
      for(int q = 0; q < nsites; ++q)
        for(int r = 0; r < nsites; ++r)
          for(int s = 0; s < nsites; ++s) {
            const auto mat_el = V[p + nsites * q + nsites2 * r + nsites3 * s];
            if(p == q and p == r and p == s)
              REQUIRE(mat_el == U);
            else
              REQUIRE(mat_el == 0.0);
          }

    // Check 1-body term
    for(int p = 0; p < nsites; ++p)
      for(int q = 0; q < nsites; ++q) {
        const auto mat_el = T[p + q * nsites];
        if(p == q)
          REQUIRE(mat_el == Approx(-U / 2));
        else if(std::abs(p - q) == 1)
          REQUIRE(mat_el == -t);
        else
          REQUIRE(mat_el == 0.0);
      }
  }
}
