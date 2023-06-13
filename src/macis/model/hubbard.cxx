/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <macis/model/hubbard.hpp>

namespace macis {

void hubbard_1d(size_t nsites, double t, double U,  
                std::vector<double>& T, std::vector<double>& V,
                bool pbc) {
  T.resize(nsites * nsites);
  V.resize(nsites * nsites * nsites * nsites);

  for(size_t p = 0; p < nsites; ++p) {
    // Half-filling Chemical Potential
    T[p * (nsites + 1)] = -U / 2;

    // On-Site Interaction
    V[p * (nsites * nsites * nsites + nsites * nsites + nsites + 1)] = U;

    // Hopping
    if(p < nsites - 1) {
      T[p + (p + 1) * nsites] = -t;
      T[(p + 1) + p * nsites] = -t;
    }
  }

  // PBC for 1-D
  if(pbc) {
    T[ (nsites-1) ]         = -t;
    T[ (nsites-1) * nsites] = -t;
  }
}

}  // namespace macis
