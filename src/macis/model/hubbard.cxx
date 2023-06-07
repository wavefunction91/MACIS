#include <macis/model/hubbard.hpp>

namespace macis {

void hubbard_1d(size_t nsites, double t, double U, std::vector<double>& T,
                std::vector<double>& V) {
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
}

}  // namespace macis
