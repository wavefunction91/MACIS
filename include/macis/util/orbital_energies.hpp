#pragma once
#include <macis/types.hpp>

namespace macis {

void canonical_orbital_energies(NumOrbital norb, NumInactive ninact,
  const double* T, size_t LDT, const double* V, size_t LDV, double* eps); 

}
