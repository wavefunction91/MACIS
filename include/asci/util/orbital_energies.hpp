#pragma once
#include <asci/types.hpp>

namespace asci {

void canonical_orbital_energies(NumOrbital norb, NumInactive ninact,
  const double* T, size_t LDT, const double* V, size_t LDV, double* eps); 

}
