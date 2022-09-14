#pragma once
#include <asci/types.hpp>


namespace asci {

void optimize_orbitals(NumOrbital norb, NumInactive ninact, NumActive nact,
  NumVirtual nvirt, double E_core, const double* T, size_t LDT, 
  const double* V, size_t LDV, const double* A1RDM, size_t LDD1, 
  const double* A2RDM, size_t LDD2, double *K, size_t LDK);

}
