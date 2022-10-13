#pragma once
#include <asci/types.hpp>

namespace asci {

void precond_cg_orbital_step(NumOrbital norb, NumInactive ninact, NumActive nact,
  NumVirtual nvirt, const double* Fi, size_t LDFi, const double* Fa, size_t LDFa,
  const double* F, size_t LDF, const double* A1RDM, size_t LDD, const double* OG,
  double* K_lin);

}
