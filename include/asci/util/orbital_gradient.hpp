#pragma once
#include <cstddef>
#include <asci/types.hpp>

namespace asci {

void orbital_rotated_generalized_fock(NumOrbital norb, NumInactive ninact, 
  NumActive nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2, 
  const double* U, size_t LDU, double* T_trans, size_t LDTT, 
  double* V_trans, size_t LDVT, double* F, size_t LDF);

double orbital_rotated_energy(NumOrbital norb, NumInactive ninact, 
  NumActive nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2, 
  const double* U, size_t LDU, double* T_trans, size_t LDTT, 
  double* V_trans, size_t LDVT, double* F, size_t LDF); 

void numerical_orbital_gradient(NumOrbital norb, 
  NumInactive ninact, NumActive nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2,
  double* OG, size_t LDOG ); 

}
