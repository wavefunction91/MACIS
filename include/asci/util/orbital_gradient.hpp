#pragma once
#include <cstddef>

namespace asci {

void numerical_orbital_gradient(size_t norb, 
  size_t ninact, size_t nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2,
  double* OG, size_t LDOG ); 

}
