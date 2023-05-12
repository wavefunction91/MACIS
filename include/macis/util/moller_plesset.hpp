#pragma once
#include <macis/types.hpp>

namespace macis {

using NumCanonicalOccupied = NamedType<size_t, struct nocc_canon_type>;
using NumCanonicalVirtual  = NamedType<size_t, struct nvir_canon_type>;

void mp2_t2(NumCanonicalOccupied nocc, NumCanonicalVirtual nvir, 
  const double* V, size_t LDV, const double* eps, double* T2); 

void mp2_1rdm(NumOrbital norb, NumCanonicalOccupied nocc, 
  NumCanonicalVirtual nvir, const double* T, size_t LDT, 
  const double* V, size_t LDV, double* ORDM, size_t LDD);

void mp2_natural_orbitals(NumOrbital norb, NumCanonicalOccupied nocc, 
  NumCanonicalVirtual nvir, const double* T, size_t LDT, 
  const double* V, size_t LDV, double* ON, double* NO_C, size_t LDC);

}
