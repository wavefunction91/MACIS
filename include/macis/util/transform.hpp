#pragma once
#include <cstddef>

namespace macis {

// Y(p,q) = C(i,p) * X(i,j) * C(j,q)
// X <- [norb_old, norb_old]
// Y <- [norb_new, norb_new]
// C <- [norb_old, norb_new]
void two_index_transform( size_t norb_old, size_t norb_new,
  const double* X, size_t LDX, const double* C, size_t LDC, 
  double* Y, size_t LDY ); 

void four_index_transform( size_t norb_old, size_t norb_new,
  size_t ncontract, const double* X, size_t LDX, 
  const double* C, size_t LDC, double* Y, size_t LDY );
 
}
