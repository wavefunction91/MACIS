#include "orbital_hessian.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <blas.hh>

namespace asci {

void approx_diag_hessian(NumInactive _ni, NumActive _na, NumVirtual _nv,
  const double* Fi, size_t LDFi, const double* Fa, size_t LDFa,
  const double* A1RDM, size_t LDD, const double* F, size_t LDF, 
  double* H_vi, double* H_va, double* H_ai) {

  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

  #define TWO_IDX(A,i,j,LDA) A[i + j*LDA]
  #define FI(i,j) TWO_IDX(Fi,i,j,LDFi)
  #define FA(i,j) TWO_IDX(Fa,i,j,LDFa)
  #define FF(i,j) TWO_IDX(F,i,j,LDF)
  #define ORDM(i,j) TWO_IDX(A1RDM,i,j,LDD)

  // Virtual - Inactive Block
  for(size_t i = 0; i < ni; ++i) {
    // Cache inactive-inactive term
    const auto ii_diff = FI(i,i) + FA(i,i);
    for(size_t a = 0; a < nv; ++a) {
      const auto a_off = a + ni + na;
      H_vi[a + i*nv] = 4. * (
        FI(a_off,a_off) + FA(a_off,a_off) - ii_diff
      );
    }
  }

  // Virtual - Active Block
  for(size_t i = 0; i < na; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto i_off = i + ni;
    const auto a_off = a + ni + na;
    H_va[a + i*nv] = 2. * ORDM(i,i) *(FI(a_off,a_off) + FA(a_off,a_off)) - 
                     2. * FF(i_off,i_off);
  }

  // Active - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < na; ++a) {
    const auto a_off = a + ni;
    H_ai[a + i*na] = 2. * ORDM(a,a) * (FI(i,i) + FA(i,i)) +
                     4. * (FI(a_off,a_off) + FA(a_off,a_off) -
                           FI(i,i)         - FA(i,i) ) -
                     2. * FF(a_off, a_off);
  }
  
  #undef TWO_IDX
  #undef FI
  #undef FA
  #undef FF
  #undef ORDM

}



#if 0
void approx_diag_hessian_2(NumInactive _ni, NumActive _na, NumVirtual _nv,
  const double* Fi, size_t LDFi, const double* Fa, size_t LDFa,
  const double* Q, size_t LDQ, const double* A1RDM, size_t LDD, 
  const double* F, size_t LDF, double* H_vi, double* H_va, double* H_ai) {

  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

  std::cout << std::scientific << std::setprecision(12);
#if 0
  for( auto x = 0; x < na; ++x ) {
    std::cout << "I[" << x << "] = " << Q[x + (ni+x)*LDQ] << std::endl;
  }
#endif

  std::vector<double> diag_Fi_x_A1(na);
  for(size_t x = 0; x < na; ++x ) {
    diag_Fi_x_A1[x] = 
      blas::dot(na, A1RDM + x*LDD, 1, Fi + ni + (x+ni)*LDFi, 1);
#if 0
    std::cout << "DIF[" << x << "] = " << diag_Fi_x_A1[x] << std::endl;
#endif
  }

  #define TWO_IDX(A,i,j,LDA) A[i + j*LDA]
  #define FI(i,j) TWO_IDX(Fi,i,j,LDFi)
  #define FA(i,j) TWO_IDX(Fa,i,j,LDFa)
  #define FF(i,j) TWO_IDX(F,i,j,LDF)
  #define ORDM(i,j) TWO_IDX(A1RDM,i,j,LDD)

  // Virtual - Inactive Block
  for(size_t i = 0; i < ni; ++i) {
    // Cache inactive-inactive term
    const auto ii_diff = FI(i,i) + FA(i,i);
    for(size_t a = 0; a < nv; ++a) {
      const auto a_off = a + ni + na;
      H_vi[a + i*nv] = 4. * (
        FI(a_off,a_off) + FA(a_off,a_off) - ii_diff
      );
      std::cout << "H_vi[" << a << ","<< i << "] = " <<
        H_vi[a + i*nv] << std::endl;
    }
  }

  // Virtual - Active Block
  for(size_t i = 0; i < na; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto i_off = i + ni;
    const auto a_off = a + ni + na;
    H_va[a + i*nv] = 2. * ORDM(i,i) *(FI(a_off,a_off) + FA(a_off,a_off)) - 
                     2. * FF(i_off,i_off);
    std::cout << "H_va[" << a << ","<< i << "] = " <<
      H_va[a + i*nv] << std::endl;
  }

  // Active - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < na; ++a) {
    const auto a_off = a + ni;
    H_ai[a + i*na] = 2. * ORDM(a,a) * (FI(i,i) + FA(i,i)) +
                     4. * (FI(a_off,a_off) + FA(a_off,a_off) -
                           FI(i,i)         - FA(i,i) ) -
                     2. * FF(a_off, a_off);
  }
  
  #undef TWO_IDX
  #undef FI
  #undef FA
  #undef FF
  #undef ORDM

}
#endif

}
