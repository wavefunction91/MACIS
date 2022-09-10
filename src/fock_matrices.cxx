#include <asci/util/fock_matrices.hpp>
#include <blas.hh>
#include <vector>

namespace asci {

double inactive_energy( size_t ninact, const double* T,
  size_t LDT, const double* Fi, size_t LDF ) {

  double E = 0.0;
  for( size_t i = 0; i < ninact; ++i )
    E += T[i*(LDT+1)] + Fi[i*(LDF+1)];
  return E;

}

void inactive_fock_matrix( size_t norb, size_t ninact,
  const double* T, size_t LDT, const double* V, size_t LDV, 
  double* Fi, size_t LDF ) {

  const size_t LDV2 = LDV  * LDV;
  const size_t LDV3 = LDV2 * LDV;

  for(size_t p = 0; p < norb;   ++p)
  for(size_t q = 0; q < norb;   ++q) {
    double tmp = 0.0;
    for(size_t i = 0; i < ninact; ++i) {
      tmp += 2 * V[p + q*LDV  + i*(LDV2 + LDV3)] -
                 V[p + q*LDV3 + i*(LDV  + LDV2)];
    }
    Fi[p + q*LDF] = T[p + q*LDT] + tmp;
  }
}

void active_fock_matrix( size_t norb, size_t ninact,
  size_t nact, const double* V, size_t LDV, 
  const double* A1RDM, size_t LDD, double* Fa, 
  size_t LDF ) {

  const size_t LDV2 = LDV  * LDV;
  const size_t LDV3 = LDV2 * LDV;

  for(size_t p = 0; p < norb;   ++p)
  for(size_t q = 0; q < norb;   ++q) {
    double tmp = 0.0;
    for(size_t v = 0; v < nact; ++v)
    for(size_t w = 0; w < nact; ++w) {
      const size_t v_off = v + ninact;
      const size_t w_off = w + ninact;
      tmp += A1RDM[v + w*LDD] * (
              V[p + q*LDV +  v_off*LDV2 + w_off*LDV3] -
        0.5 * V[p + q*LDV3 + w_off*LDV  + v_off*LDV2]
      );
    }
    Fa[p + q*LDF] = tmp;
  }

}



void aux_q_matrix( size_t nact, size_t norb, size_t ninact,
  const double* V, size_t LDV, const double* A2RDM,
  size_t LDD, double* Q, size_t LDQ ) {

  const size_t LDV2 = LDV  * LDV;
  const size_t LDV3 = LDV2 * LDV;
  const size_t LDD2 = LDD  * LDD;
  const size_t LDD3 = LDD2 * LDD;

  for(size_t v = 0; v < nact; ++v)
  for(size_t p = 0; p < norb; ++p) {
    double tmp = 0.0;
    for(size_t w = 0; w < nact; ++w)
    for(size_t x = 0; x < nact; ++x)
    for(size_t y = 0; y < nact; ++y) {
      const size_t w_off = w + ninact;
      const size_t x_off = x + ninact;
      const size_t y_off = y + ninact;

      tmp += A2RDM[v + w*LDD     + x*LDD2     + y*LDD3] *
                 V[p + w_off*LDV + x_off*LDV2 + y_off*LDV3];
    }
    Q[v + p*LDQ] = 2. * tmp;
  }

}

void generalized_fock_matrix( size_t norb, size_t ninact,
  size_t nact, const double* Fi, size_t LDFi, const double* Fa,
  size_t LDFa, const double* A1RDM, size_t LDD, 
  const double* Q, size_t LDQ, double* F, size_t LDF ) {

  // Inactive - General
  // F(i,p) = 2*( Fi(p,i) + Fa(p,i) )
  for(size_t i = 0; i < ninact; ++i)
  for(size_t p = 0; p < norb;   ++p) {
    F[i + p*LDF] = 2. * ( Fi[p + i*LDFi] + Fa[p + i*LDFa] );
  }

  // Compute X(p,x) = Fi(p,y) * A1RDM(y,x)
  std::vector<double> X(norb * nact);
  blas::gemm(blas::Layout::ColMajor,
    blas::Op::NoTrans, blas::Op::NoTrans,
    norb, nact, nact, 
    1.0, Fi + ninact*LDFi, LDFi, A1RDM, LDD,
    0.0, X.data(), norb);

  // Active - General
  for(size_t v = 0; v < nact; ++v)
  for(size_t p = 0; p < norb; ++p) {
    const size_t v_off = v + ninact;
    F[v_off + p*LDF] = X[p + v*norb] + Q[v + p*LDQ];
  }

}

double energy_from_generalized_fock(size_t ninact, size_t nact,
  const double* T, size_t LDT, const double* A1RDM, size_t LDD,
  const double* F, size_t LDF) {

  double E = 0;
  // Inactive-Inactve E <- 2*T(i,i)
  for(size_t i = 0; i < ninact; ++i) {
    E += 2.*T[i*(LDT+1)];
  }

  // Active-Active
  // E <- A1RDM(x,y) * T(x,y)
  for(size_t x = 0; x < nact; ++x)
  for(size_t y = 0; y < nact; ++y) {
    const size_t x_off = x + ninact;
    const size_t y_off = y + ninact;
    E += A1RDM[x + y*LDD] * T[x_off + y_off*LDT];
  }

  // Fock piece E <- F(i,i) + F(x,x)
  for(size_t p = 0; p < (nact+ninact); ++p) {
    E += F[p*(LDF+1)];
  }

  return 0.5 * E;

}

}
