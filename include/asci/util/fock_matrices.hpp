#pragma once
#include <stddef.h>

namespace asci {

void active_submatrix_1body(size_t nact, size_t ninact,
  const double* A_full, size_t LDAF, double* A_sub,
  size_t LDAS);

void active_subtensor_2body(size_t nact, size_t ninact,
  const double* A_full, size_t LDAF, double* A_sub,
  size_t LDAS);

double inactive_energy( size_t ninact, const double* T,
  size_t LDT, const double* Fi, size_t LDF );

void inactive_fock_matrix( size_t norb, size_t ninact,
  const double* T, size_t LDT, const double* V, size_t LDV, 
  double* Fi, size_t LDF );

void active_fock_matrix( size_t norb, size_t ninact,
  size_t nact, const double* V, size_t LDV, 
  const double* A1RDM, size_t LDD, double* Fa, 
  size_t LDF ); 

void aux_q_matrix( size_t nact, size_t norb, size_t ninact,
  const double* V, size_t LDV, const double* A2RDM,
  size_t LDD, double* Q, size_t LDQ );

void generalized_fock_matrix( size_t norb, size_t ninact,
  size_t nact, const double* Fi, size_t LDFi, const double* Fa,
  size_t LDFa, const double* A1RDM, size_t LDD, 
  const double* Q, size_t LDQ, double* F, size_t LDF ); 

void generalized_fock_matrix_comp_mat( size_t norb, 
  size_t ninact, size_t nact, const double* Fi, size_t LDFi,
  const double* V_full, size_t LDV, 
  const double* A1RDM, size_t LDD1, const double* A2RDM,
  size_t LDD2, double* F, size_t LDF );

double energy_from_generalized_fock(size_t ninact, size_t nact,
  const double* T, size_t LDT, const double* A1RDM, size_t LDD,
  const double* F, size_t LDF);

}
