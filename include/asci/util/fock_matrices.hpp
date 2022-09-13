#pragma once
#include <stddef.h>
#include <asci/types.hpp>

namespace asci {

void active_submatrix_1body(NumActive nact, NumInactive ninact,
  const double* A_full, size_t LDAF, double* A_sub,
  size_t LDAS);

void active_subtensor_2body(NumActive nact, NumInactive ninact,
  const double* A_full, size_t LDAF, double* A_sub,
  size_t LDAS);

void active_hamiltonian(NumOrbital norb, NumActive nact, NumInactive ninact,
  const double* T_full, size_t LDTF, const double* V_full, size_t LDVF,
  double* Fi, size_t LDFi, double* T_active, size_t LDTA, double* V_active,
  size_t LDVA);

double inactive_energy( NumInactive ninact, const double* T,
  size_t LDT, const double* Fi, size_t LDF );

void inactive_fock_matrix( NumOrbital norb, NumInactive ninact,
  const double* T, size_t LDT, const double* V, size_t LDV, 
  double* Fi, size_t LDF );

void active_fock_matrix( NumOrbital norb,  NumInactive ninact,
  NumActive nact, const double* V, size_t LDV, 
  const double* A1RDM, size_t LDD, double* Fa, 
  size_t LDF ); 

void aux_q_matrix( NumActive nact, NumOrbital norb,  NumInactive ninact,
  const double* V, size_t LDV, const double* A2RDM,
  size_t LDD, double* Q, size_t LDQ );

void generalized_fock_matrix( NumOrbital norb,  NumInactive ninact,
  NumActive nact, const double* Fi, size_t LDFi, const double* Fa,
  size_t LDFa, const double* A1RDM, size_t LDD, 
  const double* Q, size_t LDQ, double* F, size_t LDF ); 

void generalized_fock_matrix_comp_mat1( NumOrbital norb, 
  NumInactive ninact, NumActive nact, const double* Fi, size_t LDFi,
  const double* V_full, size_t LDV, 
  const double* A1RDM, size_t LDD1, const double* A2RDM,
  size_t LDD2, double* F, size_t LDF );


void generalized_fock_matrix_comp_mat2( NumOrbital norb, 
  NumInactive ninact, NumActive nact, const double* T, size_t LDT,
  const double* V_full, size_t LDV, 
  const double* A1RDM, size_t LDD1, const double* A2RDM,
  size_t LDD2, double* F, size_t LDF );

double energy_from_generalized_fock( NumInactive ninact, NumActive nact,
  const double* T, size_t LDT, const double* A1RDM, size_t LDD,
  const double* F, size_t LDF);

}
