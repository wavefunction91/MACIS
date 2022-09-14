#pragma once
#include <asci/types.hpp>
#include <mpi.h>


namespace asci {

void optimize_orbitals(NumOrbital norb, NumInactive ninact, NumActive nact,
  NumVirtual nvirt, double E_core, const double* T, size_t LDT, 
  const double* V, size_t LDV, const double* A1RDM, size_t LDD1, 
  const double* A2RDM, size_t LDD2, double *K, size_t LDK);

void casscf_bfgs(NumElectron nalpha, NumElectron nbeta, NumOrbital norb, 
  NumInactive ninact, NumActive nact, NumVirtual nvirt, double E_core, 
  double* T, size_t LDT, double* V, size_t LDV, double* A1RDM, size_t LDD1, 
  double* A2RDM, size_t LDD2, MPI_Comm comm); 

}
