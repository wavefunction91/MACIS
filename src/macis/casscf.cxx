#include <macis/util/mcscf.hpp>
#include <macis/util/mcscf_impl.hpp>
#include <macis/util/cas.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>

namespace macis {

double casscf_diis(MCSCFSettings settings, NumElectron nalpha, NumElectron nbeta, 
  NumOrbital norb, NumInactive ninact, NumActive nact, NumVirtual nvirt, 
  double E_core, double* T, size_t LDT, double* V, size_t LDV, 
  double* A1RDM, size_t LDD1, double* A2RDM, size_t LDD2, MPI_Comm comm) { 

  using generator_t = DoubleLoopHamiltonianGenerator<64>;
  using functor_t   = CASRDMFunctor<generator_t>;
  functor_t op;
  return mcscf_impl<functor_t>(op, settings, nalpha, nbeta, norb, ninact, nact, 
    nvirt, E_core, T, LDT, V, LDV, A1RDM, LDD1, A2RDM, LDD2, comm);

}

}
