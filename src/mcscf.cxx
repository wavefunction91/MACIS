#include <asci/util/mcscf.hpp>
#include <asci/util/orbital_gradient.hpp>
#include <asci/util/fock_matrices.hpp>
#include <Eigen/Core>

namespace asci {

void linear_orb_rot_to_matrix(NumInactive _ni, NumActive _na,
  NumVirtual _nv, const double* K_vi, const double* K_va, 
  const double* K_ai, double* K, size_t LDK ) {

  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

  // Virtual - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto a_off = a + ni + na;
    const auto k = K_vi[a + i*nv];
    K[a_off + i*LDK] =  k;
    K[i + a_off*LDK] = -k;
  }

  // Virtual - Active Block
  for(size_t i = 0; i < na; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto i_off = i + ni;
    const auto a_off = a + ni + na;
    const auto k = K_va[a + i*nv];
    K[a_off + i_off*LDK] =  k;
    K[i_off + a_off*LDK] = -k;
  }

  // Active - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < na; ++a) {
    const auto a_off = a + ni;
    const auto k = K_ai[a + i*na];
    K[a_off + i*LDK] =  k;
    K[i + a_off*LDK] = -k;
  }

}

void linear_orb_rot_to_matrix(NumInactive ni, NumActive na,
  NumVirtual nv, const double* K_lin, double* K, size_t LDK ) {
  const auto K_vi = K_lin;
  const auto K_va = K_vi + nv.get() * ni.get();
  const auto K_ai = K_va + nv.get() * na.get();
  linear_orb_rot_to_matrix(ni, na, nv, K_vi, K_va, K_ai, K, LDK);
}





void fock_to_linear_orb_grad(NumInactive _ni, NumActive _na,
  NumVirtual _nv, const double* F, size_t LDF, 
  double* G_vi, double* G_va, double* G_ai ) {

  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

  #define FOCK(i,j) F[i + j*LDF]

  // Virtual - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto a_off = a + ni + na;
    G_vi[a + i*nv] = 2*(FOCK(a_off, i) - FOCK(i, a_off)); 
  }

  // Virtual - Active Block
  for(size_t i = 0; i < na; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto i_off = i + ni;
    const auto a_off = a + ni + na;
    G_va[a + i*nv] = 2*(FOCK(a_off, i_off) - FOCK(i_off, a_off)); 
  }

  // Active - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < na; ++a) {
    const auto a_off = a + ni;
    G_ai[a + i*na] = 2*(FOCK(a_off, i) - FOCK(i, a_off)); 
  }

 #undef FOCK

}

void fock_to_linear_orb_grad(NumInactive ni, NumActive na,
  NumVirtual nv, const double* F, size_t LDF, double* G_lin ) {
  auto G_vi = G_lin;
  auto G_va = G_vi + nv.get() * ni.get();
  auto G_ai = G_va + nv.get() * na.get();
  fock_to_linear_orb_grad(ni, na, nv, F, LDF, G_vi, G_va, G_ai);
}







struct bfgs_mcscf_functor {
    using argument_type = Eigen::VectorXd;
    using return_type   = double;

    NumOrbital  norb;
    NumInactive ninact;
    NumActive   nact;
    NumVirtual  nvirt;

    const double E_core;
    const double *T, *V, *A1RDM, *A2RDM;

    bfgs_mcscf_functor() = delete;

    bfgs_mcscf_functor(NumOrbital no, NumInactive ni, NumActive na,
      NumVirtual nv, double ec, const double* t, const double* v, const double* d1,
      const double* d2) :
      norb(no), ninact(ni), nact(na), nvirt(nv), E_core(ec),
      T(t), V(v), A1RDM(d1), A2RDM(d2) {}

    return_type eval(const argument_type& K) {

      // Expand linear rotation vector into full antisymmetric
      // matrix
      std::vector<double> K_expand(norb.get() * norb.get());
      linear_orb_rot_to_matrix(ninact, nact, nvirt, K.data(),
        K_expand.data(), norb.get());

      // Compute U = EXP[-K]
      std::vector<double> U(norb.get() * norb.get());
      compute_orbital_rotation(norb, 1.0, K_expand.data(), norb.get(),
        U.data(), norb.get() );

      // Compute energy evaluated at rotated orbitals
      return orbital_rotated_energy(norb, ninact, nact, T, norb.get(),
        V, norb.get(), A1RDM, nact.get(), A2RDM, nact.get(), U.data(),
        norb.get()) + E_core;

    }

    argument_type grad(const argument_type& K) {

      // Expand linear rotation vector into full antisymmetric
      // matrix
      std::vector<double> K_expand(norb.get() * norb.get());
      linear_orb_rot_to_matrix(ninact, nact, nvirt, K.data(),
        K_expand.data(), norb.get());

      // Compute U = EXP[-K]
      std::vector<double> U(norb.get() * norb.get());
      compute_orbital_rotation(norb, 1.0, K_expand.data(), norb.get(),
        U.data(), norb.get() );


      // Compute Rotated Fock Matrix
      const size_t no = norb.get();
      std::vector<double> Tt(no*no), Vt(no*no*no*no);
      std::vector<double> F(no*no);
      asci::orbital_rotated_generalized_fock(norb, ninact, nact,
        T, no, V, no, A1RDM, nact.get(), A2RDM, nact.get(), U.data(), 
        no, Tt.data(), no, Vt.data(), no, F.data(), no);

      argument_type G(K.size());
      fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
        G.data());
    
      return G;
    }

    return_type operator()(const argument_type& x, 
      argument_type& g) {
        g = grad(x);
        return eval(x);
    }

    static return_type dot( const argument_type& x, 
      const argument_type& y ) {
        return x.dot(y);
    }

    static void axpy( return_type alpha, 
      const argument_type& x, argument_type& y ) {
        y += alpha * x;
    }

    static return_type norm( const argument_type& x ) {
      return x.norm();
    }


    static void scal( return_type alpha, argument_type& x ) {
      x *= alpha;
    }
    static argument_type subtract( const argument_type& x,
      const argument_type& y ){
        return x - y;
    }

};

}

#include "bfgs/bfgs.hpp"

namespace asci {
void optimize_orbitals(NumOrbital norb, NumInactive ninact, NumActive nact,
  NumVirtual nvirt, double E_core, const double* T, size_t LDT, 
  const double* V, size_t LDV, const double* A1RDM, size_t LDD1, 
  const double* A2RDM, size_t LDD2, double *K, size_t LDK) {

  // Create BFGS Functor
  bfgs_mcscf_functor op(norb, ninact, nact, nvirt, E_core, T, V, A1RDM, A2RDM);

  // Initial guess of K = 0
  size_t ni = ninact.get(), na = nact.get(), nv = nvirt.get();
  Eigen::VectorXd K0(nv*(na+ni) + na*ni);
  K0.fill(0.);

  std::cout << "Initial Grad Norm = " << op.grad(K0).norm() << std::endl;

  // Optimize Orbitals
  K0 = bfgs::bfgs(op, K0);

  // Expand into full matrix
  linear_orb_rot_to_matrix(ninact, nact, nvirt, K0.data(), K, LDK);

}
}
