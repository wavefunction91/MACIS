#include <asci/util/mcscf.hpp>
#include <asci/util/orbital_gradient.hpp>
#include <asci/util/fock_matrices.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/selected_ci_diag.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/davidson.hpp>
#include <Eigen/Core>

namespace asci {

template <typename T>
auto split_linear_orb_rot( NumInactive ni, NumActive na, NumVirtual nv, 
  T&& V_lin ) {
  auto V_vi = V_lin;
  auto V_va = V_vi + nv.get() * ni.get();
  auto V_ai = V_va + nv.get() * na.get();
  return std::make_tuple(V_vi, V_va, V_ai);
}

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

  auto [K_vi, K_va, K_ai] = split_linear_orb_rot(ni,na,nv,K_lin);
  linear_orb_rot_to_matrix(ni, na, nv, K_vi, K_va, K_ai, K, LDK);

}

void matrix_to_linear_orb_rot(NumInactive _ni, NumActive _na,
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
    G_vi[a + i*nv] = FOCK(a_off, i); 
  }

  // Virtual - Active Block
  for(size_t i = 0; i < na; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto i_off = i + ni;
    const auto a_off = a + ni + na;
    G_va[a + i*nv] = FOCK(a_off, i_off); 
  }

  // Active - Inactive Block
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < na; ++a) {
    const auto a_off = a + ni;
    G_ai[a + i*na] = FOCK(a_off, i); 
  }

 #undef FOCK

}

void matrix_to_linear_orb_rot(NumInactive ni, NumActive na,
  NumVirtual nv, const double* F, size_t LDF, double* G_lin ) {

  auto [G_vi, G_va, G_ai] = split_linear_orb_rot(ni,na,nv,G_lin);
  matrix_to_linear_orb_rot(ni, na, nv, F, LDF, G_vi, G_va, G_ai);

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

  auto [G_vi, G_va, G_ai] = split_linear_orb_rot(ni,na,nv,G_lin);
  fock_to_linear_orb_grad(ni, na, nv, F, LDF, G_vi, G_va, G_ai);

}


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
  for(size_t i = 0; i < ni; ++i)
  for(size_t a = 0; a < nv; ++a) {
    const auto a_off = a + ni + na;
    H_vi[a + i*nv] = 4. * (
      FI(a_off,a_off) + FA(a_off,a_off) -
      FI(i,i)         - FA(i,i)
    );
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

void approx_diag_hessian(NumInactive ni, NumActive na, NumVirtual nv,
  const double* Fi, size_t LDFi, const double* Fa, size_t LDFa,
  const double* A1RDM, size_t LDD, const double* F, size_t LDF, 
  double* H_lin) {

  auto [H_vi, H_va, H_ai] = split_linear_orb_rot(ni, na, nv, H_lin);
  approx_diag_hessian(ni, na, nv, Fi, LDFi, Fa, LDFa, A1RDM, LDD, F, LDF,
    H_vi, H_va, H_ai);

}

template <typename... Args>
void approx_diag_hessian(NumOrbital norb, NumInactive ninact, NumActive nact,
  NumVirtual nvirt, const double* T, size_t LDT, const double* V, size_t LDV,
  const double* A1RDM, size_t LDD1, const double* A2RDM, size_t LDD2,
  Args&&... args) {

  const size_t no = norb.get();
  const size_t ni = ninact.get();
  const size_t na = nact.get();
  const size_t nv = nvirt.get();

  // Compute inactive Fock
  std::vector<double> Fi(no*no);
  inactive_fock_matrix(norb, ninact, T, LDT, V, LDV, Fi.data(), no);

  // Compute active fock
  std::vector<double> Fa(no*no);
  active_fock_matrix(norb, ninact, nact, V, LDV, A1RDM, LDD1, Fa.data(), no);

  // Compute Q matrix
  std::vector<double> Q(na*no);
  aux_q_matrix(nact, norb, ninact, V, LDV, A2RDM, LDD2, Q.data(), na);

  // Compute generalized Fock
  std::vector<double> F(no*no);
  generalized_fock_matrix(norb, ninact, nact, Fi.data(), no, Fa.data(), no,
    A1RDM, LDD1, Q.data(), na, F.data(), no);

  // Compute approximate diagonal hessian
  approx_diag_hessian(ninact, nact, nvirt, Fi.data(), no, Fa.data(), no,
    A1RDM, LDD1, F.data(), no, std::forward<Args>(args)... );
}







struct bfgs_mcscf_functor {
    using argument_type = Eigen::VectorXd;
    using return_type   = double;

    NumOrbital  norb;
    NumInactive ninact;
    NumActive   nact;
    NumVirtual  nvirt;

    const double grad_tol;
    const double E_core;
    const double *T, *V, *A1RDM, *A2RDM;

    bfgs_mcscf_functor() = delete;

    bfgs_mcscf_functor(NumOrbital no, NumInactive ni, NumActive na,
      NumVirtual nv, double ec, const double* t, const double* v, 
      const double* d1, const double* d2, double tol) :
      grad_tol(tol),
      norb(no), ninact(ni), nact(na), nvirt(nv), E_core(ec),
      T(t), V(v), A1RDM(d1), A2RDM(d2) {}

    bool converged(const argument_type& X, const argument_type& G) {
      return G.norm() < grad_tol; 
    }

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



void one_index_transformed_hamiltonian(NumOrbital norb,
  const double* T, size_t LDT, const double* V, size_t LDV,
  const double* X, size_t LDX, double* Tx, size_t LDTx,
  double* Vx, size_t LDVx) {

  const size_t no = norb.get();

  for(size_t p = 0; p < no; ++p)
  for(size_t q = 0; q < no; ++q) {
    double tmp = 0.0;
    for(size_t o = 0; o < no; ++o) {
      tmp += X[p + o*LDX] * T[o + q*LDT] +
             X[q + o*LDX] * T[p + o*LDX];
    }
    Tx[p + q*LDTx] = tmp;
  }

  for(size_t p = 0; p < no; ++p)
  for(size_t q = 0; q < no; ++q) 
  for(size_t r = 0; r < no; ++r) 
  for(size_t s = 0; s < no; ++s) {
    double tmp = 0.0;
    for(size_t o = 0; o < no; ++o) {
      tmp += X[p + o*LDX] * V[o + q*LDV + r*LDV*LDV + s*LDV*LDV*LDV] +
             X[q + o*LDX] * V[p + o*LDV + r*LDV*LDV + s*LDV*LDV*LDV] +
             X[r + o*LDX] * V[p + q*LDV + o*LDV*LDV + s*LDV*LDV*LDV] +
             X[s + o*LDX] * V[p + q*LDV + r*LDV*LDV + o*LDV*LDV*LDV];
    }
    Vx[p + q*LDVx + r*LDVx*LDVx + s*LDVx*LDVx*LDVx] = tmp;
  }

}


void orb_orb_hessian_contract(NumOrbital norb, NumInactive ninact, 
  NumActive nact, NumVirtual nvirt, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM, size_t LDD1,
  const double* A2RDM, size_t LDD2, const double* OG,
  const double* K_lin, double* HK_lin) {

  const size_t no  = norb.get();
  const size_t na  = nact.get();
  const size_t no2 = no  * no;
  const size_t no4 = no2 * no2;
  const size_t orb_rot_sz = 
    nvirt.get() * (nact.get() + ninact.get()) + nact.get() * ninact.get();

  // Expand to full antisymmetric K
  std::vector<double> K_full(no2);
  linear_orb_rot_to_matrix( ninact, nact, nvirt, K_lin, K_full.data(), no);

  // Compute one-index transformed hamiltonian
  std::vector<double> Tk(no2), Vk(no4);
  one_index_transformed_hamiltonian(norb, T, LDT, V, LDV, K_full.data(), no,
    Tk.data(), no, Vk.data(), no);

  // Compute gradient-like term with transformed Hamiltonian
  std::vector<double> Fk(no2);
  generalized_fock_matrix_comp_mat2(norb, ninact, nact, Tk.data(), no,
    Vk.data(), no, A1RDM, LDD1, A2RDM, LDD2, Fk.data(), no);
  // Store in HK to initialize the memory 
  fock_to_linear_orb_grad(ninact, nact, nvirt, Fk.data(), no, HK_lin);

  // Expand OG into full antisymmetric matrix (reuse Tk)
  std::fill(Tk.begin(), Tk.end(), 0);
  linear_orb_rot_to_matrix(ninact, nact, nvirt, OG, Tk.data(), no);

  // Compute G*K in Fk
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    no, no, no, 1., Tk.data(), no, K_full.data(), no, 0., Fk.data(), no);

  // Replace Tk with [G,K]
  for(size_t p = 0; p < no; ++p)
  for(size_t q = 0; q < no; ++q) {
    Tk[p + q*no] = Fk[p + q*no] - Fk[q + p*no];
  }      

  // Take out orbital rotation piece into a temporary buffer
  std::vector<double> tmp_hk(orb_rot_sz);
  matrix_to_linear_orb_rot(ninact, nact, nvirt, Tk.data(), no, tmp_hk.data());

  // Add to final result
  for(size_t i = 0; i < orb_rot_sz; ++i) HK_lin[i] += tmp_hk[i];

}

}

#include "bfgs/bfgs.hpp"

namespace asci {

void optimize_orbitals(MCSCFSettings settings, NumOrbital norb, 
  NumInactive ninact, NumActive nact, NumVirtual nvirt, double E_core, 
  const double* T, size_t LDT, const double* V, size_t LDV, 
  const double* A1RDM, size_t LDD1, const double* A2RDM, size_t LDD2, 
  double *K, size_t LDK) {

  size_t no = norb.get(), ni = ninact.get(), na = nact.get(), nv = nvirt.get();

  // Compute Diagonal Hessian Approximation
  std::vector<double> DH(nv*(na+ni) + na*ni);
  approx_diag_hessian(norb, ninact, nact, nvirt, T, LDT, V, LDV, A1RDM, LDD1,
    A2RDM, LDD2, DH.data() );

  // Create BFGS Functor
  bfgs_mcscf_functor op(norb, ninact, nact, nvirt, E_core, T, V, A1RDM, A2RDM,
    settings.orb_grad_tol_bfgs);

  // Initial diagonal hessian
  bfgs::DiagInitializedBFGSHessian<bfgs_mcscf_functor> H0(DH.size(), DH.data());

  // Initial guess of K = 0
  Eigen::VectorXd K0(nv*(na+ni) + na*ni);
  K0.fill(0.);

  // Optimize Orbitals
  K0 = bfgs::bfgs(op, K0, H0, bfgs::BFGSSettings{settings.max_bfgs_iter});

  // Expand into full matrix
  linear_orb_rot_to_matrix(ninact, nact, nvirt, K0.data(), K, LDK);

}



template <typename HamGen>
double compute_casci_rdms(MCSCFSettings settings, NumOrbital norb, 
  size_t nalpha, size_t nbeta, double* T, double* V, double* ORDM, 
  double* TRDM, MPI_Comm comm) {

  constexpr auto nbits = HamGen::nbits;

  int rank; MPI_Comm_rank(comm, &rank);

  // Hamiltonian Matrix Element Generator
  asci::DoubleLoopHamiltonianGenerator<nbits> ham_gen( norb.get(), V, T );
  
  // Compute Lowest Energy Eigenvalue (ED)
  std::vector<double> C;
  auto dets = asci::generate_hilbert_space<nbits>(norb.get(), nalpha, nbeta);
  double E0 = asci::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    settings.ci_matel_tol, settings.ci_max_subspace, settings.ci_res_tol, C, 
    comm, true);

  // Compute RDMs
  ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
    C.data(), ORDM, TRDM);

  return E0;
}


template <typename HamGen>
void casscf_bfgs_impl(MCSCFSettings settings, NumElectron nalpha, 
  NumElectron nbeta, NumOrbital norb, NumInactive ninact, NumActive nact, 
  NumVirtual nvirt, double E_core, double* T, size_t LDT, double* V, size_t LDV, 
  double* A1RDM, size_t LDD1, double* A2RDM, size_t LDD2, MPI_Comm comm) {

  auto logger = spdlog::get("casscf");
  if(!logger) logger = spdlog::stdout_color_mt("casscf");

  logger->info("[MCSCF Settings]:");
  logger->info(
    "  {:13} = {:3}, {:13} = {:3}, {:13} = {:3}",
    "NACTIVE_ALPHA", nalpha.get(),
    "NACTIVE_BETA" , nbeta.get(),
    "NORB_TOTAL",    norb.get()
  );
  logger->info(
    "  {:13} = {:3}, {:13} = {:3}, {:13} = {:3}",
    "NINACTIVE", ninact.get(),
    "NACTIVE",    nact.get(),
    "NVIRTUAL",  nvirt.get()
  );
  logger->info( "  {:13} = {:.6f}", "E_CORE", E_core);
  logger->info( 
    "  {:13} = {:.6e}, {:13} = {:.6e}, {:13} = {:3}",
    "ORBGRAD_TOL", settings.orb_grad_tol_mcscf,
    "BFGS_TOL",    settings.orb_grad_tol_bfgs,
    "BFGS_MAX_ITER", settings.max_bfgs_iter
   );
  logger->info("  {:13} = {:.6e}, {:13} = {:.6e}, {:13} = {:3}",
    "CI_RES_TOL",  settings.ci_res_tol,
    "CI_MATEL_TOL", settings.ci_matel_tol, 
    "CI_MAX_SUB",   settings.ci_max_subspace
  );

  const size_t no = norb.get(), ni = ninact.get(), na = nact.get(), 
               nv = nvirt.get();

  const size_t no2 = no  * no;
  const size_t no4 = no2 * no2;
  const size_t na2 = na  * na;
  const size_t na4 = na2 * na2;

  const size_t orb_rot_sz = nv*(ni + na) + na*ni;

  // Compute Active Space Hamiltonian 
  std::vector<double> T_active(na2), V_active(na4), F_inactive(no2);
  active_hamiltonian(norb, nact, ninact, T, LDT, V, LDV, F_inactive.data(),
    no, T_active.data(), na, V_active.data(), na);


  // Compute Inactive Energy
  auto E_inactive = inactive_energy(ninact, T, LDT, F_inactive.data(), no);
  E_inactive += E_core;

#if 0
  // Compute active RDMs
  auto E0 = compute_casci_rdms<HamGen>(NumOrbital(na), nalpha.get(), nbeta.get(),
    T_active.data(), V_active.data(), A1RDM, A2RDM, comm);
#else
  auto E0 = blas::dot(na2, A1RDM, 1, T_active.data(), 1) +
            blas::dot(na4, A2RDM, 1, V_active.data(), 1) +
            E_inactive; 
#endif

  const std::string fmt_string = "iter = {:4} E(CI) = {:.10f}, dE = {:18.10e}, |orb_grad| = {:18.10e}";

  // Compute Gradient
  bool converged = false;
  {
  std::vector<double> F(no2), OG(orb_rot_sz);
  generalized_fock_matrix_comp_mat1( norb, ninact, nact, 
    F_inactive.data(), no, V, LDV, A1RDM, LDD1, A2RDM, LDD2, 
    F.data(), no);
  fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
    OG.data());

  double grad_nrm = std::accumulate(OG.begin(),OG.end(),0.0,
    [](auto a, auto b){ return a + b*b; });
  grad_nrm = std::sqrt(grad_nrm);
  converged = grad_nrm < settings.orb_grad_tol_mcscf;

  logger->info(fmt_string, 0, E0, 0.0, grad_nrm);
  }

  // MCSCF Loop
  for(size_t iter = 0; iter < settings.max_macro_iter; ++iter) {
     if(converged) break;
     // Save old E0
     const double E0_old = E0;

     // Optimize Orbitals
     std::vector<double> K_opt(no2);
     optimize_orbitals(settings, norb, ninact, nact, nvirt, E_core, T, LDT, 
       V, LDV, A1RDM, LDD1, A2RDM, LDD2, K_opt.data(), no);

     // Rotate MO Hamiltonian in place
     std::vector<double> U(no2);
     compute_orbital_rotation(norb, 1.0, K_opt.data(), no, U.data(), no );
     two_index_transform(no, no, T, LDT, U.data(), no, T, LDT);
     four_index_transform(no, no, 0, V, LDV, U.data(), no, V, LDV);

     // Compute Active Space Hamiltonian 
     active_hamiltonian(norb, nact, ninact, T, LDT, V, LDV, F_inactive.data(),
       no, T_active.data(), na, V_active.data(), na);

     // Compute Inactive Energy
     E_inactive = inactive_energy(ninact, T, LDT, F_inactive.data(), no) + E_core;

     // Compute active RDMs
     for( auto i = 0; i < na2; ++i ) A1RDM[i] = 0.0;
     for( auto i = 0; i < na4; ++i ) A2RDM[i] = 0.0;
     E0 = compute_casci_rdms<HamGen>(settings, NumOrbital(na), 
       nalpha.get(), nbeta.get(), T_active.data(), V_active.data(), A1RDM, 
       A2RDM, comm) + E_inactive;

    // Compute Gradient
    std::vector<double> F(no2), OG(orb_rot_sz);
    generalized_fock_matrix_comp_mat1( norb, ninact, nact, 
      F_inactive.data(), no, V, LDV, A1RDM, LDD1, A2RDM, LDD2, 
      F.data(), no);
    fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
      OG.data());

    // Gradient Norm
     double grad_nrm = std::accumulate(OG.begin(),OG.end(),0.0,
       [](auto a, auto b){ return a + b*b; });
     grad_nrm = std::sqrt(grad_nrm);
     logger->info(fmt_string, iter+1, E0, E0 - E0_old, grad_nrm);

     converged = grad_nrm < settings.orb_grad_tol_mcscf;
  }

  if(converged) logger->info("CASSCF Converged");

#if 0
  // Check orbital stability
  struct OrbHessianOperator {
    
    NumOrbital  norb;
    NumInactive ninact;
    NumActive   nact;
    NumVirtual  nvirt;
    const double *m_T, *m_V, *m_OG, *m_A1RDM, *m_A2RDM;

    void operator_action(size_t m, double alpha, const double* K, size_t LDK,
      double beta, double* AK, size_t LDAK) const {

      const size_t no = norb.get();
      const size_t na = nact.get();
      orb_orb_hessian_contract(norb, ninact, nact, nvirt, m_T, no,
        m_V, no, m_A1RDM, na, m_A2RDM, na, m_OG, K, AK);
      
    }

  };


  // Compute the gradient again
  std::vector<double> F(no2), OG(orb_rot_sz);
  generalized_fock_matrix_comp_mat1( norb, ninact, nact, 
    F_inactive.data(), no, V, LDV, A1RDM, LDD1, A2RDM, LDD2, 
    F.data(), no);
  fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
    OG.data());

  OrbHessianOperator op{norb, ninact, nact, nvirt, T, V, OG.data(),
    A1RDM, A2RDM};

  std::vector<double> X(orb_rot_sz, 0);
  std::vector<double> DH(orb_rot_sz);
  approx_diag_hessian(norb, ninact, nact, nvirt, T, LDT, V, LDV, A1RDM, LDD1,
    A2RDM, LDD2, DH.data() );
  auto D_min = std::min_element(DH.begin(), DH.end());
  auto min_idx = std::distance( DH.begin(), D_min );
  std::cout << "DMIN = " << *D_min << std::endl;
  X[min_idx] = 1.;

    

  auto L = davidson(orb_rot_sz, 100, op, DH.data(), 1e-8, X.data() );

  std::vector<double> iden(orb_rot_sz * orb_rot_sz), H(iden.size());
  for( auto i = 0; i < orb_rot_sz; ++i) iden[i*(orb_rot_sz+1)] = 1.0;
  for( auto i = 0; i < orb_rot_sz; ++i) {
    op.operator_action(1, 1.0, iden.data() + i*orb_rot_sz, orb_rot_sz,
      0.0, H.data() + i*orb_rot_sz, orb_rot_sz);
  }

  //std::vector<std::complex<double>> W(orb_rot_sz);
  //lapack::geev(lapack::Job::NoVec, lapack::Job::NoVec,
  //  orb_rot_sz, H.data(), orb_rot_sz, W.data(), NULL, 1, NULL, 1);
  //std::sort(W.begin(),W.end(),[](auto a, auto b){ return std::real(a) < std::real(b);});
  //for( auto w : W ) std::cout << w << std::endl;
#endif 

}



void casscf_bfgs(MCSCFSettings settings, NumElectron nalpha, NumElectron nbeta, 
  NumOrbital norb, NumInactive ninact, NumActive nact, NumVirtual nvirt, 
  double E_core, double* T, size_t LDT, double* V, size_t LDV, 
  double* A1RDM, size_t LDD1, double* A2RDM, size_t LDD2, MPI_Comm comm) { 

  using generator_t = DoubleLoopHamiltonianGenerator<64>;
  casscf_bfgs_impl<generator_t>(settings, nalpha, nbeta, norb, ninact, nact, 
    nvirt, E_core, T, LDT, V, LDV, A1RDM, LDD1, A2RDM, LDD2, comm);

}

}
