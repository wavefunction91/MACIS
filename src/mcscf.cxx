#include <asci/util/mcscf.hpp>
#include <asci/util/orbital_gradient.hpp>
#include <asci/util/fock_matrices.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/selected_ci_diag.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/davidson.hpp>
#include <asci/fcidump.hpp>
#include <Eigen/Core>

#include "orbital_rotation_utilities.hpp"
#include "orbital_hessian.hpp"
#include "orbital_steps.hpp"
#include "diis.hpp"

namespace asci {



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

namespace asci {

template <typename HamGen>
double compute_casci_rdms(MCSCFSettings settings, NumOrbital norb, 
  size_t nalpha, size_t nbeta, double* T, double* V, double* ORDM, 
  double* TRDM, std::vector<double>& C, MPI_Comm comm) {

  constexpr auto nbits = HamGen::nbits;

  int rank; MPI_Comm_rank(comm, &rank);

  // Hamiltonian Matrix Element Generator
  size_t no = norb.get();
  HamGen ham_gen( 
    matrix_span<double>(T,no,no),
    rank4_span<double>(V,no,no,no,no) 
  );
  
  // Compute Lowest Energy Eigenvalue (ED)
  auto dets = asci::generate_hilbert_space<nbits>(norb.get(), nalpha, nbeta);
  double E0 = asci::selected_ci_diag( dets.begin(), dets.end(), ham_gen,
    settings.ci_matel_tol, settings.ci_max_subspace, settings.ci_res_tol, C, 
    comm, true);

  // Compute RDMs
  ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
    C.data(), matrix_span<double>(ORDM,no,no), 
    rank4_span<double>(TRDM,no,no,no,no));

  return E0;
}


template <typename HamGen>
void casscf_bfgs_impl(MCSCFSettings settings, NumElectron nalpha, 
  NumElectron nbeta, NumOrbital norb, NumInactive ninact, NumActive nact, 
  NumVirtual nvirt, double E_core, double* T, size_t LDT, double* V, size_t LDV, 
  double* A1RDM, size_t LDD1, double* A2RDM, size_t LDD2, MPI_Comm comm) {


  /******************************************************************
   *  Top of CASSCF Routine - Setup and print header info to logger *
   ******************************************************************/

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


  // MCSCF Iteration format string
  const std::string fmt_string = 
    "iter = {:4} E(CI) = {:.10f}, dE = {:18.10e}, |orb_rms| = {:18.10e}";



  /*********************************************************
   *  Calculate persistant derived dimensions to be reused *
   *  throughout this routine                              *
   *********************************************************/

  const size_t no = norb.get(), ni = ninact.get(), na = nact.get(), 
               nv = nvirt.get();

  const size_t no2 = no  * no;
  const size_t no4 = no2 * no2;
  const size_t na2 = na  * na;
  const size_t na4 = na2 * na2;

  const size_t orb_rot_sz = nv*(ni + na) + na*ni;
  const double rms_factor = std::sqrt(orb_rot_sz);
  logger->info("  {:13} = {}","ORB_ROT_SZ", orb_rot_sz);






  /********************************************************
   *               Allocate persistant data               *
   ********************************************************/ 

  // Energies
  double E_inactive, E0;

  // Convergence data
  double grad_nrm;
  bool converged = false;

  // Storage for active space Hamitonian
  std::vector<double> T_active(na2), V_active(na4);

  // CI vector - will be resized on first CI call
  std::vector<double> X_CI; 

  // Orbital Gradient and Generalized Fock Matrix
  std::vector<double> F(no2), OG(orb_rot_sz), F_inactive(no2), 
    F_active(no2), Q(na * no);

  // Storage for transformed integrals
  std::vector<double> transT(T, T+no2), transV(V, V+no4);

  // Storage for total transformation
  std::vector<double> U_total(no2, 0.0), K_total(no2, 0.0);

  // DIIS Object
  DIIS<std::vector<double>> diis;






  /**************************************************************
   *    Precompute Active Space Hamiltonian given input data    *
   *                                                            *
   *     This will be used to compute initial energies and      *
   *      gradients to decide whether to proceed with the       *
   *                   MCSCF optimization.                      *
   **************************************************************/

  // Compute Active Space Hamiltonian and Inactive Fock Matrix
  active_hamiltonian(norb, nact, ninact, T, LDT, V, LDV, 
    F_inactive.data(), no, T_active.data(), na, 
    V_active.data(), na);

  // Compute Inactive Energy
  E_inactive = inactive_energy(ninact, T, LDT, F_inactive.data(), no);
  E_inactive += E_core;





  /**************************************************************
   *     Either compute or read initial RDMs from input         *
   *                                                            *
   * If the trace of the input 1RDM is != to the total number   *
   * of active electrons, RDMs will be computed, otherwise the  *
   *      input RDMs will be taken as an initial guess.         *
   **************************************************************/

  // Compute the trace of the input A1RDM
  double iAtr = 0.0;
  for(size_t i = 0; i < na; ++i) iAtr += A1RDM[i*(LDD1+1)];
  bool comp_rdms = std::abs(iAtr - nalpha.get() - nbeta.get()) > 1e-6; 

  if(comp_rdms) {
    // Compute active RDMs
    logger->info("Computing Initial RDMs");
    std::fill_n(A1RDM, na2, 0.0);
    std::fill_n(A2RDM, na4, 0.0);
    compute_casci_rdms<HamGen>(settings, NumOrbital(na), nalpha.get(), 
      nbeta.get(), T_active.data(), V_active.data(), A1RDM, A2RDM, X_CI,
      comm) + E_inactive;
  } else {
    logger->info("Using Passed RDMs");
  }


  /***************************************************************
   * Compute initial energy and gradient from computed (or read) * 
   * RDMs                                                        *
   ***************************************************************/

  // Compute Energy from RDMs
  double E_1RDM = blas::dot(na2, A1RDM, 1, T_active.data(), 1);
  double E_2RDM = blas::dot(na4, A2RDM, 1, V_active.data(), 1);

  E0 =  E_1RDM + E_2RDM + E_inactive; 
  logger->info("{:8} = {:20.12f}","E(1RDM)",E_1RDM);
  logger->info("{:8} = {:20.12f}","E(2RDM)",E_2RDM);
  logger->info("{:8} = {:20.12f}","E(CI)",E0);


  // Compute initial Fock and gradient
  active_fock_matrix(norb, ninact, nact, V, LDV, A1RDM, LDD1, 
    F_active.data(), no );
  aux_q_matrix(nact, norb, ninact, V, LDV, A2RDM, LDD2, 
    Q.data(), na);
  generalized_fock_matrix( norb, ninact, nact, F_inactive.data(), no, 
    F_active.data(), no, A1RDM, LDD1, Q.data(), na, F.data(), no);
  fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
    OG.data());





  /**************************************************************
   *      Compute initial Gradient norm and decide whether      *
   *           input data is sufficiently converged             *
   **************************************************************/

  grad_nrm = blas::nrm2(OG.size(), OG.data(), 1);
  converged = grad_nrm < settings.orb_grad_tol_mcscf;
  logger->info(fmt_string, 0, E0, 0.0, grad_nrm/rms_factor);





  /**************************************************************
   *                     MCSCF Iterations                       *
   **************************************************************/

  for(size_t iter = 0; iter < settings.max_macro_iter; ++iter) {

     // Check for convergence signal
     if(converged) break;

     // Save old data 
     const double E0_old = E0;
     std::vector<double> K_total_sav(K_total); 

    /************************************************************
     *                  Compute Orbital Step                    *
     ************************************************************/

     std::vector<double> K_step(no2);
#if 0
     std::fill(OG.begin(),OG.end(),0.0);
     optimize_orbitals(settings, norb, ninact, nact, nvirt, E_core, 
       transT.data(), no, transV.data(), no, A1RDM, LDD1, A2RDM, LDD2, 
       OG.data(), K_step.data(), no);
#else
     // Compute the step
     std::vector<double> K_step_linear(orb_rot_sz);
     precond_cg_orbital_step(norb, ninact, nact, nvirt, F_inactive.data(),
       no, F_active.data(), no, F.data(), no, A1RDM, LDD1, 
       OG.data(), K_step_linear.data());

     // Compute norms / max
     auto step_nrm  = blas::nrm2(orb_rot_sz, K_step_linear.data(), 1);
     auto step_amax = std::abs(
       K_step_linear[ blas::iamax(orb_rot_sz, K_step_linear.data(), 1) ] 
     );
     logger->debug("{:12}step_nrm = {:.4e}, step_amax = {:.4e}", "", 
       step_nrm, step_amax);

     // Scale step if necessacary 
     if( step_amax > 0.5 ) { 
       logger->info("  * decresing step from {:.2f} to {:.2f}", step_amax, 0.5);
       blas::scal(orb_rot_sz, 0.5 / step_amax, K_step_linear.data(), 1);
     }

     // Expand info full matrix
     linear_orb_rot_to_matrix(ninact, nact, nvirt, K_step_linear.data(),
       K_step.data(), no);
#endif

     // Increment total step
     blas::axpy(no2, 1.0, K_step.data(), 1, K_total.data(), 1);

     // DIIS Extrapolation
     if(iter > 2) {
       diis.add_vector(K_total, OG);
       if(iter > 4) {
         K_total = diis.extrapolate();
       }
     }



    /************************************************************
     *   Compute orbital rotation matrix corresponding to the   * 
     *                 total (accumulated) step                 *
     ************************************************************/
     if(!iter) {

       // If its the first iteration U_total = EXP[-K_total]
       compute_orbital_rotation(norb, 1.0, K_total.data(), no, 
         U_total.data(), no );

     } else {

       // Compute the rotation matrix for the *actual* step taken, 
       // accounting for possible extrapolation
       // 
       // U_step = EXP[-(K_total - K_total_old)]
       std::vector<double> U_step(no2);
       blas::axpy(no2, -1.0, K_total.data(), 1, K_total_sav.data(), 1);
       blas::scal(no2, -1.0, K_total_sav.data(), 1);
       compute_orbital_rotation(norb, 1.0, K_total_sav.data(), no, 
         U_step.data(), no );

       // U_total = U_total * U_step
       std::vector<double> tmp(no2);
       blas::gemm(blas::Layout::ColMajor,
         blas::Op::NoTrans, blas::Op::NoTrans,
         no, no, no, 
         1.0, U_total.data(), no, U_step.data(), no,
         0.0, tmp.data(), no
       );

       U_total = std::move(tmp);
     }


    /************************************************************
     *          Transform Hamiltonian into new MO basis         * 
     ************************************************************/
     two_index_transform(no, no, T, LDT, U_total.data(), no, 
       transT.data(), no);
     four_index_transform(no, no, 0, V, LDV, U_total.data(), no, 
       transV.data(), no);

     
    /************************************************************
     *      Compute Active Space Hamiltonian and associated     *
     *                    scalar quantities                     *
     ************************************************************/

     // Compute Active Space Hamiltonian + inactive Fock
     active_hamiltonian(norb, nact, ninact, transT.data(), no, transV.data(), no, 
      F_inactive.data(), no, T_active.data(), na, V_active.data(), na);

     // Compute Inactive Energy
     E_inactive = inactive_energy(ninact, transT.data(), no, 
       F_inactive.data(), no) + E_core;



    /************************************************************
     *       Compute new Active Space RDMs and GS energy        *
     ************************************************************/

     std::fill_n( A1RDM, na2, 0.0);
     std::fill_n( A2RDM, na4, 0.0);
     E0 = compute_casci_rdms<HamGen>(settings, NumOrbital(na), 
       nalpha.get(), nbeta.get(), T_active.data(), V_active.data(), A1RDM, 
       A2RDM, X_CI, comm) + E_inactive;

    /************************************************************
     *               Compute new Orbital Gradient               *
     ************************************************************/

    std::fill(F.begin(), F.end(), 0.0);
#if 0
    generalized_fock_matrix_comp_mat1( norb, ninact, nact, 
      F_inactive.data(), no, transV.data(), no, A1RDM, LDD1, A2RDM, LDD2, 
      F.data(), no);
#else
    // Update active fock + Q
    active_fock_matrix(norb, ninact, nact, transV.data(), no, A1RDM, LDD1, 
      F_active.data(), no );
    aux_q_matrix(nact, norb, ninact, transV.data(), no, A2RDM, LDD2, 
      Q.data(), na);

    generalized_fock_matrix( norb, ninact, nact, F_inactive.data(), no, 
      F_active.data(), no, A1RDM, LDD1, Q.data(), na, F.data(), no);
#endif
    fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
      OG.data());

    // Gradient Norm
    grad_nrm = blas::nrm2(OG.size(), OG.data(), 1);
    logger->info(fmt_string, iter+1, E0, E0 - E0_old, grad_nrm/rms_factor);

    converged = grad_nrm/rms_factor < settings.orb_grad_tol_mcscf;
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

    

  //auto L = davidson(orb_rot_sz, 100, op, DH.data(), 1e-8, X.data() );

  std::vector<double> iden(orb_rot_sz * orb_rot_sz), H(iden.size());
  for( auto i = 0; i < orb_rot_sz; ++i) iden[i*(orb_rot_sz+1)] = 1.0;
  for( auto i = 0; i < orb_rot_sz; ++i) {
    op.operator_action(1, 1.0, iden.data() + i*orb_rot_sz, orb_rot_sz,
      0.0, H.data() + i*orb_rot_sz, orb_rot_sz);
  }

  std::vector<double> num_OH(no4);
  numerical_orbital_hessian(norb, ninact, nact, T, LDT, V, LDV,
    A1RDM, LDD1, A2RDM, LDD2, num_OH.data(), no);

  const auto [vi_off, va_off, ai_off] = 
    split_linear_orb_rot(ninact,nact,nvirt, 0);

  // Virtual-Inactive / Virtual-Inactive
  for(size_t i = 0; i < ninact.get(); ++i)
  for(size_t a = 0; a < nvirt.get();  ++a) 
  for(size_t j = 0; j < ninact.get(); ++j)
  for(size_t b = 0; b < nvirt.get();  ++b) {
    const auto i_off = i;
    const auto a_off = a + ninact.get() + nact.get();
    const auto j_off = j;
    const auto b_off = b + ninact.get() + nact.get();

    const size_t ai_lin = a + i*nvirt.get() + vi_off;
    const size_t bj_lin = b + j*nvirt.get() + vi_off;

    auto calc = H[ai_lin + bj_lin*orb_rot_sz];
    auto num  = num_OH[a_off + i_off*no + b_off*no2 + j_off*no2*no];

    logger->info("VI {} {} VI {} {}, {:20.10e} {:20.10e} {:20.10e}",
      i, a, j, b, calc, num, std::abs(calc - num) );
  
  }

  logger->info("");
  // Virtual-Inactive / Virtual-Active
  for(size_t i = 0; i < ninact.get(); ++i)
  for(size_t a = 0; a < nvirt.get();  ++a) 
  for(size_t j = 0; j < nact.get(); ++j)
  for(size_t b = 0; b < nvirt.get();  ++b) {
    const auto i_off = i;
    const auto a_off = a + ninact.get() + nact.get();
    const auto j_off = j + ninact.get();
    const auto b_off = b + ninact.get() + nact.get();

    const size_t ai_lin = a + i*nvirt.get() + vi_off;
    const size_t bj_lin = b + j*nvirt.get() + va_off;

    auto calc = H[ai_lin + bj_lin*orb_rot_sz];
    auto num  = num_OH[a_off + i_off*no + b_off*no2 + j_off*no2*no];

    logger->info("VI {} {} VA {} {}, {:20.10e} {:20.10e} {:20.10e}",
      i, a, j, b, calc, num, std::abs(calc - num) );
  
  }

  logger->info("");
  // Virtual-Inactive / Active-Inactive  
  for(size_t i = 0; i < ninact.get(); ++i)
  for(size_t a = 0; a < nvirt.get();  ++a) 
  for(size_t j = 0; j < ninact.get(); ++j)
  for(size_t b = 0; b < nact.get();  ++b) {
    const auto i_off = i;
    const auto a_off = a + ninact.get() + nact.get();
    const auto j_off = j;
    const auto b_off = b + ninact.get();

    const size_t ai_lin = a + i*nvirt.get() + vi_off;
    const size_t bj_lin = b + j*nact.get()  + ai_off;

    auto calc = H[ai_lin + bj_lin*orb_rot_sz];
    auto num  = num_OH[a_off + i_off*no + b_off*no2 + j_off*no2*no];

    logger->info("VI {} {} AI {} {}, {:20.10e} {:20.10e} {:20.10e}",
      i, a, j, b, calc, num, std::abs(calc - num) );
  
  }


  // Virtual-Active / Virtual-Inactive
  for(size_t i = 0; i < nact.get(); ++i)
  for(size_t a = 0; a < nvirt.get();  ++a) 
  for(size_t j = 0; j < ninact.get(); ++j)
  for(size_t b = 0; b < nvirt.get();  ++b) {
    const auto i_off = i + ninact.get();
    const auto a_off = a + ninact.get() + nact.get();
    const auto j_off = j;
    const auto b_off = b + ninact.get() + nact.get();

    const size_t ai_lin = a + i*nvirt.get() + va_off;
    const size_t bj_lin = b + j*nvirt.get() + vi_off;

    auto calc = H[ai_lin + bj_lin*orb_rot_sz];
    auto num  = num_OH[a_off + i_off*no + b_off*no2 + j_off*no2*no];

    logger->info("VA {} {} VI {} {}, {:20.10e} {:20.10e} {:20.10e}",
      i, a, j, b, calc, num, std::abs(calc - num) );
  
  }

  logger->info("");
  // Virtual-Active / Virtual-Active
  for(size_t i = 0; i < nact.get(); ++i)
  for(size_t a = 0; a < nvirt.get();  ++a) 
  for(size_t j = 0; j < nact.get(); ++j)
  for(size_t b = 0; b < nvirt.get();  ++b) {
    const auto i_off = i + ninact.get();
    const auto a_off = a + ninact.get() + nact.get();
    const auto j_off = j + ninact.get();
    const auto b_off = b + ninact.get() + nact.get();

    const size_t ai_lin = a + i*nvirt.get() + va_off;
    const size_t bj_lin = b + j*nvirt.get() + va_off;

    auto calc = H[ai_lin + bj_lin*orb_rot_sz];
    auto num  = num_OH[a_off + i_off*no + b_off*no2 + j_off*no2*no];

    logger->info("VA {} {} VA {} {}, {:20.10e} {:20.10e} {:20.10e}",
      i, a, j, b, calc, num, std::abs(calc - num) );
  
  }

  logger->info("");
  // Virtual-Active / Active-Inactive  
  for(size_t i = 0; i < nact.get(); ++i)
  for(size_t a = 0; a < nvirt.get();  ++a) 
  for(size_t j = 0; j < ninact.get(); ++j)
  for(size_t b = 0; b < nact.get();  ++b) {
    const auto i_off = i + ninact.get();
    const auto a_off = a + ninact.get() + nact.get();
    const auto j_off = j;
    const auto b_off = b + ninact.get();

    const size_t ai_lin = a + i*nvirt.get() + va_off;
    const size_t bj_lin = b + j*nact.get()  + ai_off;

    auto calc = H[ai_lin + bj_lin*orb_rot_sz];
    auto num  = num_OH[a_off + i_off*no + b_off*no2 + j_off*no2*no];

    logger->info("VA {} {} AI {} {}, {:20.10e} {:20.10e} {:20.10e}",
      i, a, j, b, calc, num, std::abs(calc - num) );
  
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
