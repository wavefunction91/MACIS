#include <asci/util/orbital_gradient.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/fock_matrices.hpp>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

namespace asci {


double orbital_rotated_energy(size_t norb, size_t ninact, 
  size_t nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2, 
  const double* U, size_t LDU, double* T_trans, size_t LDTT, 
  double* V_trans, size_t LDVT, double* F, size_t LDF) {

  // Transform Integrals
  two_index_transform( norb, norb, T, LDT, U, LDU,
    T_trans, LDTT);
  four_index_transform( norb, norb, 0, V, LDV, U ,LDU,
    V_trans, LDVT);

  // Compute Fock Matrix
  generalized_fock_matrix_comp_mat2(norb, ninact, nact,
    T_trans, LDTT, V_trans, LDVT, A1RDM, LDD1, A2RDM,
    LDD2, F, LDF);

  // Compute energy
  return energy_from_generalized_fock(ninact, nact,
    T_trans, LDTT, A1RDM, LDD1, F, LDF);

}

void numerical_orbital_gradient(size_t norb, 
  size_t ninact, size_t nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2,
  double* OG, size_t LDOG ) { 

  using mat_t = Eigen::MatrixXd;
  using map_t = Eigen::Map<mat_t>;

  const size_t norb2 = norb  * norb;
  const size_t norb4 = norb2 * norb2;
  std::vector<double> F(norb2), T_trans(norb2), 
    V_trans(norb4), K(norb2), U(norb2);

  map_t K_map(K.data(), norb, norb);
  map_t U_map(U.data(), norb, norb);
  map_t OG_map(OG, LDOG, norb);

  auto energy = [&]() {
    return orbital_rotated_energy(norb, ninact, nact, 
      T, LDT, V, LDV, A1RDM, LDD1, A2RDM, LDD2, U.data(), norb,
      T_trans.data(), norb, V_trans.data(), norb, F.data(),
      norb);
  };
  
  const double dk = 0.001;
  for(size_t i = 0;   i < norb; ++i) 
  for(size_t a = i+1; a < norb; ++a) {
    K_map = mat_t::Zero(norb,norb);
    // a > i, so K(a,i) >= 0 (Pink book 10.1.8,9)
    K_map(a,i) = dk;
    K_map(i,a) = -dk;
    // U = EXP[-K] (Pink book 10.1.8)

    U_map = (-2.*K_map).exp();
    auto E_p2 = energy();

    U_map = (-K_map).exp();
    auto E_p1 = energy();
  
    U_map = (K_map).exp();
    auto E_m1 = energy();

    U_map = (2.*K_map).exp();
    auto E_m2 = energy();

    OG_map(a,i) = (E_m2-E_p2 + 8*(E_p1-E_m1))/(12*dk);
  }

}

}
