#include <asci/util/orbital_gradient.hpp>
#include <asci/util/transform.hpp>
#include <asci/util/fock_matrices.hpp>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

namespace asci {

void compute_orbital_rotation(NumOrbital _norb, double alpha, const double* K, 
  size_t LDK, double* U, size_t LDU) {

  const size_t norb = _norb.get();
  using mat_t = Eigen::MatrixXd;
  using map_t = Eigen::Map<mat_t>;
  using cmap_t = Eigen::Map<const mat_t>;

  cmap_t K_map(K, LDK, norb);
  map_t  U_map(U, LDU, norb);

  // U = EXP[-K] (Pink book 10.1.8)
  U_map.block(0,0,norb,norb) = (-alpha * K_map.block(0,0,norb,norb)).exp();

}


void orbital_rotated_generalized_fock(NumOrbital _norb, NumInactive _ninact, 
  NumActive _nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2, 
  const double* U, size_t LDU, double* T_trans, size_t LDTT, 
  double* V_trans, size_t LDVT, double* F, size_t LDF) {

  const auto norb   = _norb.get();
  const auto nact   = _nact.get();
  const auto ninact = _ninact.get();

  // Transform Integrals
  two_index_transform( norb, norb, T, LDT, U, LDU,
    T_trans, LDTT);
  four_index_transform( norb, norb, 0, V, LDV, U ,LDU,
    V_trans, LDVT);

  // Compute Fock Matrix
  generalized_fock_matrix_comp_mat2(_norb, _ninact, _nact,
    T_trans, LDTT, V_trans, LDVT, A1RDM, LDD1, A2RDM,
    LDD2, F, LDF);

}


double orbital_rotated_energy(NumOrbital norb, NumInactive ninact, 
  NumActive nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2, 
  const double* U, size_t LDU, double* T_trans, size_t LDTT, 
  double* V_trans, size_t LDVT, double* F, size_t LDF) {

  orbital_rotated_generalized_fock(norb, ninact, nact, T, LDT,
    V, LDV, A1RDM, LDD1, A2RDM, LDD2, U, LDU, T_trans, LDTT,
    V_trans, LDVT, F, LDF);

  // Compute energy
  return energy_from_generalized_fock(ninact, nact,
    T_trans, LDTT, A1RDM, LDD1, F, LDF);
}

double orbital_rotated_energy(NumOrbital _norb, NumInactive ninact, 
  NumActive nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2, 
  const double* U, size_t LDU) {

  const auto norb = _norb.get();
  size_t norb2 = norb  * norb;
  size_t norb4 = norb2 * norb2;
  std::vector<double> T_trans(norb2), V_trans(norb4),
    F(norb2);

  return orbital_rotated_energy(_norb, ninact, nact, T, LDT,
    V, LDV, A1RDM, LDD1, A2RDM, LDD2, U, LDU, T_trans.data(), 
    norb, V_trans.data(), norb, F.data(), norb);

}

void numerical_orbital_gradient(NumOrbital _norb, 
  NumInactive ninact, NumActive nact, const double* T, size_t LDT,
  const double* V, size_t LDV, const double* A1RDM,
  size_t LDD1, const double* A2RDM, size_t LDD2,
  double* OG, size_t LDOG ) { 


  const auto norb = _norb.get();

  const size_t norb2 = norb  * norb;
  const size_t norb4 = norb2 * norb2;
  std::vector<double> F(norb2), T_trans(norb2), 
    V_trans(norb4), K(norb2), U(norb2);

  auto energy = [&]() {
    return orbital_rotated_energy(_norb, ninact, nact, 
      T, LDT, V, LDV, A1RDM, LDD1, A2RDM, LDD2, U.data(), norb,
      T_trans.data(), norb, V_trans.data(), norb, F.data(),
      norb);
  };
  
  const double dk = 0.001;
  for(size_t i = 0;   i < norb; ++i) 
  for(size_t a = i+1; a < norb; ++a) {
    std::fill(K.begin(), K.end(), 0);
    K[a + i*norb] = dk;
    K[i + a*norb] = -dk;

    compute_orbital_rotation(_norb, 2.0, K.data(), norb, U.data(), norb);
    auto E_p2 = energy();

    compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
    auto E_p1 = energy();
  
    //U_map = (K_map).exp();
    compute_orbital_rotation(_norb, -1.0, K.data(), norb, U.data(), norb);
    auto E_m1 = energy();

    compute_orbital_rotation(_norb, -2.0, K.data(), norb, U.data(), norb);
    auto E_m2 = energy();

    OG[a + i*norb] = (E_m2-E_p2 + 8*(E_p1-E_m1))/(12*dk);
  }

}

}
