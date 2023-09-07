/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>

namespace macis {

template <typename WfnType>
class HamiltonianGenerator {

 public:

  using full_det_t = WfnType;
  using spin_det_t = spin_wfn_t<WfnType>;

  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

  using full_det_container = std::vector<WfnType>;
  using full_det_iterator = typename full_det_container::iterator;

  using matrix_span_t = matrix_span<double>;
  using rank3_span_t = rank3_span<double>;
  using rank4_span_t = rank4_span<double>;

 public:

  size_t norb_;
  size_t norb2_;
  size_t norb3_;
  matrix_span_t T_pq_;
  rank4_span_t V_pqrs_;

  // G(i,j,k,l) = (ij|kl) - (il|kj)
  std::vector<double> G_pqrs_data_;
  rank4_span_t G_pqrs_;

  // G_red(i,j,k) = G(i,j,k,k)
  std::vector<double> G_red_data_;
  rank3_span_t G_red_;

  // V_red(i,j,k) = (ij|kk)
  std::vector<double> V_red_data_;
  rank3_span_t V_red_;

  // G2_red(i,j)  = 0.5 * G(i,i,j,j)
  std::vector<double> G2_red_data_;
  matrix_span_t G2_red_;

  // V2_red(i,j)  = (ii|jj)
  std::vector<double> V2_red_data_;
  matrix_span_t V2_red_;

  virtual sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator, full_det_iterator, full_det_iterator,
      full_det_iterator, double) = 0;

  virtual sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator, full_det_iterator, full_det_iterator,
      full_det_iterator, double) = 0;

 public:
  HamiltonianGenerator(matrix_span_t T, rank4_span_t V);
  virtual ~HamiltonianGenerator() noexcept = default;

  void generate_integral_intermediates(rank4_span_t V);

  inline auto* T() const { return T_pq_.data_handle(); }
  inline auto* G_red() const { return G_red_data_.data(); }
  inline auto* V_red() const { return V_red_data_.data(); }
  inline auto* G() const { return G_pqrs_data_.data(); }
  inline auto* V() const { return V_pqrs_.data_handle(); }

  double matrix_element_4(spin_det_t bra, spin_det_t ket, spin_det_t ex) const;
  double matrix_element_22(spin_det_t bra_alpha, spin_det_t ket_alpha,
                           spin_det_t ex_alpha, spin_det_t bra_beta,
                           spin_det_t ket_beta, spin_det_t ex_beta) const;

  double matrix_element_2(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                          const std::vector<uint32_t>& bra_occ_alpha,
                          const std::vector<uint32_t>& bra_occ_beta) const;

  double matrix_element_diag(const std::vector<uint32_t>& occ_alpha,
                             const std::vector<uint32_t>& occ_beta) const;

  double matrix_element(spin_det_t bra_alpha, spin_det_t ket_alpha,
                        spin_det_t ex_alpha, spin_det_t bra_beta,
                        spin_det_t ket_beta, spin_det_t ex_beta,
                        const std::vector<uint32_t>& bra_occ_alpha,
                        const std::vector<uint32_t>& bra_occ_beta) const;

  double single_orbital_en(uint32_t orb, const std::vector<uint32_t>& ss_occ,
                           const std::vector<uint32_t>& os_occ) const;

  std::vector<double> single_orbital_ens(
      size_t norb, const std::vector<uint32_t>& ss_occ,
      const std::vector<uint32_t>& os_occ) const;

  double fast_diag_single(const std::vector<uint32_t>& ss_occ,
                          const std::vector<uint32_t>& os_occ, uint32_t orb_hol,
                          uint32_t orb_par, double orig_det_Hii) const;

  double fast_diag_single(double hol_en, double par_en, uint32_t orb_hol,
                          uint32_t orb_par, double orig_det_Hii) const;

  double fast_diag_ss_double(double en_hol1, double en_hol2, double en_par1,
                             double en_par2, uint32_t orb_hol1,
                             uint32_t orb_hol2, uint32_t orb_par1,
                             uint32_t orb_par2, double orig_det_Hii) const;

  double fast_diag_ss_double(const std::vector<uint32_t>& ss_occ,
                             const std::vector<uint32_t>& os_occ,
                             uint32_t orb_hol1, uint32_t orb_hol2,
                             uint32_t orb_par1, uint32_t orb_par2,
                             double orig_det_Hii) const;

  double fast_diag_os_double(double en_holu, double en_hold, double en_paru,
                             double en_pard, uint32_t orb_holu,
                             uint32_t orb_hold, uint32_t orb_paru,
                             uint32_t orb_pard, double orig_det_Hii) const;

  double fast_diag_os_double(const std::vector<uint32_t>& up_occ,
                             const std::vector<uint32_t>& do_occ,
                             uint32_t orb_holu, uint32_t orb_hold,
                             uint32_t orb_paru, uint32_t orb_pard,
                             double orig_det_Hii) const;

  double matrix_element(full_det_t bra, full_det_t ket) const;

  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    if constexpr(std::is_same_v<index_t, int32_t>)
      return make_csr_hamiltonian_block_32bit_(bra_begin, bra_end, ket_begin,
                                               ket_end, H_thresh);
    else if constexpr(std::is_same_v<index_t, int64_t>)
      return make_csr_hamiltonian_block_64bit_(bra_begin, bra_end, ket_begin,
                                               ket_end, H_thresh);
    else {
      throw std::runtime_error("Unsupported index_t");
      abort();
    }
  }

  void rdm_contributions_4(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                           double val, rank4_span_t trdm);
  void rdm_contributions_22(spin_det_t bra_alpha, spin_det_t ket_alpha,
                            spin_det_t ex_alpha, spin_det_t bra_beta,
                            spin_det_t ket_beta, spin_det_t ex_beta, double val,
                            rank4_span_t trdm);
  void rdm_contributions_2(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                           const std::vector<uint32_t>& bra_occ_alpha,
                           const std::vector<uint32_t>& bra_occ_beta,
                           double val, matrix_span_t ordm, rank4_span_t trdm);
  void rdm_contributions_diag(const std::vector<uint32_t>& occ_alpha,
                              const std::vector<uint32_t>& occ_beta, double val,
                              matrix_span_t ordm, rank4_span_t trdm);

  void rdm_contributions(spin_det_t bra_alpha, spin_det_t ket_alpha,
                         spin_det_t ex_alpha, spin_det_t bra_beta,
                         spin_det_t ket_beta, spin_det_t ex_beta,
                         const std::vector<uint32_t>& bra_occ_alpha,
                         const std::vector<uint32_t>& bra_occ_beta, double val,
                         matrix_span_t ordm, rank4_span_t trdm);

  virtual void form_rdms(full_det_iterator, full_det_iterator,
                         full_det_iterator, full_det_iterator, double* C,
                         matrix_span_t ordm, rank4_span_t trdm) = 0;

  void rotate_hamiltonian_ordm(const double* ordm);

  virtual void SetJustSingles(bool /*_js*/) {}
  virtual bool GetJustSingles() { return false; }
  //virtual size_t GetNimp() const { return N / 2; }
};

}  // namespace macis

// Implementation
#include <macis/hamiltonian_generator/impl.hpp>
