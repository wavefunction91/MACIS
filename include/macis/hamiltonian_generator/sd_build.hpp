/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator.hpp>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>
#include <set>

namespace macis {

template <size_t N>
struct det_pos {
 public:
  std::bitset<N> det;
  uint32_t id;
};

template <size_t N>
bool operator<(const det_pos<N>& a, const det_pos<N>& b) {
  return bitset_less<N>(a.det, b.det);
}

template <size_t N>
class SDBuildHamiltonianGenerator : public HamiltonianGenerator<N> {
 public:
  using base_type = HamiltonianGenerator<N>;
  using full_det_t = typename base_type::full_det_t;
  using spin_det_t = typename base_type::spin_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;
  using matrix_span_t = typename base_type::matrix_span_t;
  using rank4_span_t = typename base_type::rank4_span_t;

  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

 protected:
  size_t nimp, nimp2, nimp3;

  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    std::vector<index_t> colind, rowptr(nbra_dets + 1);
    std::vector<double> nzval;

    // List of impurity orbitals, assumed to be the first nimp.
    std::vector<uint32_t> imp_orbs(nimp, 0);
    for(int ii = 0; ii < nimp; ii++) imp_orbs[ii] = ii;
    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    std::set<det_pos<N> > kets;
    for(full_det_iterator it = ket_begin; it != ket_end; it++) {
      det_pos<N> a;
      a.det = *it;
      a.id = std::distance(ket_begin, it);
      kets.insert(a);
    }

    rowptr[0] = 0;

    // Loop over bra determinants
    for(size_t i = 0; i < nbra_dets; ++i) {
      // if( (i%1000) == 0 ) std::cout << i << ", " << rowptr[i] << std::endl;
      const auto bra = *(bra_begin + i);

      size_t nrow = 0;
      if(bra.count()) {
        // Separate out into alpha/beta components
        spin_det_t bra_alpha = bitset_lo_word(bra);
        spin_det_t bra_beta = bitset_hi_word(bra);

        // Get occupied indices
        bits_to_indices(bra_alpha, bra_occ_alpha);
        bits_to_indices(bra_beta, bra_occ_beta);

        // Get singles and doubles
        // (Note that doubles only involve impurity orbitals)
        std::vector<full_det_t> excs, doubles;
        if(just_singles)
          generate_singles_spin(this->norb_, bra, excs);
        else {
          std::vector<full_det_t> singls;
          generate_singles_spin(this->norb_, bra, excs);
          // This will store in singls sinles among impurity orbitals, which we
          // have already taken into account.
          generate_singles_doubles_spin_as(this->norb_, bra, singls, doubles,
                                           imp_orbs);
          excs.insert(excs.end(), doubles.begin(), doubles.end());
        }

        // Diagonal term
        full_det_t ex_diag = bra ^ bra;
        spin_det_t exd_alpha = bitset_lo_word(ex_diag);
        spin_det_t exd_beta = bitset_hi_word(ex_diag);

        // Compute Matrix Element
        const auto h_eld =
            this->matrix_element_diag(bra_occ_alpha, bra_occ_beta);

        if(std::abs(h_eld) > H_thresh) {
          nrow++;
          colind.emplace_back(i);
          nzval.emplace_back(h_eld);
        }

        // Loop over ket determinants
        for(const auto pos_ket : excs) {
          det_pos<N> pos_ket2;
          pos_ket2.det = pos_ket;
          pos_ket2.id = 0;
          auto it = kets.find(pos_ket2);
          if(it != kets.end()) {
            int j = it->id;
            spin_det_t ket_alpha = bitset_lo_word(pos_ket);
            spin_det_t ket_beta = bitset_hi_word(pos_ket);

            full_det_t ex_total = bra ^ pos_ket;

            spin_det_t ex_alpha = bitset_lo_word(ex_total);
            spin_det_t ex_beta = bitset_hi_word(ex_total);

            // Compute Matrix Element
            const auto h_el = this->matrix_element(
                bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                bra_occ_alpha, bra_occ_beta);

            if(std::abs(h_el) > H_thresh) {
              nrow++;
              colind.emplace_back(j);
              nzval.emplace_back(h_el);
            }

          }  // Non-zero ket determinant
        }    // Loop over ket determinants

      }  // Non-zero bra determinant

      rowptr[i + 1] = rowptr[i] + nrow;  // Update rowptr

    }  // Loop over bra determinants

    colind.shrink_to_fit();
    nzval.shrink_to_fit();

    return sparse_matrix_type<index_t>(nbra_dets, nket_dets, std::move(rowptr),
                                       std::move(colind), std::move(nzval));
  }

  sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int32_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

  sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int64_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

 public:
  void form_rdms(full_det_iterator bra_begin, full_det_iterator bra_end,
                 full_det_iterator ket_begin, full_det_iterator ket_end,
                 double* C, matrix_span_t ordm, rank4_span_t trdm) override {
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    std::set<det_pos<N> > kets;
    for(full_det_iterator it = ket_begin; it != ket_end; it++) {
      det_pos<N> a;
      a.det = *it;
      a.id = std::distance(ket_begin, it);
      kets.insert(a);
    }

    // Loop over bra determinants
    for(size_t i = 0; i < nbra_dets; ++i) {
      const auto bra = *(bra_begin + i);
      // if( (i%1000) == 0 ) std::cout << i  << std::endl;
      if(bra.count()) {
        // Separate out into alpha/beta components
        spin_det_t bra_alpha = bitset_lo_word(bra);
        spin_det_t bra_beta = bitset_hi_word(bra);

        // Get occupied indices
        bits_to_indices(bra_alpha, bra_occ_alpha);
        bits_to_indices(bra_beta, bra_occ_beta);

        // Get singles and doubles
        std::vector<full_det_t> excs;
        if(trdm.data_handle()) {
          std::vector<full_det_t> doubles;
          generate_singles_doubles_spin(this->norb_, bra, excs, doubles);
          excs.insert(excs.end(), doubles.begin(), doubles.end());
        } else {
          generate_singles_spin(this->norb_, bra, excs);
        }

        // Diagonal term
        full_det_t ex_diag = bra ^ bra;
        spin_det_t exd_alpha = bitset_lo_word(ex_diag);
        spin_det_t exd_beta = bitset_hi_word(ex_diag);

        // Compute Matrix Element
        rdm_contributions(bra_alpha, bra_alpha, exd_alpha, bra_beta, bra_beta,
                          exd_beta, bra_occ_alpha, bra_occ_beta, C[i] * C[i],
                          ordm, trdm);

        // Loop over excitations
        for(const auto pos_ket : excs) {
          det_pos<N> pos_ket2;
          pos_ket2.det = pos_ket;
          pos_ket2.id = 0;
          auto it = kets.find(pos_ket2);
          if(it != kets.end()) {
            int j = it->id;
            spin_det_t ket_alpha = bitset_lo_word(pos_ket);
            spin_det_t ket_beta = bitset_hi_word(pos_ket);

            full_det_t ex_total = bra ^ pos_ket;
            int ex_lim = 2;
            if(trdm.data_handle()) ex_lim = 4;
            if(ex_total.count() <= ex_lim) {
              spin_det_t ex_alpha = bitset_lo_word(ex_total);
              spin_det_t ex_beta = bitset_hi_word(ex_total);

              const double val = C[i] * C[j];

              // Compute Matrix Element
              rdm_contributions(bra_alpha, ket_alpha, ex_alpha, bra_beta,
                                ket_beta, ex_beta, bra_occ_alpha, bra_occ_beta,
                                val, ordm, trdm);

            }  // Possible non-zero connection (Hamming distance)

          }  // Non-zero ket determinant
        }    // Loop over ket determinants

      }  // Non-zero bra determinant
    }    // Loop over bra determinants
  }

 public:
  bool just_singles;

  template <typename... Args>
  SDBuildHamiltonianGenerator(Args&&... args)
      : HamiltonianGenerator<N>(std::forward<Args>(args)...),
        just_singles(false) {
    SetNimp(this->norb_);
  }

  void SetJustSingles(bool _js) override { just_singles = _js; }
  void SetNimp(size_t _n) {
    nimp = _n;
    nimp2 = _n * _n;
    nimp3 = nimp2 * _n;
  }
  size_t GetNimp() const override { return nimp; }
  bool GetJustSingles() const override { return just_singles; }
};

}  // namespace macis
