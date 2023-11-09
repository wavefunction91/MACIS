/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <chrono>
#include <macis/hamiltonian_generator.hpp>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>

namespace macis {

template <typename WfnType>
class SortedDoubleLoopHamiltonianGenerator
    : public HamiltonianGenerator<WfnType> {
 public:
  using base_type = HamiltonianGenerator<WfnType>;
  using full_det_t = typename base_type::full_det_t;
  using spin_det_t = typename base_type::spin_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;
  using matrix_span_t = typename base_type::matrix_span_t;
  using rank4_span_t = typename base_type::rank4_span_t;

  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

 protected:
  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    using wfn_traits = wavefunction_traits<WfnType>;
    using spin_wfn_type = typename wfn_traits::spin_wfn_type;
    using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    const bool is_symm = bra_begin == ket_begin and bra_end == ket_end;

    // Get unique alpha strings
    auto setup_st = std::chrono::high_resolution_clock::now();
    auto unique_alpha_bra = get_unique_alpha(bra_begin, bra_end);
    auto unique_alpha_ket =
        is_symm ? unique_alpha_bra : get_unique_alpha(ket_begin, ket_end);

    const size_t nuniq_bra = unique_alpha_bra.size();
    const size_t nuniq_ket = unique_alpha_ket.size();

    // Compute offsets
    std::vector<size_t> unique_alpha_bra_idx(nuniq_bra + 1);
    std::transform_exclusive_scan(
        unique_alpha_bra.begin(), unique_alpha_bra.end(),
        unique_alpha_bra_idx.begin(), 0ul, std::plus<size_t>{},
        [](auto& x) { return x.second; });
    std::vector<size_t> unique_alpha_ket_idx(nuniq_ket + 1);
    if(is_symm) {
      unique_alpha_ket_idx = unique_alpha_bra_idx;
    } else {
      std::transform_exclusive_scan(
          unique_alpha_ket.begin(), unique_alpha_ket.end(),
          unique_alpha_ket_idx.begin(), 0ul, std::plus<size_t>{},
          [](auto& x) { return x.second; });
    }

    unique_alpha_bra_idx.back() = nbra_dets;
    unique_alpha_ket_idx.back() = nket_dets;
    auto setup_en = std::chrono::high_resolution_clock::now();

    // std::cout << "AVERAGE NBETA = " <<
    //   std::accumulate(unique_alpha_bra.begin(), unique_alpha_bra.end(),
    //     0ul, [](auto a, auto b){ return a + b.second; }) / double(nuniq_bra)
    //     << std::endl;

    // Populate COO matrix locally
    // sparsexx::coo_matrix<double, index_t> coo_mat(nbra_dets, nket_dets, 0,
    // 0);
    std::vector<index_t> row_ind, col_ind;
    std::vector<double> nz_val;

    // size_t skip1 = 0;
    // size_t skip2 = 0;

    std::mutex coo_mat_thread_mutex;

    // Loop over uniq alphas in bra/ket
    auto pop_st = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
      std::vector<index_t> row_ind_loc, col_ind_loc;
      std::vector<double> nz_val_loc;
      std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;
#pragma omp for schedule(dynamic)
      for(size_t ia_bra = 0; ia_bra < nuniq_bra; ++ia_bra) {
        if(unique_alpha_bra[ia_bra].first.any()) {
          // Extract alpha bra
          const auto bra_alpha = unique_alpha_bra[ia_bra].first;
          const size_t beta_st_bra = unique_alpha_bra_idx[ia_bra];
          const size_t beta_en_bra = unique_alpha_bra_idx[ia_bra + 1];
          spin_wfn_traits::state_to_occ(bra_alpha, bra_occ_alpha);

          const auto ket_lower = is_symm ? ia_bra : 0;
          for(size_t ia_ket = ket_lower; ia_ket < nuniq_ket; ++ia_ket) {
            if(unique_alpha_ket[ia_ket].first.any()) {
              // Extract alpha ket
              const auto ket_alpha = unique_alpha_ket[ia_ket].first;

              // Compute alpha excitation
              const auto ex_alpha = bra_alpha ^ ket_alpha;
              const auto ex_alpha_count = spin_wfn_traits::count(ex_alpha);

              // Early exit
              if(ex_alpha_count > 4) {
                // skip1++;
                continue;
              }

              // Precompute all-alpha excitation if it will be used
              const double mat_el_4_alpha =
                  (ex_alpha_count == 4)
                      ? this->matrix_element_4(bra_alpha, ket_alpha, ex_alpha)
                      : 0.0;
              if(ex_alpha_count == 4 and std::abs(mat_el_4_alpha) < H_thresh) {
                // The only possible matrix element is too-small, skip everyhing
                // skip2++;
                continue;
              }

              const size_t beta_st_ket = unique_alpha_ket_idx[ia_ket];
              const size_t beta_en_ket = unique_alpha_ket_idx[ia_ket + 1];

              // Loop over local betas according to their global indices
              for(size_t ibra = beta_st_bra; ibra < beta_en_bra; ++ibra) {
                const auto bra_beta =
                    wfn_traits::beta_string(*(bra_begin + ibra));
                spin_wfn_traits::state_to_occ(bra_beta, bra_occ_beta);
                for(size_t iket = beta_st_ket; iket < beta_en_ket; ++iket) {
                  if(is_symm and (iket < ibra)) continue;
                  const auto ket_beta =
                      wfn_traits::beta_string(*(ket_begin + iket));

                  // Compute beta excitation
                  const auto ex_beta = bra_beta ^ ket_beta;
                  const auto ex_beta_count = spin_wfn_traits::count(ex_beta);

                  if((ex_alpha_count + ex_beta_count) > 4) continue;

                  double h_el = 0.0;
                  if(ex_alpha_count == 4) {
                    // Use precomputed value
                    h_el = mat_el_4_alpha;
                  } else if(ex_beta_count == 4) {
                    h_el = this->matrix_element_4(bra_beta, ket_beta, ex_beta);
                  } else if(ex_alpha_count == 2) {
                    if(ex_beta_count == 2) {
                      h_el = this->matrix_element_22(bra_alpha, ket_alpha,
                                                     ex_alpha, bra_beta,
                                                     ket_beta, ex_beta);
                    } else {
                      h_el =
                          this->matrix_element_2(bra_alpha, ket_alpha, ex_alpha,
                                                 bra_occ_alpha, bra_occ_beta);
                    }
                  } else if(ex_beta_count == 2) {
                    h_el = this->matrix_element_2(bra_beta, ket_beta, ex_beta,
                                                  bra_occ_beta, bra_occ_alpha);
                  } else {
                    // Diagonal matrix element
                    h_el =
                        this->matrix_element_diag(bra_occ_alpha, bra_occ_beta);
                  }

                  // Insert matrix element
                  if(std::abs(h_el) > H_thresh) {
                    // coo_mat.template insert<false>(ibra, iket, h_el);
                    row_ind_loc.emplace_back(ibra);
                    col_ind_loc.emplace_back(iket);
                    nz_val_loc.emplace_back(h_el);
                    if(is_symm and ibra != iket) {
                      // coo_mat.template insert<false>(iket, ibra, h_el);
                      row_ind_loc.emplace_back(iket);
                      col_ind_loc.emplace_back(ibra);
                      nz_val_loc.emplace_back(h_el);
                    }
                  }

                }  // ket beta
              }    // bra beta
            }
          }  // Loop over ket alphas
        }
      }  // Loop over bra alphas

// Atomically insert into larger matrix arrays
#pragma omp critical
      {
        row_ind.insert(row_ind.end(), row_ind_loc.begin(), row_ind_loc.end());
        // row_ind_loc.clear(); row_ind_loc.shrink_to_fit();
        col_ind.insert(col_ind.end(), col_ind_loc.begin(), col_ind_loc.end());
        // col_ind_loc.clear(); col_ind_loc.shrink_to_fit();
        nz_val.insert(nz_val.end(), nz_val_loc.begin(), nz_val_loc.end());
        // nz_val_loc.clear(); nz_val_loc.shrink_to_fit();
      }

    }  // OpenMP
    auto pop_en = std::chrono::high_resolution_clock::now();

    // Generate Sparse Matrix
    sparsexx::coo_matrix<double, index_t> coo_mat(
        nbra_dets, nket_dets, std::move(col_ind), std::move(row_ind),
        std::move(nz_val), 0);

    // Sort for CSR Conversion
    auto sort_st = std::chrono::high_resolution_clock::now();
    coo_mat.sort_by_row_index();
    auto sort_en = std::chrono::high_resolution_clock::now();

    auto conv_st = std::chrono::high_resolution_clock::now();
    sparse_matrix_type<index_t> csr_mat(coo_mat);  // Convert to CSR Matrix
    auto conv_en = std::chrono::high_resolution_clock::now();

    printf("Setup %.2e Pop %.2e Sort %.2e Conv %.2e\n",
           std::chrono::duration<double>(setup_en - setup_st).count(),
           std::chrono::duration<double>(pop_en - pop_st).count(),
           std::chrono::duration<double>(sort_en - sort_st).count(),
           std::chrono::duration<double>(conv_en - conv_st).count());

    return csr_mat;
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
    using wfn_traits = wavefunction_traits<WfnType>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    // Loop over bra determinants
    for(size_t i = 0; i < nbra_dets; ++i) {
      const auto bra = *(bra_begin + i);
      // if( (i%1000) == 0 ) std::cout << i  << std::endl;
      if(wfn_traits::count(bra)) {
        // Separate out into alpha/beta components
        spin_det_t bra_alpha = wfn_traits::alpha_string(bra);
        spin_det_t bra_beta = wfn_traits::beta_string(bra);

        // Get occupied indices
        bits_to_indices(bra_alpha, bra_occ_alpha);
        bits_to_indices(bra_beta, bra_occ_beta);

        // Loop over ket determinants
        for(size_t j = 0; j < nket_dets; ++j) {
          const auto ket = *(ket_begin + j);
          if(wfn_traits::count(ket)) {
            spin_det_t ket_alpha = wfn_traits::alpha_string(ket);
            spin_det_t ket_beta = wfn_traits::beta_string(ket);

            full_det_t ex_total = bra ^ ket;
            if(wfn_traits::count(ex_total) <= 4) {
              spin_det_t ex_alpha = wfn_traits::alpha_string(ex_total);
              spin_det_t ex_beta = wfn_traits::beta_string(ex_total);

              const double val = C[i] * C[j];

              // Compute Matrix Element
              if(std::abs(val) > 1e-16) {
                rdm_contributions(bra_alpha, ket_alpha, ex_alpha, bra_beta,
                                  ket_beta, ex_beta, bra_occ_alpha,
                                  bra_occ_beta, val, ordm, trdm);
              }
            }  // Possible non-zero connection (Hamming distance)

          }  // Non-zero ket determinant
        }    // Loop over ket determinants

      }  // Non-zero bra determinant
    }    // Loop over bra determinants
  }

 public:
  template <typename... Args>
  SortedDoubleLoopHamiltonianGenerator(Args&&... args)
      : HamiltonianGenerator<WfnType>(std::forward<Args>(args)...) {}
};

}  // namespace macis
