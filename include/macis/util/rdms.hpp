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

namespace macis {

template <typename T, size_t N>
inline void rdm_contributions_4(wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex, T val,
                                rank4_span<T> trdm) {
  auto [o1, v1, o2, v2, sign] = doubles_sign_indices(bra, ket, ex);

  val *= sign * 0.5;
  trdm(v1, o1, v2, o2) += val;
  trdm(v2, o1, v1, o2) -= val;
  trdm(v1, o2, v2, o1) -= val;
  trdm(v2, o2, v1, o1) += val;
}

template <typename T, size_t N>
inline void rdm_contributions_22(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                                 wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                                 wfn_t<N> ket_beta, wfn_t<N> ex_beta, T val,
                                 rank4_span<T> trdm_ud, rank4_span<T> trdm_du) {
  auto [o1, v1, sign_a] =
      single_excitation_sign_indices(bra_alpha, ket_alpha, ex_alpha);
  auto [o2, v2, sign_b] =
      single_excitation_sign_indices(bra_beta, ket_beta, ex_beta);
  auto sign = sign_a * sign_b;

  val *= sign * 0.5;
  trdm_ud(v1, o1, v2, o2) += val;
  trdm_du(v2, o2, v1, o1) += val;
}

template <typename T, size_t N, typename IndexType>
inline void rdm_contributions_2(wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex,
                                const IndexType& bra_occ_alpha,
                                const IndexType& bra_occ_beta, T val,
                                matrix_span<T> ordm, rank4_span<T> trdm_ss,
                                rank4_span<T> trdm_os, rank4_span<T> trdm_so) {
  auto [o1, v1, sign] = single_excitation_sign_indices(bra, ket, ex);

  ordm(v1, o1) += sign * val;

  if(trdm_ss.data_handle()) {
    val *= sign * 0.5;
    for(auto p : bra_occ_alpha) {
      trdm_ss(v1, o1, p, p) += val;
      trdm_ss(p, p, v1, o1) += val;
      trdm_ss(v1, p, p, o1) -= val;
      trdm_ss(p, o1, v1, p) -= val;
    }

    for(auto p : bra_occ_beta) {
      trdm_so(v1, o1, p, p) += val;
      trdm_os(p, p, v1, o1) += val;
    }
  }
}

template <typename T, typename IndexType>
inline void rdm_contributions_diag(const IndexType& occ_alpha,
                                   const IndexType& occ_beta, T val,
                                   matrix_span<T> ordm_u, matrix_span<T> ordm_d,
                                   rank4_span<T> trdm_uu, rank4_span<T> trdm_ud,
                                   rank4_span<T> trdm_du,
                                   rank4_span<T> trdm_dd) {
  // One-electron piece
  for(auto p : occ_alpha) ordm_u(p, p) += val;
  for(auto p : occ_beta) ordm_d(p, p) += val;

  if(trdm_uu.data_handle()) {
    val *= 0.5;
    // Same-spin two-body term
    for(auto q : occ_alpha)
      for(auto p : occ_alpha) {
        trdm_uu(p, p, q, q) += val;
        trdm_uu(p, q, p, q) -= val;
      }
    for(auto q : occ_beta)
      for(auto p : occ_beta) {
        trdm_dd(p, p, q, q) += val;
        trdm_dd(p, q, p, q) -= val;
      }

    // Opposite-spin two-body term
    for(auto q : occ_beta)
      for(auto p : occ_alpha) {
        trdm_ud(p, p, q, q) += val;
        trdm_du(q, q, p, p) += val;
      }
  }
}

template <typename T, size_t N, typename IndexType>
inline void rdm_contributions(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                              wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                              wfn_t<N> ket_beta, wfn_t<N> ex_beta,
                              const IndexType& bra_occ_alpha,
                              const IndexType& bra_occ_beta, T val,
                              matrix_span<T> ordm_u, matrix_span<T> ordm_d,
                              rank4_span<T> trdm_uu, rank4_span<T> trdm_ud,
                              rank4_span<T> trdm_du, rank4_span<T> trdm_dd) {
  const uint32_t ex_alpha_count = ex_alpha.count();
  const uint32_t ex_beta_count = ex_beta.count();

  if((ex_alpha_count + ex_beta_count) > 4) return;

  const auto trdm_ptr = trdm_uu.data_handle();
  if(ex_alpha_count == 4 and trdm_ptr)
    rdm_contributions_4(bra_alpha, ket_alpha, ex_alpha, val, trdm_uu);

  else if(ex_beta_count == 4 and trdm_ptr)
    rdm_contributions_4(bra_beta, ket_beta, ex_beta, val, trdm_dd);

  else if(ex_alpha_count == 2 and ex_beta_count == 2 and trdm_ptr)
    rdm_contributions_22(bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta,
                         ex_beta, val, trdm_ud, trdm_du);

  else if(ex_alpha_count == 2)
    rdm_contributions_2(bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
                        bra_occ_beta, val, ordm_u, trdm_uu, trdm_ud, trdm_du);

  else if(ex_beta_count == 2)
    rdm_contributions_2(bra_beta, ket_beta, ex_beta, bra_occ_beta,
                        bra_occ_alpha, val, ordm_d, trdm_dd, trdm_du, trdm_ud);

  else
    rdm_contributions_diag(bra_occ_alpha, bra_occ_beta, val, ordm_u, ordm_d,
                           trdm_uu, trdm_ud, trdm_du, trdm_dd);
}

}  // namespace macis
