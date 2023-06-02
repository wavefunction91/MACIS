/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/bitset_operations.hpp>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>

#include "ut_common.hpp"

template <size_t NRadix, size_t NBits>
std::array<unsigned, NRadix> top_set_indices(std::bitset<NBits> word) {
  std::array<unsigned, NRadix> top_set_indices;
  for(size_t i = 0; i < NRadix; ++i) {
    auto r = macis::fls(word);
    top_set_indices[i] = r;
    word.flip(r);
  }
  return top_set_indices;
}

template <size_t NRadix, size_t NBits>
size_t top_set_ordinal(std::bitset<NBits> word, size_t NSet) {
  auto ind = top_set_indices<NRadix>(word);
  size_t ord = 0;
  size_t fact = 1;
  for(size_t i = 0; i < NRadix; ++i) {
    ord += ind[NRadix - i - 1] * fact;
    fact *= NSet;
  }
  return ord;
}

TEST_CASE("Triplets") {
  constexpr size_t num_bits = 64;
  size_t norb = 32;

  using wfn_less_comparator = macis::bitset_less_comparator<num_bits>;

  // Generate ficticious wfns
  std::vector<macis::wfn_t<num_bits>> wfn_a = {
      15  //, 30, 15, 29
  };

  std::vector<macis::wfn_t<num_bits>> wfn_b = {
      15  //, 15, 30, 15
  };

  const size_t ndet = wfn_a.size();

  // Combine the alpha/beta strings
  std::vector<macis::wfn_t<num_bits>> wfns(ndet);
  for(int i = 0; i < ndet; ++i) {
    wfns[i] = (wfn_a[i] << num_bits / 2) | wfn_b[i];
  }

  // Total sort of combined bit-strings
  std::sort(wfns.begin(), wfns.end(), wfn_less_comparator{});

  // Extract unique alphas
  std::vector<macis::wfn_t<num_bits>> wfn_a_uniq(wfns);
  {
    // Extract alpha strings
    std::transform(wfn_a_uniq.begin(), wfn_a_uniq.end(), wfn_a_uniq.begin(),
                   [=](const auto& w) { return w >> (num_bits / 2); });

    // Determine unique alphas in place
    auto it = std::unique(wfn_a_uniq.begin(), wfn_a_uniq.end());

    // Remove excess
    wfn_a_uniq.erase(it, wfn_a_uniq.end());
  }

  // Count beta dets per unique alpha
  const size_t nuniq_alpha = wfn_a_uniq.size();
  std::vector<size_t> nbetas(nuniq_alpha, 0);
  for(size_t i = 0, i_alpha = 0; i < ndet; ++i) {
    const auto& w = wfns[i];
    const auto& w_a = wfn_a_uniq[i_alpha];
    if((w >> num_bits / 2) != w_a) i_alpha++;
    nbetas[i_alpha]++;
  }

  // Print beta counts
  // for( size_t i = 0; i < nuniq_alpha; ++i )
  //  std::cout << wfn_a_uniq[i].to_ulong() << " " << nbetas[i] << std::endl;

  // Compute Histograms
  std::vector<size_t> triplet_hist(norb * norb * norb, 0);
  std::vector<size_t> quad_hist(norb * norb * norb * norb, 0);
  for(auto i = 0; i < nuniq_alpha; ++i) {
    // Constant dimensions
    const size_t nocc = wfn_a_uniq[i].count();
    const size_t nvir = norb - nocc;
    const size_t n_singles = nocc * nvir;
    const size_t n_doubles = (n_singles * (n_singles - nocc - nvir + 1)) / 4;

    // Generate singles and doubles
    std::vector<macis::wfn_t<num_bits>> s_a, d_a;
    macis::generate_singles_doubles(norb, wfn_a_uniq[i], s_a, d_a);

    // Histogram contribution from root determinant
    {
      const auto label = top_set_ordinal<3>(wfn_a_uniq[i], norb);
      triplet_hist[label] += n_singles + n_doubles + 1;
    }
    {
      const auto label = top_set_ordinal<4>(wfn_a_uniq[i], norb);
      quad_hist[label] += n_singles + n_doubles + 1;
    }

    // Histogram contribtutions from single excitations
    for(const auto& w : s_a) {
      const auto label = top_set_ordinal<3>(w, norb);
      triplet_hist[label] += n_singles + 1;
    }
    for(const auto& w : s_a) {
      const auto label = top_set_ordinal<4>(w, norb);
      quad_hist[label] += n_singles + 1;
    }

    // Histogram contribtuions from double excitations
    for(const auto& w : d_a) {
      const auto label = top_set_ordinal<3>(w, norb);
      triplet_hist[label]++;
    }
    for(const auto& w : d_a) {
      const auto label = top_set_ordinal<4>(w, norb);
      quad_hist[label]++;
    }
  }

  REQUIRE(std::accumulate(triplet_hist.begin(), triplet_hist.end(), 0ul) ==
          std::accumulate(quad_hist.begin(), quad_hist.end(), 0ul));

  // Print Histogram
  // for( auto i = 0; i < triplet_hist.size() ; ++ i ) {
  //  if(triplet_hist[i]) std::cout << i << " " << triplet_hist[i] << std::endl;
  //}

  std::vector<std::tuple<unsigned, unsigned, unsigned>> triplets;
  for(int i = 0; i < norb; ++i)
    for(int j = 0; j < i; ++j)
      for(int k = 0; k < j; ++k) {
        triplets.emplace_back(i, j, k);
      }

  const auto overfill = macis::full_mask<num_bits>(norb);

  std::vector<size_t> new_triplet_hist(triplet_hist.size(), 0);
  for(auto [i, j, k] : triplets) {
    const auto label = i * 32 * 32 + j * 32 + k;
    auto [T, B, T_min] = macis::make_triplet<num_bits>(i, j, k);

    for(auto det : wfn_a_uniq) {
      const size_t nocc = det.count();
      const size_t nvir = norb - nocc;
      const size_t n_singles = nocc * nvir;
      const size_t n_doubles = (n_singles * (n_singles - nocc - nvir + 1)) / 4;

      new_triplet_hist[label] += macis::constraint_histogram(
          det, n_singles, n_doubles, T, overfill, B);
    }
  }

  REQUIRE(triplet_hist == new_triplet_hist);

  std::vector<size_t> new_quad_hist(quad_hist.size(), 0);
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>> quads;
  for(int i = 0; i < norb; ++i)
    for(int j = 0; j < i; ++j)
      for(int k = 0; k < j; ++k)
        for(int l = 0; l < k; ++l) {
          quads.emplace_back(i, j, k, l);
        }

  for(auto [i, j, k, l] : quads) {
    const size_t label =
        i * 32ul * 32ul * 32ul + j * 32ul * 32ul + k * 32ul + l;
    auto [Q, B, Q_min] = macis::make_quad<num_bits>(i, j, k, l);

    for(auto det : wfn_a_uniq) {
      const size_t nocc = det.count();
      const size_t nvir = norb - nocc;
      const size_t n_singles = nocc * nvir;
      const size_t n_doubles = (n_singles * (n_singles - nocc - nvir + 1)) / 4;

      new_quad_hist[label] += macis::constraint_histogram(
          det, n_singles, n_doubles, Q, overfill, B);
    }
  }

  REQUIRE(quad_hist == new_quad_hist);
}
