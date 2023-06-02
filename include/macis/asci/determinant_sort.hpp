/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/asci/determinant_contributions.hpp>
#if __has_include(<boost/sort/pdqsort/pdqsort.hpp>)
#define MACIS_USE_BOOST_SORT
#include <boost/sort/pdqsort/pdqsort.hpp>
#endif

namespace macis {

template <typename WfnT>
void reorder_ci_on_coeff(std::vector<WfnT>& dets, std::vector<double>& C) {
  size_t nlocal = C.size();
  size_t ndets = dets.size();
  std::vector<uint64_t> idx(nlocal);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](auto i, auto j) { return std::abs(C[i]) > std::abs(C[j]); });

  std::vector<double> reorder_C(nlocal);
  std::vector<WfnT> reorder_dets(ndets);
  assert(nlocal == ndets);
  for(auto i = 0ul; i < ndets; ++i) {
    reorder_C[i] = C[idx[i]];
    reorder_dets[i] = dets[idx[i]];
  }

  C = std::move(reorder_C);
  dets = std::move(reorder_dets);
}

template <typename PairIterator>
PairIterator sort_and_accumulate_asci_pairs(PairIterator pairs_begin,
                                            PairIterator pairs_end) {
  const size_t npairs = std::distance(pairs_begin, pairs_end);

  if(!npairs) return pairs_end;

  auto comparator = [](const auto& x, const auto& y) {
    return bitset_less(x.state, y.state);
  };

// Sort by bitstring
#ifdef MACIS_USE_BOOST_SORT
  boost::sort::pdqsort_branchless
#else
  std::sort
#endif
      (pairs_begin, pairs_end, comparator);

  // Accumulate the ASCI scores into first instance of unique bitstrings
  auto cur_it = pairs_begin;
  for(auto it = cur_it + 1; it != pairs_end; ++it) {
    // If iterate is not the one being tracked, update the iterator
    if(it->state != cur_it->state) {
      cur_it = it;
    }

    // Accumulate
    else {
      cur_it->rv += it->rv;
      it->rv = 0;  // Zero out to expose potential bugs
    }
  }

  // Remote duplicate bitstrings
  return std::unique(pairs_begin, pairs_end,
                     [](auto x, auto y) { return x.state == y.state; });
}

template <typename WfnT>
void sort_and_accumulate_asci_pairs(asci_contrib_container<WfnT>& asci_pairs) {
  auto uit =
      sort_and_accumulate_asci_pairs(asci_pairs.begin(), asci_pairs.end());
  asci_pairs.erase(uit, asci_pairs.end());  // Erase dead space
}

template <typename WfnT>
void keep_only_largest_copy_asci_pairs(
    asci_contrib_container<WfnT>& asci_pairs) {
  if(!asci_pairs.size()) return;
  auto comparator = [](const auto& x, const auto& y) {
    return bitset_less(x.state, y.state);
  };

// Sort by bitstring
#ifdef MACIS_USE_BOOST_SORT
  boost::sort::pdqsort_branchless
#else
  std::sort
#endif
      (asci_pairs.begin(), asci_pairs.end(), comparator);

  // Keep the largest ASCI score in the unique instance of each bit string
  auto cur_it = asci_pairs.begin();
  for(auto it = cur_it + 1; it != asci_pairs.end(); ++it) {
    // If iterate is not the one being tracked, update the iterator
    if(it->state != cur_it->state) {
      cur_it = it;
    }

    // Keep only max value
    else {
      cur_it->rv = std::max(cur_it->rv, it->rv);
    }
  }

  // Remote duplicate bitstrings
  auto uit = std::unique(asci_pairs.begin(), asci_pairs.end(),
                         [](auto x, auto y) { return x.state == y.state; });
  asci_pairs.erase(uit, asci_pairs.end());  // Erase dead space
}

}  // namespace macis
