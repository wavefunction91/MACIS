#pragma once
#include <asci/util/asci_contributions.hpp>

namespace asci {

template <typename WfnT>
void reorder_ci_on_coeff( std::vector<WfnT>& dets, 
  std::vector<double>& C_local, MPI_Comm /* comm: will need for dist*/ ) {

  size_t nlocal = C_local.size();
  size_t ndets  = dets.size();
  std::vector<uint64_t> idx( nlocal );
  std::iota( idx.begin(), idx.end(), 0 );
  std::sort( idx.begin(), idx.end(), [&](auto i, auto j) {
    return std::abs(C_local[i]) > std::abs(C_local[j]);
  });

  std::vector<double> reorder_C( nlocal );
  std::vector<WfnT> reorder_dets( ndets );
  assert( nlocal == ndets );
  for( auto i = 0ul; i < ndets; ++i ) {
    reorder_C[i]    = C_local[idx[i]];
    reorder_dets[i] = dets[idx[i]];
  }

  C_local = std::move(reorder_C);
  dets    = std::move(reorder_dets);

}

template <typename WfnT>
void sort_and_accumulate_asci_pairs( asci_contrib_container<WfnT>& asci_pairs ) {
  auto comparator = [](const auto& x, const auto& y) {
    return bitset_less(x.state, y.state);
  };

  // Sort by bitstring
  std::sort( asci_pairs.begin(), asci_pairs.end(), comparator );

  // Accumulate the ASCI scores into first instance of unique bitstrings
  auto cur_it = asci_pairs.begin();
  for( auto it = cur_it + 1; it != asci_pairs.end(); ++it ) {
    // If iterate is not the one being tracked, update the iterator
    if( it->state != cur_it->state ) { cur_it = it; }

    // Accumulate
    else {
      cur_it->rv += it->rv;
      it->rv = 0; // Zero out to expose potential bugs
    }
  }

  // Remote duplicate bitstrings
  auto uit = std::unique( asci_pairs.begin(), asci_pairs.end(),
    [](auto x, auto y){ return x.state == y.state; } );
  asci_pairs.erase(uit, asci_pairs.end()); // Erase dead space
}

template <typename WfnT>
void keep_only_largest_copy_asci_pairs( 
  asci_contrib_container<WfnT>& asci_pairs 
) {
  auto comparator = [](const auto& x, const auto& y) {
    return not (x.state == y.state or bitset_less(x.state, y.state));
  };

  // Sort by bitstring
  std::sort( asci_pairs.begin(), asci_pairs.end(), comparator );

  // Keep the largest ASCI score in the unique instance of each bit string
  auto cur_it = asci_pairs.begin();
  for( auto it = cur_it + 1; it != asci_pairs.end(); ++it ) {
    // If iterate is not the one being tracked, update the iterator
    if( it->state != cur_it->state ) { cur_it = it; }

    // Keep only max value
    else { cur_it->rv = std::max( cur_it->rv, it->rv ); }
  }

  // Remote duplicate bitstrings
  auto uit = std::unique( asci_pairs.begin(), asci_pairs.end(),
    [](auto x, auto y){ return x.state == y.state; } );
  asci_pairs.erase(uit, asci_pairs.end()); // Erase dead space

}

} // namespace asci