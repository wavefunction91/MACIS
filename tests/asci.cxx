#include "ut_common.hpp"
#include <asci/types.hpp>
#include <asci/bitset_operations.hpp>
#include <asci/sd_operations.hpp>
#include <iostream>

template <size_t NRadix, size_t NBits>
std::array<unsigned, NRadix> top_set_indices( std::bitset<NBits> word ) {
  std::array<unsigned, NRadix> top_set_indices;
  for(size_t i = 0; i < NRadix; ++i) {
    auto r = asci::fls(word);
    top_set_indices[i] = r;
    word.flip(r);
  }
  return top_set_indices;
}

template <size_t NRadix, size_t NSet, size_t NBits>
size_t top_set_ordinal( std::bitset<NBits> word ) {
  auto ind = top_set_indices<NRadix>(word);
  size_t ord = 0; size_t fact = 1;
  for(size_t i = 0; i < NRadix; ++i) {
    ord += ind[NRadix - i - 1] * fact;
    fact *= NSet;
  }
  return ord;
}

namespace asci {

template <size_t N, typename IndContainer>
void generate_pairs( const IndContainer& inds, std::vector<wfn_t<N>>& w ) {
  const size_t nind = inds.size();
  w.resize((nind * (nind-1))/2,0);
  for(int i = 0, ij = 0; i < nind; ++i      )
  for(int j = i+1;       j < nind; ++j, ++ij) {
    w[ij].flip(inds[i]).flip(inds[j]);
  }
}



/**
 *  @param[in]  det       Input root determinant
 *  @param[in]  T         Triplet constraint mask
 *  @param[in]  O         Overfill mask (full mask 0 -> norb)
 *  @param[in]  B         B mask (?)
 *  @param[out] t_doubles 
 */
template <size_t N>
void generate_triplet_doubles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  std::vector<wfn_t<N>>& t_doubles
) {

  if( (det & T) == 0 ) return;

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;

  // Occ/Vir pairs to generate excitations
  std::vector<wfn_t<N>> O,V; 

  // Generate Virtual Pairs
  if( (o & T).count() >= 2 ) {
    v = o & T;
    o ^= v;
  }

  const auto virt_ind = bits_to_indices(v);
  const auto o_and_t = o & T;
  switch( (o & T).count() ) {
    case 1:
      for( auto a : virt_ind ) {
        V.emplace_back(o_and_t).flip(a);
      }
      o ^= o_and_t;
      break;
    default:
      generate_pairs(virt_ind, V);
      break;
  }

  // Generate Occupied Pairs
  const auto o_and_not_b = o & ~B;
  if( o_and_not_b.count() > 2 ) return;

  switch(o_and_not_b.count()) {
    case 1 :
      for( auto i : bits_to_indices( o & B ) ) {
        O.emplace_back(o_and_not_b).flip(i);
      }
      break;
    default:
      if( o_and_not_b.count() == 2 ) o = o_and_not_b;
      generate_pairs( bits_to_indices(o), O );
      break;
  }

  t_doubles.clear();
  for(auto ij : O) {
    const auto temp = det ^ ij;
    for( auto ab : V ) {
      t_doubles.emplace_back(temp | ab);
    }
  }
}

/**
 *  @param[in]  det       Input root determinant
 *  @param[in]  T         Triplet constraint mask
 *  @param[in]  O         Overfill mask (full mask 0 -> norb)
 *  @param[in]  B         B mask (?)
 *  @param[out] t_singles 
 */
template <size_t N>
void generate_triplet_singles( wfn_t<N> det, 
  wfn_t<N> T, wfn_t<N> O_mask, wfn_t<N> B,
  std::vector<wfn_t<N>>& t_singles
) {

  if( (det & T).count() < 2 ) return;

  auto o = det ^ T;
  auto v = (~det) & O_mask & B;

  if( (o & T).count() >= 1 ) {
    v = o & T;
    o ^= v;
  }

  if( (o & ~B).count() >  1 ) return;
  if( (o & ~B).count() == 1 ) o &= ~B;

  const auto occ = bits_to_indices(o);
  const auto vir = bits_to_indices(v);
  t_singles.clear();
  t_singles.reserve(occ.size() * vir.size());
  for( auto i : occ ) {
    auto temp = det; temp.flip(i);
    for( auto a : vir ) t_singles.emplace_back(temp).flip(a);
  }
}

}

TEST_CASE("Triplets") {

  constexpr size_t num_bits = 64;
  size_t norb = 32;

  using wfn_less_comparator = asci::bitset_less_comparator<num_bits>;

  // Generate ficticious wfns
  std::vector<asci::wfn_t<num_bits>> wfn_a = {
    15//, 30, 15, 29
  }; 

  std::vector<asci::wfn_t<num_bits>> wfn_b = {
    15//, 15, 30, 15
  }; 

  const size_t ndet = wfn_a.size();

  // Combine the alpha/beta strings
  std::vector<asci::wfn_t<num_bits>> wfns(ndet);
  for(int i = 0; i < ndet; ++i) {
    wfns[i] = (wfn_a[i] << num_bits/2) | wfn_b[i];
  }

  // Total sort of combined bit-strings
  std::sort( wfns.begin(), wfns.end(), wfn_less_comparator{} );

  // Extract unique alphas
  std::vector<asci::wfn_t<num_bits>> wfn_a_uniq(wfns);
  {
    // Extract alpha strings
    std::transform( wfn_a_uniq.begin(), wfn_a_uniq.end(), wfn_a_uniq.begin(),
      [=](const auto& w){ return w >> (num_bits/2); } );

    // Determine unique alphas in place
    auto it = std::unique(wfn_a_uniq.begin(), wfn_a_uniq.end());

    // Remove excess
    wfn_a_uniq.erase( it , wfn_a_uniq.end() );
  }


  // Count beta dets per unique alpha
  const size_t nuniq_alpha = wfn_a_uniq.size();
  std::vector<size_t> nbetas(nuniq_alpha, 0);
  for( size_t i = 0, i_alpha = 0; i < ndet; ++i ) {
    const auto& w   = wfns[i];
    const auto& w_a = wfn_a_uniq[i_alpha];
    if( (w >> num_bits/2) != w_a ) i_alpha++;
    nbetas[i_alpha]++;
  }

  // Print beta counts
  for( size_t i = 0; i < nuniq_alpha; ++i ) 
    std::cout << wfn_a_uniq[i].to_ulong() << " " << nbetas[i] << std::endl;


  // Compute Histograms
  std::vector<size_t> hist(num_bits/2*num_bits/2*num_bits/2, 0);
  for( auto i = 0; i < nuniq_alpha; ++i) {

    // Constant dimensions
    const size_t nocc = wfn_a_uniq[i].count();
    const size_t nvir = norb - nocc;
    const size_t n_singles = nocc * nvir;
    const size_t n_doubles = 
      (n_singles * (n_singles - nocc - nvir + 1))/4;

    // Generate singles and doubles
    std::vector<asci::wfn_t<num_bits>> s_a, d_a;
    asci::generate_singles_doubles( norb, wfn_a_uniq[i], s_a, d_a );

    // Histogram contribution from root determinant
    {
      const auto label = top_set_ordinal<3, num_bits/2>(wfn_a_uniq[i]);
      hist[label] += n_singles + n_doubles;
    }

    // Histogram contribtutions from single excitations
    for(const auto& w : s_a) {
      const auto label = top_set_ordinal<3, num_bits/2>(w);
      hist[label] += n_singles + 1;
    }

    // Histogram contribtuions from double excitations
    for(const auto& w : d_a) {
      const auto label = top_set_ordinal<3, num_bits/2>(w);
      hist[label]++;
    }

  }

  //std::cout << std::accumulate(hist.begin(), hist.end(),0ul) << std::endl;

  // Print Histogram
  //for( auto i = 0; i < hist.size() ; ++ i ) {
  //  if(hist[i]) std::cout << i << " " << hist[i] << std::endl;
  //}

  std::vector<std::tuple<unsigned, unsigned, unsigned>> triplets; 
  for(int i = 0; i < 32; ++i)
  for(int j = 0; j < i;  ++j)
  for(int k = 0; k < j;  ++k) {
    triplets.emplace_back(i,j,k);
  }

  std::vector<size_t> new_hist(hist.size(), 0);
  for( auto [i,j,k] : triplets ) {
    const auto label = i*32*32 + j*32 + k;
    // Create masks
    asci::wfn_t<num_bits> T(0); 
    T.flip(i).flip(j).flip(k); 

    auto overfill = asci::full_mask<num_bits>(norb);
    asci::wfn_t<num_bits> B(1); B <<= k; B = B.to_ullong() - 1;

    std::vector<asci::wfn_t<num_bits>> t_doubles, t_singles;
    for( auto det : wfn_a_uniq ) {
      const size_t nocc = det.count();
      const size_t nvir = norb - nocc;
      const size_t n_singles = nocc * nvir;
      const size_t n_doubles = 
        (n_singles * (n_singles - nocc - nvir + 1))/4;

      asci::generate_triplet_doubles( det, T, overfill, B, t_doubles );
      asci::generate_triplet_singles( det, T, overfill, B, t_singles );
      new_hist[label] += t_doubles.size() + t_singles.size() * (n_singles + 1);
    }
  }

  for( auto det : wfn_a_uniq ) {
    const size_t nocc = det.count();
    const size_t nvir = norb - nocc;
    const size_t n_singles = nocc * nvir;
    const size_t n_doubles = 
      (n_singles * (n_singles - nocc - nvir + 1))/4;

    const auto label = top_set_ordinal<3, num_bits/2>(det);
    new_hist[label] += n_singles + n_doubles;
  }
  
  std::cout << std::boolalpha << (hist == new_hist) << std::endl;
}

