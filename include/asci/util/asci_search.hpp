#pragma once
#include <asci/types.hpp>
#include <asci/sd_operations.hpp>
#include <asci/util/asci_contributions.hpp>
#include <asci/util/asci_sort.hpp>

#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace asci {

struct ASCISettings {
  size_t ntdets_max        = 1e5;
  size_t ncdets_max        = 100;
  double h_el_tol          = 1e-8;
  double rv_prune_tol      = 1e-8;
  size_t pair_size_max     = 2e9;
  bool   just_singles      = false;
  size_t grow_factor       = 8;
  size_t max_refine_iter   = 6;
  double refine_energy_tol = 1e-6;
};


template <size_t N>
std::vector< wfn_t<N> > asci_search( 
  ASCISettings               asci_settings, 
  size_t                     ndets_max,
  wavefunction_iterator_t<N> cdets_begin, 
  wavefunction_iterator_t<N> cdets_end,
  const double               E_ASCI, 
  const std::vector<double>& C_local,
  size_t                     norb,
  const double*              T_pq,
  const double*              G_red,
  const double*              V_red,
  const double*              G_pqrs,
  const double*              V_pqrs,
  HamiltonianGenerator<N>&   ham_gen,
  MPI_Comm                   comm
) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  auto logger = spdlog::get("asci_search");
  if(!logger) logger = spdlog::stdout_color_mt("asci_search");
  

  // Print to logger
  const size_t ncdets = std::distance(cdets_begin, cdets_end);
  logger->info("[ASCI Search Settings]:");
  logger->info(
    "  NCDETS = {:6}, NDETS_MAX = {:9}, H_EL_TOL = {:4e}, RV_TOL = {:4e}", 
    ncdets, ndets_max, asci_settings.h_el_tol, 
    asci_settings.rv_prune_tol);
  logger->info("  MAX_RV_SIZE = {}, JUST_SINGLES = {}", 
    asci_settings.pair_size_max, asci_settings.just_singles);

  
  // Expand Search Space
  auto pairs_st = clock_type::now();
  std::vector<asci_contrib<wfn_t<N>>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;
  for( size_t i = 0; i < ncdets; ++i ) {

    // Alias state data
    auto state       = *(cdets_begin + i);
    auto state_alpha = truncate_bitset<N/2>(state);
    auto state_beta  = truncate_bitset<N/2>(state >> (N/2));
    auto coeff       = C_local[i]; // XXX: not valid for parallel search

    // Get occupied and virtual indices
    bitset_to_occ_vir( norb, state_alpha, occ_alpha, vir_alpha ); 
    bitset_to_occ_vir( norb, state_beta,  occ_beta,  vir_beta  ); 

    // Compute base diagonal matrix element
    double h_diag = ham_gen.matrix_element(state, state);

    const double h_el_tol = asci_settings.h_el_tol;

    // Singles - AA
    append_singles_asci_contributions<(N/2),0>( coeff, state, state_alpha,
      occ_alpha, vir_alpha, occ_beta, T_pq, norb, G_red, norb, V_red, norb, 
      h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );

    // Singles - BB
    append_singles_asci_contributions<(N/2),(N/2)>( coeff, state, state_beta, 
      occ_beta, vir_beta, occ_alpha, T_pq, norb, G_red, norb, V_red, norb, 
      h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );

    if(not asci_settings.just_singles) {
      // Doubles - AAAA
      append_ss_doubles_asci_contributions<N/2,0>( coeff, state, state_alpha, 
        occ_alpha, vir_alpha, occ_beta, G_pqrs, norb, h_el_tol, h_diag, E_ASCI, 
        ham_gen, asci_pairs);

      // Doubles - BBBB
      append_ss_doubles_asci_contributions<N/2,N/2>( coeff, state, state_beta, 
        occ_beta, vir_beta, occ_alpha, G_pqrs, norb, h_el_tol, h_diag, E_ASCI,
        ham_gen, asci_pairs);

      // Doubles - AABB
      append_os_doubles_asci_contributions( coeff, state, state_alpha, state_beta, 
        occ_alpha, occ_beta, vir_alpha, vir_beta, V_pqrs, norb, h_el_tol, h_diag,
        E_ASCI, ham_gen, asci_pairs );
    }

    // Prune Down Contributions
    if( asci_pairs.size() > asci_settings.pair_size_max ) {

      std::cout << "PRUNING" << std::endl;
      // Remove small contributions
      auto it = std::partition( asci_pairs.begin(), asci_pairs.end(),
        [=](const auto& x){ 
          return std::abs(x.rv) > asci_settings.rv_prune_tol;
        });
      asci_pairs.erase(it, asci_pairs.end());
      logger->info("  * Pruning at DET = {} NSZ = {}", i, asci_pairs.size() );

      // Extra Pruning if not sufficient
      if( asci_pairs.size() > asci_settings.pair_size_max ) {
        logger->info("    * Removing Duplicates");
        sort_and_accumulate_asci_pairs( asci_pairs );
        logger->info("    * NSZ = {}", asci_pairs.size());
      }

    } // Pruning 
  } // Loop over search determinants

#if 0

  // Get unique alpha strings
  std::vector<wfn_t<N>> uniq_alpha_wfn(cdets_begin, cdets_end);
  std::transform( uniq_alpha_wfn.begin(), uniq_alpha_wfn.end(),
    uniq_alpha_wfn.begin(),
    [=](const auto& w){ return w & full_mask<N/2,N>(); });
  std::sort(uniq_alpha_wfn.begin(), uniq_alpha_wfn.end(),
    bitset_less_comparator<N>{});
  {
    auto it = std::unique(uniq_alpha_wfn.begin(), uniq_alpha_wfn.end());
    uniq_alpha_wfn.erase(it, uniq_alpha_wfn.end());
  }
  const size_t nuniq_alpha = uniq_alpha_wfn.size();


  // For each unique alpha, create a list of beta string and store coeff
  struct beta_coeff_data {
    wfn_t<N> beta_string;
    double coeff;
  };
  struct unique_alpha_data {
    std::vector<beta_coeff_data> bcd;
  };
  std::vector<unique_alpha_data> uad(nuniq_alpha);
  for(auto i = 0; i < nuniq_alpha; ++i) {
    const auto wfn_a = uniq_alpha_wfn[i];
    for(auto j = 0; j < ncdets; ++j) { 
       const auto w = *(cdets_begin + j);
      if( (w & full_mask<N/2,N>()) == wfn_a ) {
        uad[i].bcd.push_back( {w >> (N/2), C_local[j]} );
      }
    }
  }

  size_t nacc = 0;
  for( auto &x : uad ) nacc += x.bcd.size();

  std::cout << "NTOT = " << ncdets << " NU = " << nuniq_alpha << " NACC = " << nacc << std::endl;

  // Loop over triplets
  std::vector<asci_contrib<wfn_t<N>>> new_asci_pairs;
  for(int t_i = 0; t_i < norb; ++t_i)
  for(int t_j = 0; t_j < t_i;  ++t_j)
  for(int t_k = 0; t_k < t_j;  ++t_k) {

    // Create masks
    wfn_t<N> T(0); T.flip(t_i).flip(t_j).flip(t_k);
    auto overfill = full_mask<N>(norb);
    wfn_t<N> B(1); B <<= t_k; B = B.to_ullong() - 1;

    const double h_el_tol = asci_settings.h_el_tol;
    std::cout << "T " << t_i << " " << t_j << " " << t_k << std::endl;
    // Loop over unique alpha strings
    for( size_t i_alpha = 0; i_alpha < nuniq_alpha; ++i_alpha ) {

      //std::cout << "IALPHA = " << i_alpha << std::endl;
      // Generate valid alpha excitation strings subject to T
      const auto& det = uniq_alpha_wfn[i_alpha];
      std::vector<wfn_t<N>> t_doubles, t_singles;
      asci::generate_triplet_doubles( det, T, overfill, B, t_doubles );
      asci::generate_triplet_singles( det, T, overfill, B, t_singles );

      // AA excitations
      for( auto s_aa : t_singles )
      for( auto [beta, coeff] : uad[i_alpha].bcd ) {
        auto state     = det  | (beta << N/2);
        auto new_state = s_aa | (beta << N/2);
        auto mat_el = ham_gen.matrix_element(state, new_state);
        if( std::abs(mat_el) < h_el_tol ) continue;

        auto h_diag = ham_gen.matrix_element(new_state, new_state);
        mat_el /= E_ASCI - h_diag;
        new_asci_pairs.push_back( {new_state, coeff * mat_el} );
      }
      //std::cout << "  AA Done" << std::endl;

      // AAAA excitations
      for( auto s_aa : t_doubles )
      for( auto [beta, coeff] : uad[i_alpha].bcd ) {
        auto state     = det  | (beta << N/2);
        auto new_state = s_aa | (beta << N/2);
        auto mat_el = ham_gen.matrix_element(state, new_state);
        if( std::abs(mat_el) < h_el_tol ) continue;

        auto h_diag = ham_gen.matrix_element(new_state, new_state);
        mat_el /= E_ASCI - h_diag;
        new_asci_pairs.push_back( {new_state, coeff * mat_el} );
      }
      //std::cout << "  AAAA Done" << std::endl;

      // AABB excitations
      for( auto s_aa : t_singles )
      for( auto [beta, coeff] : uad[i_alpha].bcd ) {
        auto state = det  | (beta << N/2);
        std::vector<wfn_t<N>> beta_singles;
        generate_singles(norb, beta, beta_singles);
        for( auto s_bb : beta_singles) {
          auto new_state = s_aa | (s_bb << N/2);
          auto mat_el = ham_gen.matrix_element(state, new_state);
          if( std::abs(mat_el) < h_el_tol ) continue;

          auto h_diag = ham_gen.matrix_element(new_state, new_state);
          mat_el /= E_ASCI - h_diag;
          new_asci_pairs.push_back( {new_state, coeff * mat_el} );
        }
      }
      //std::cout << "  AABB Done" << std::endl;

      if( (det & T).count() == 3 and ((det ^ T) >> t_k).count() == 0 ) {

        for( auto [beta, coeff] : uad[i_alpha].bcd ) {
          auto state = det  | (beta << N/2);
          std::vector<wfn_t<N>> beta_singles, beta_doubles;
          generate_singles_doubles(norb, beta, beta_singles, beta_doubles);

          // BB excitations
          for( auto s_bb : beta_singles ) {
            auto new_state = det | (s_bb << N/2);
            auto mat_el = ham_gen.matrix_element(state, new_state);
            if( std::abs(mat_el) < h_el_tol ) continue;

            auto h_diag = ham_gen.matrix_element(new_state, new_state);
            mat_el /= E_ASCI - h_diag;
            new_asci_pairs.push_back( {new_state, coeff * mat_el} );
          }

          // BBBB excitations
          for( auto d_bb : beta_doubles ) {
            auto new_state = det | (d_bb << N/2);
            auto mat_el = ham_gen.matrix_element(state, new_state);
            if( std::abs(mat_el) < h_el_tol ) continue;

            auto h_diag = ham_gen.matrix_element(new_state, new_state);
            mat_el /= E_ASCI - h_diag;
            new_asci_pairs.push_back( {new_state, coeff * mat_el} );
          }
        }
      }
    }

  }

  sort_and_accumulate_asci_pairs(new_asci_pairs);
  sort_and_accumulate_asci_pairs(asci_pairs);

  std::cout << std::scientific;
  if( asci_pairs.size() != new_asci_pairs.size() )
    throw std::runtime_error("Different Sizes");
  for( int i = 0; i < asci_pairs.size(); ++i ) {
    auto [ref_d, ref_c] = asci_pairs[i];
    auto [new_d, new_c] = new_asci_pairs[i];
    if( ref_d != new_d ) throw std::runtime_error("Different Det");
    if( std::abs(ref_c - new_c) > 1e-12 ) {
      std::cout << ref_c - new_c << std::endl;
      throw std::runtime_error("Different Contrib");
    }
  }

  //throw std::runtime_error("DIE DIE DIE");

#endif
  auto pairs_en = clock_type::now();

  logger->info("  * ASCI Kept {} Pairs", asci_pairs.size());

#if 0
  std::cout << "ASCI PAIRS" << std::endl;
  for(auto [s,rv] : asci_pairs) {
    std::cout << to_canonical_string(s) << ", " << std::scientific << rv << std::endl;
  }
#endif

  // Accumulate unique score contributions
  auto bit_sort_st = clock_type::now();
  sort_and_accumulate_asci_pairs( asci_pairs );
  auto bit_sort_en = clock_type::now();

  logger->info("  * ASCI will search over {} unique determinants",
    asci_pairs.size() );
  logger->info("  * PAIR_DUR = {:.2e} s, SORT_ACC_DUR = {:.2e} s",
    duration_type(pairs_en - pairs_st).count(),
    duration_type(bit_sort_en - bit_sort_st).count()
  );
 
  // Finalize scores
  std::transform(asci_pairs.begin(),asci_pairs.end(),asci_pairs.begin(),
    [](auto x) {x.rv =  -std::abs(x.rv); return x;});

  // Insert all dets with their coefficients as seeds
  for( size_t i = 0; i < ncdets; ++i ) {
    auto state = *(cdets_begin + i);
    asci_pairs.push_back({state, std::abs(C_local[i])});
  }

  // Check duplicates (which correspond to the initial truncation),
  // and keep only the duplicate with positive coefficient. 
  keep_only_largest_copy_asci_pairs(asci_pairs);

  // Sort pairs by ASCI score
  auto asci_sort_st = clock_type::now();
  std::nth_element( asci_pairs.begin(), 
    asci_pairs.begin() + ndets_max,
    asci_pairs.end(), 
    [](auto x, auto y){ 
      return std::abs(x.rv) > std::abs(y.rv);
    });
  auto asci_sort_en = clock_type::now();
  logger->info("  * ASCI_SORT_DUR = {:.2e} s", 
    duration_type(asci_sort_en - asci_sort_st).count()
  );

  // Shrink to max search space
  if( asci_pairs.size() > ndets_max )
    asci_pairs.erase( asci_pairs.begin() + ndets_max, 
      asci_pairs.end() );
  asci_pairs.shrink_to_fit();

  // Extract new search determinants
  std::vector<std::bitset<N>> new_dets( asci_pairs.size() );
  std::transform( asci_pairs.begin(), asci_pairs.end(), new_dets.begin(),
    [](auto x){ return x.state; } );

#if 0
  std::sort(new_dets.begin(), new_dets.end(),
  [](auto x, auto y){ return bitset_less(x,y); });
  std::cout << "NEW DETS " << new_dets.size() << std::endl;
  for( auto s : new_dets ) {
    std::cout << to_canonical_string(s) << std::endl;
  }
#endif

  return new_dets;
}

} // namespace asci
