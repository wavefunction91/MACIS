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
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;
  std::vector<asci_contrib<wfn_t<N>>> asci_pairs;
  auto pairs_st = clock_type::now();
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
