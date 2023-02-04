#pragma once
#include <asci/types.hpp>
#include <asci/sd_operations.hpp>
#include <asci/util/asci_contributions.hpp>
#include <asci/util/asci_sort.hpp>
#include <asci/util/memory.hpp>

#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace asci {

struct ASCISettings {
  size_t ntdets_max        = 1e5;
  size_t ncdets_max        = 100;
  double h_el_tol          = 1e-8;
  double rv_prune_tol      = 1e-8;
  size_t pair_size_max     = 5e8;
  bool   just_singles      = false;
  size_t grow_factor       = 8;
  size_t max_refine_iter   = 6;
  double refine_energy_tol = 1e-6;
};

template <size_t N>
asci_contrib_container<wfn_t<N>> asci_contributions_standard(
  ASCISettings               asci_settings, 
  wavefunction_iterator_t<N> cdets_begin, 
  wavefunction_iterator_t<N> cdets_end,
  const double               E_ASCI, 
  const std::vector<double>& C,
  size_t                     norb,
  const double*              T_pq,
  const double*              G_red,
  const double*              V_red,
  const double*              G_pqrs,
  const double*              V_pqrs,
  HamiltonianGenerator<N>&   ham_gen
) {

  auto logger = spdlog::get("asci_search");
  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  asci_contrib_container<wfn_t<N>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;
  asci_pairs.reserve(asci_settings.pair_size_max);
  for( size_t i = 0; i < ncdets; ++i ) {

    // Alias state data
    auto state       = *(cdets_begin + i);
    auto state_alpha = truncate_bitset<N/2>(state);
    auto state_beta  = truncate_bitset<N/2>(state >> (N/2));
    auto coeff       = C[i]; 

    // Get occupied and virtual indices
    bitset_to_occ_vir( norb, state_alpha, occ_alpha, vir_alpha ); 
    bitset_to_occ_vir( norb, state_beta,  occ_beta,  vir_beta  ); 

    // Precompute orbital energies
    auto eps_alpha = ham_gen.single_orbital_ens( norb, occ_alpha, occ_beta );
    auto eps_beta  = ham_gen.single_orbital_ens( norb, occ_beta, occ_alpha );

    // Compute base diagonal matrix element
    double h_diag = ham_gen.matrix_element(state, state);

    const double h_el_tol = asci_settings.h_el_tol;

    // Singles - AA
    append_singles_asci_contributions<(N/2),0>( coeff, state, state_alpha,
      occ_alpha, vir_alpha, occ_beta, eps_alpha.data(), T_pq, norb, G_red, 
      norb, V_red, norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );

    // Singles - BB
    append_singles_asci_contributions<(N/2),(N/2)>( coeff, state, state_beta, 
      occ_beta, vir_beta, occ_alpha, eps_beta.data(), T_pq, norb, G_red, 
      norb, V_red, norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );

    if(not asci_settings.just_singles) {
      // Doubles - AAAA
      append_ss_doubles_asci_contributions<N/2,0>( coeff, state, state_alpha, 
        occ_alpha, vir_alpha, occ_beta, eps_alpha.data(), G_pqrs, norb, 
        h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

      // Doubles - BBBB
      append_ss_doubles_asci_contributions<N/2,N/2>( coeff, state, 
        state_beta, occ_beta, vir_beta, occ_alpha, eps_beta.data(),
        G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

      // Doubles - AABB
      append_os_doubles_asci_contributions( coeff, state, state_alpha, 
        state_beta, occ_alpha, occ_beta, vir_alpha, vir_beta, 
        eps_alpha.data(), eps_beta.data(), V_pqrs, norb, 
        h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );
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

  return asci_pairs;
}

template <size_t N>
asci_contrib_container<wfn_t<N>> asci_contributions_triplet(
  ASCISettings               asci_settings, 
  wavefunction_iterator_t<N> cdets_begin, 
  wavefunction_iterator_t<N> cdets_end,
  const double               E_ASCI, 
  const std::vector<double>& C,
  size_t                     norb,
  const double*              T_pq,
  const double*              G_red,
  const double*              V_red,
  const double*              G_pqrs,
  const double*              V_pqrs,
  HamiltonianGenerator<N>&   ham_gen
) {

  auto logger = spdlog::get("asci_search");
  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  asci_contrib_container<wfn_t<N>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;

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

  // For each unique alpha, create a list of beta string and store metadata
  struct beta_coeff_data {
    wfn_t<N> beta_string;
    std::vector<uint32_t> occ_beta;
    std::vector<uint32_t> vir_beta;
    std::vector<double>   orb_ens_alpha;
    std::vector<double>   orb_ens_beta;
    double coeff;
    double h_diag;

    beta_coeff_data( double c, size_t norb, const std::vector<uint32_t>& occ_alpha, 
      wfn_t<N> w, const HamiltonianGenerator<N>& ham_gen ) {

      coeff = c;

      // Compute Beta string 
      const auto beta_shift = w >> N/2;
      beta_string = beta_shift << N/2; // Reduce the number of times things shift in inner loop
    
      // Compute diagonal matrix element
      h_diag = ham_gen.matrix_element(w,w);

      // Compute occ/vir for beta string
      bitset_to_occ_vir(norb, beta_shift, occ_beta, vir_beta);

      // Precompute orbital energies
      orb_ens_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
      orb_ens_beta  = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);
      
    }

  };

  struct unique_alpha_data {
    std::vector<beta_coeff_data> bcd;
  };

  std::vector<unique_alpha_data> uad(nuniq_alpha);
  for(auto i = 0; i < nuniq_alpha; ++i) {
    const auto wfn_a = uniq_alpha_wfn[i];
    std::vector<uint32_t> occ_alpha, vir_alpha;
    bitset_to_occ_vir( norb, wfn_a, occ_alpha, vir_alpha );
    for(auto j = 0; j < ncdets; ++j) { 
       const auto w = *(cdets_begin + j);
      if( (w & full_mask<N/2,N>()) == wfn_a ) {
        uad[i].bcd.emplace_back( C[j], norb, occ_alpha, w, ham_gen );
      }
    }
  }

  asci_pairs.reserve(asci_settings.pair_size_max);
  // Loop over triplets
  for(int t_i = 0; t_i < norb; ++t_i)
  for(int t_j = 0; t_j < t_i;  ++t_j)
  for(int t_k = 0; t_k < t_j;  ++t_k) {

    auto [T,O,B] = 
      make_triplet_masks<N>(norb,t_i,t_j,t_k);

    const double h_el_tol = asci_settings.h_el_tol;

    // Loop over unique alpha strings
    for( size_t i_alpha = 0; i_alpha < nuniq_alpha; ++i_alpha ) {

      const auto& det = uniq_alpha_wfn[i_alpha];
      const auto occ_alpha = bits_to_indices(det);

      // AA excitations
      for( const auto& bcd : uad[i_alpha].bcd ) {
        const auto& beta     = bcd.beta_string;
        const auto& coeff    = bcd.coeff;
        const auto& h_diag   = bcd.h_diag;
        const auto& occ_beta = bcd.occ_beta;
        const auto& orb_ens_alpha  = bcd.orb_ens_alpha;
        generate_triplet_singles_contributions_ss(
          coeff, det, T, O, B, beta, occ_alpha, occ_beta, 
          orb_ens_alpha.data(), T_pq, norb, G_red, norb, V_red, norb,
          h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );
      }

      // AAAA excitations
      for( const auto& bcd : uad[i_alpha].bcd ) {
        const auto& beta     = bcd.beta_string;
        const auto& coeff    = bcd.coeff;
        const auto& h_diag   = bcd.h_diag;
        const auto& occ_beta = bcd.occ_beta;
        const auto& orb_ens_alpha  = bcd.orb_ens_alpha;
        generate_triplet_doubles_contributions_ss(
          coeff, det, T, O, B, beta, occ_alpha, occ_beta, 
          orb_ens_alpha.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, 
          ham_gen, asci_pairs );
      }

      // AABB excitations
      for( const auto& bcd : uad[i_alpha].bcd ) {
        const auto& beta     = bcd.beta_string;
        const auto& coeff    = bcd.coeff;
        const auto& h_diag   = bcd.h_diag;
        const auto& occ_beta = bcd.occ_beta;
        const auto& vir_beta = bcd.vir_beta;
        const auto& orb_ens_alpha  = bcd.orb_ens_alpha;
        const auto& orb_ens_beta  = bcd.orb_ens_beta;
        generate_triplet_doubles_contributions_os(
          coeff, det, T, O, B, beta, occ_alpha, occ_beta,
          vir_beta, orb_ens_alpha.data(), orb_ens_beta.data(),
          V_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );
      }

      if( (det & T).count() == 3 and ((det ^ T) >> t_k).count() == 0 ) {
        for( const auto& bcd : uad[i_alpha].bcd ) {

          const auto& beta     = bcd.beta_string;
          const auto& coeff    = bcd.coeff;
          const auto& h_diag   = bcd.h_diag;
          const auto& occ_beta = bcd.occ_beta;
          const auto& vir_beta = bcd.vir_beta;
          const auto& eps_beta  = bcd.orb_ens_beta;

          const auto state = det | beta;
          const auto state_beta = truncate_bitset<N/2>(beta >> N/2);
          // BB Excitations
          append_singles_asci_contributions<(N/2),(N/2)>( coeff, state,
            state_beta, occ_beta, vir_beta, occ_alpha, eps_beta.data(), 
            T_pq, norb, G_red, norb, V_red, norb, h_el_tol, h_diag, E_ASCI, 
            ham_gen, asci_pairs ); 

          // BBBB Excitations
          append_ss_doubles_asci_contributions<N/2,N/2>( coeff, state,
            state_beta, occ_beta, vir_beta, occ_alpha, eps_beta.data(),
            G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs );

        } // Beta Loop
      } // Triplet Check

      // Prune Down Contributions
      if( asci_pairs.size() > asci_settings.pair_size_max ) {

        // Remove small contributions
        auto it = std::partition( asci_pairs.begin(), asci_pairs.end(),
          [=](const auto& x){ 
            return std::abs(x.rv) > asci_settings.rv_prune_tol;
          });
        asci_pairs.erase(it, asci_pairs.end());
        logger->info("  * Pruning at TRIPLET = {} {} {}, NSZ = {}", 
          t_i, t_j, t_k,  asci_pairs.size() );

        // Extra Pruning if not sufficient
        if( asci_pairs.size() > asci_settings.pair_size_max ) {
          logger->info("    * Removing Duplicates");
          sort_and_accumulate_asci_pairs( asci_pairs );
          logger->info("    * NSZ = {}", asci_pairs.size());
        }

      } // Pruning 
    } // Unique Alpha Loop
  } // Triplet Loop

  return asci_pairs;
}

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
#if 0
  auto asci_pairs = asci_contributions_standard( asci_settings, 
    cdets_begin, cdets_end, E_ASCI, C_local, norb, T_pq, G_red,
    V_red, G_pqrs, V_pqrs, ham_gen );
#else
  auto asci_pairs = asci_contributions_triplet( asci_settings, 
    cdets_begin, cdets_end, E_ASCI, C_local, norb, T_pq, G_red,
    V_red, G_pqrs, V_pqrs, ham_gen );
#endif
  auto pairs_en = clock_type::now();


#if 0
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



  logger->info("  * ASCI Kept {} Pairs", asci_pairs.size());
  logger->info("  * Pairs Mem = {:.2e} GiB", to_gib(asci_pairs));

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

  logger->info("  * New Dets Mem = {:.2e} GiB", to_gib(new_dets));

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
