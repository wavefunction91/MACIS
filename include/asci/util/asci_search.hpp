#pragma once
#include <asci/types.hpp>
#include <asci/sd_operations.hpp>
#include <asci/util/asci_contributions.hpp>
#include <asci/util/asci_sort.hpp>
#include <asci/util/memory.hpp>
#include <asci/util/mpi.hpp>
//#include <asci/util/topk_parallel.hpp>
#include <asci/util/dist_quickselect.hpp>

#include <chrono>
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace asci {

// Generate MPI types
template <size_t N>
struct mpi_traits<std::bitset<N>> {
  using type = std::bitset<N>;
  inline static mpi_datatype datatype() { 
    return make_contiguous_mpi_datatype<char>(sizeof(type));
  }
};

template <typename WfnT>
struct mpi_traits<asci_contrib<WfnT>> {
  using type = asci_contrib<WfnT>;
  inline static mpi_datatype datatype() {
  
    type dummy;
  
    int lengths[2] = {1,1};
    MPI_Aint displacements[2];
    MPI_Aint base_address;
    MPI_Get_address(&dummy,       &base_address);
    MPI_Get_address(&dummy.state, displacements + 0);
    MPI_Get_address(&dummy.rv,    displacements + 1);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  
    auto wfn_dtype = mpi_traits<WfnT>::datatype();
    MPI_Datatype types[2] = {wfn_dtype, MPI_DOUBLE};
    MPI_Datatype custom_type;
    MPI_Type_create_struct( 2, lengths, displacements, types, &custom_type );
    MPI_Type_commit( &custom_type );
  
    return make_managed_mpi_datatype( custom_type );
    
  }
};

template <typename WfnT>
struct asci_contrib_topk_comparator {
  using type = asci_contrib<WfnT>;
  constexpr bool operator()(const type& a, const type& b) const {
    return std::abs(a.rv) > std::abs(b.rv);
  }
};



struct ASCISettings {
  size_t ntdets_max        = 1e5;
  size_t ntdets_min        = 100;
  size_t ncdets_max        = 100;
  double h_el_tol          = 1e-8;
  double rv_prune_tol      = 1e-8;
  size_t pair_size_max     = 5e8;
  bool   just_singles      = false;
  size_t grow_factor       = 8;
  size_t max_refine_iter   = 6;
  double refine_energy_tol = 1e-6;

  bool grow_with_rot    = false;
  size_t rot_size_start = 1000;

  bool dist_triplet_random = false;
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
    auto state_alpha = bitset_lo_word(state);
    auto state_beta  = bitset_hi_word(state);
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
  HamiltonianGenerator<N>&   ham_gen,
  MPI_Comm                   comm
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

    beta_coeff_data( double c, size_t norb, 
      const std::vector<uint32_t>& occ_alpha, wfn_t<N> w, 
      const HamiltonianGenerator<N>& ham_gen ) {

      coeff = c;

      // Compute Beta string 
      const auto beta_shift = w >> N/2;
      // Reduce the number of times things shift in inner loop
      beta_string = beta_shift << N/2; 
    
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

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  std::vector<size_t> world_workloads(world_size, 0);

  const auto n_occ_alpha = uniq_alpha_wfn[0].count();
  const auto n_vir_alpha = norb - n_occ_alpha;
  const auto n_sing_alpha = n_occ_alpha * n_vir_alpha;
  const auto n_doub_alpha = 
    (n_sing_alpha * ( n_sing_alpha - norb + 1 )) / 4;

  // Generate triplets
  std::vector< std::tuple<int,int,int> > triplets; 
  triplets.reserve(norb*norb*norb);
  for(int t_i = 0; t_i < norb; ++t_i)
  for(int t_j = 0; t_j < t_i;  ++t_j)
  for(int t_k = 0; t_k < t_j;  ++t_k) {

    if(world_size > 1) {
      auto [T,O,B] = 
        make_triplet_masks<N>(norb,t_i,t_j,t_k);
      size_t nw = 0;
      for( const auto& alpha : uniq_alpha_wfn ) {
         nw += 
           triplet_histogram(alpha, n_sing_alpha, n_doub_alpha, T, O, B );
      }

      if( asci_settings.dist_triplet_random ) {
        if(nw) triplets.emplace_back(t_i,t_j,t_k);
      } else {
        auto min_rank_it = 
          std::min_element(world_workloads.begin(), world_workloads.end());
        int min_rank = std::distance(world_workloads.begin(), min_rank_it);

        *min_rank_it += nw;
        if( world_rank == min_rank and nw) triplets.emplace_back(t_i,t_j,t_k);
      }
    } else {
      triplets.emplace_back(t_i,t_j,t_k);
    }
  }

  if(world_size > 1 and asci_settings.dist_triplet_random) {
    std::default_random_engine g(155039);
    std::shuffle(triplets.begin(),triplets.end(),g);
    std::vector< std::tuple<int,int,int> > local_triplets;
    local_triplets.reserve(triplets.size() / world_size);
    for( auto i = 0; i < triplets.size(); ++i) 
    if( i % world_size == world_rank ) {
      local_triplets.emplace_back(triplets[i]);
    }
    triplets = std::move(local_triplets);
  }

  //if(!world_rank) {
  //std::cout << "WORKLOADS ";
  //for( auto w : world_workloads ) std::cout << w << " ";
  //std::cout << std::endl;
  //}

  
  size_t max_size = std::min(asci_settings.pair_size_max,
    ncdets * 
      ( 2*n_sing_alpha + // AA + BB 
        2*n_doub_alpha + // AAAA + BBBB
        n_sing_alpha*n_sing_alpha // AABB
      )
  );
  asci_pairs.reserve(max_size);
  // Loop over triplets
  for( auto [t_i, t_j, t_k] : triplets ) {

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
          const auto state_beta = bitset_hi_word(beta);
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

  //std::cout << "PAIRS SIZE = " << asci_pairs.size() << std::endl;
  //if(world_size > 1) throw std::runtime_error("DIE DIE DIE");

  return asci_pairs;
}

template <size_t N>
std::vector< wfn_t<N> > asci_search( 
  ASCISettings               asci_settings, 
  size_t                     ndets_max,
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
  HamiltonianGenerator<N>&   ham_gen,
  MPI_Comm                   comm
) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  // MPI Info
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  auto logger = spdlog::get("asci_search");
  if(!logger) logger = world_rank ? 
    spdlog::null_logger_mt ("asci_search") :
    spdlog::stdout_color_mt("asci_search");
  

  // Print to logger
  const size_t ncdets = std::distance(cdets_begin, cdets_end);
  logger->info("[ASCI Search Settings]:");
  logger->info(
    "  NCDETS = {:6}, NDETS_MAX = {:9}, H_EL_TOL = {:4e}, RV_TOL = {:4e}", 
    ncdets, ndets_max, asci_settings.h_el_tol, 
    asci_settings.rv_prune_tol);
  logger->info("  MAX_RV_SIZE = {}, JUST_SINGLES = {}", 
    asci_settings.pair_size_max, asci_settings.just_singles);
  if(world_size > 1) {
    logger->info("  DIST_TRIPLET_RANDOM = {}", asci_settings.dist_triplet_random);
  }

  auto asci_search_st = clock_type::now();
  
  // Expand Search Space with Connected ASCI Contributions 
  auto pairs_st = clock_type::now();
  asci_contrib_container<wfn_t<N>> asci_pairs;
  if(world_size == 1)
    asci_pairs = asci_contributions_standard( asci_settings, 
      cdets_begin, cdets_end, E_ASCI, C, norb, T_pq, G_red,
      V_red, G_pqrs, V_pqrs, ham_gen );
  else
    asci_pairs = asci_contributions_triplet( asci_settings, 
      cdets_begin, cdets_end, E_ASCI, C, norb, T_pq, G_red,
      V_red, G_pqrs, V_pqrs, ham_gen, comm );
  auto pairs_en = clock_type::now();

  {
  size_t npairs  = allreduce( asci_pairs.size(), MPI_SUM, comm );
  logger->info("  * ASCI Kept {} Pairs", npairs);

  if(world_size > 1) {
    size_t npairs_max = allreduce( asci_pairs.size(), MPI_MAX, comm);
    size_t npairs_min = allreduce( asci_pairs.size(), MPI_MIN, comm);
    logger->info("    * PAIRS_MIN = {}, PAIRS_MAX = {}, PAIRS_AVG = {}", npairs_min, npairs_max, npairs / double(world_size) );
  } 
  logger->info("  * Pairs Mem = {:.2e} GiB", to_gib(asci_pairs));
  }

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

  {
  size_t npairs  = allreduce( asci_pairs.size(), MPI_SUM, comm );;
  logger->info("  * ASCI will search over {} unique determinants",
    npairs );

  float pairs_dur = duration_type(pairs_en - pairs_st).count();
  float bit_sort_dur = duration_type(bit_sort_en - bit_sort_st).count();
  logger->info("  * PAIR_DUR = {:.2e} s, SORT_ACC_DUR = {:.2e} s",
    pairs_dur, bit_sort_dur );

  if(world_size > 1) {
    float timings[2] = {pairs_dur, bit_sort_dur};
    float timings_max[2], timings_min[2], timings_avg[2];
    allreduce( timings, timings_max, 2, MPI_MAX, comm );
    allreduce( timings, timings_min, 2, MPI_MIN, comm );
    allreduce( timings, timings_avg, 2, MPI_SUM, comm );
    timings_avg[0] /= world_size;
    timings_avg[1] /= world_size;

    logger->info("    * PAIR_DUR_MIN = {:.2e} s, SORT_ACC_DUR_MIN = {:.2e} s",
      timings_min[0], timings_min[1] );
    logger->info("    * PAIR_DUR_MAX = {:.2e} s, SORT_ACC_DUR_MAX = {:.2e} s",
      timings_max[0], timings_max[1] );
    logger->info("    * PAIR_DUR_AVG = {:.2e} s, SORT_ACC_DUR_AVG = {:.2e} s",
      timings_avg[0], timings_avg[1] );
  }
  }

#define REMOVE_CDETS

//#ifndef REMOVE_CDETS
  //if(world_size > 1) throw std::runtime_error("MPI + !REMOVE_CDETS wont work robustly");
 
  auto keep_large_st = clock_type::now();
  // Finalize scores
  for( auto& x : asci_pairs ) x.rv = -std::abs(x.rv);

  // Insert all dets with their coefficients as seeds
  for( size_t i = 0; i < ncdets; ++i ) {
    auto state = *(cdets_begin + i);
    asci_pairs.push_back({state, std::abs(C[i])});
  }

  // Check duplicates (which correspond to the initial truncation),
  // and keep only the duplicate with positive coefficient. 
  keep_only_largest_copy_asci_pairs(asci_pairs);
 


#ifdef REMOVE_CDETS

  asci_pairs.erase(
    std::partition(asci_pairs.begin(), asci_pairs.end(),
      [](const auto& p) { return p.rv < 0.0; } ),
    asci_pairs.end()
  );

  // Only do top-K on (ndets_max - ncdets) b/c CDETS will be added later
  const size_t top_k_elements = ndets_max - ncdets;

#else

  if(world_size > 1) throw std::runtime_error("MPI + !REMOVE_CDETS wont work robustly");

  // CDETS are included in list, so search over full NDETS_MAX
  const size_t top_k_elements = ndets_max;

#endif

  auto keep_large_en = clock_type::now();
  duration_type keep_large_dur = keep_large_en - keep_large_st;
  logger->info("  * KEEP_LARGE_DUR = {:.2e} s", keep_large_dur.count() );
  if(world_size > 1) {
    float dur = keep_large_dur.count();
    auto  dmin = allreduce( dur, MPI_MIN, comm );
    auto  dmax = allreduce( dur, MPI_MAX, comm );
    auto  davg = allreduce( dur, MPI_SUM, comm ) / world_size;
    logger->info("    * KEEP_LARGE_DUR_MIN = {:.2e} s, MAX = {:.2e} s, AVG = {:.2e} s",
      dmin, dmax, davg );
  }

//#else
//
//  // Finalize the scores
//  for( auto& p : asci_pairs ) p.rv = std::abs(p.rv);
//
//  // Remove wfns in CDETS from the ranking sort
//  for( size_t i = 0; i < ncdets; ++i ) {
//    auto state = *(cdets_begin + i);
//
//    asci_contrib<wfn_t<N>> state_c{ state,  -1 };
//    auto comparator = [](const auto& a, const auto& b) { 
//        return bitset_less(a.state, b.state);
//    };
//
//    // Do binary search for state in local pairs array
//    // XXX This assumes that sort_and_accumulate_asci_pairs uses
//    //     the same comparator
//    auto it = std::lower_bound( asci_pairs.begin(), asci_pairs.end(),
//      state_c, comparator );
//
//    // If found = replace
//    // All contribs are positive, inserting a negative makes
//    // the next search easier
//    if( it != asci_pairs.end() and !comparator(state_c, *it) ) {
//      *it = state_c;
//    } 
//  }
//
//  // Perform implicit removal of CDETS
//  { // Scope temp iterator
//  auto it = std::partition(asci_pairs.begin(), asci_pairs.end(),
//    [](const auto& p) { return p.rv > 0.0; } );
//  asci_pairs.erase(it, asci_pairs.end());
//  }
//
//  // Only do top-K on (ndets_max - ncdets) b/c CDETS will be added later
//  const size_t top_k_elements = ndets_max - ncdets;
//
//#endif

  // Do Top-K to get the largest determinant contributions
  auto asci_sort_st = clock_type::now();
  if( world_size > 1 or asci_pairs.size() > top_k_elements ) {
#if 0
    std::nth_element( asci_pairs.begin(), 
      asci_pairs.begin() + top_k_elements,
      asci_pairs.end(), 
      //[](auto x, auto y){ 
      //  return std::abs(x.rv) > std::abs(y.rv);
      //}
      asci_contrib_topk_comparator<wfn_t<N>>{}
    );
    asci_pairs.erase( asci_pairs.begin() + top_k_elements, 
      asci_pairs.end() );
#else
    std::vector<asci_contrib<wfn_t<N>>> topk(top_k_elements);
    if( world_size > 1 ) {
      //topk_allreduce<512>( 
      //  asci_pairs.data(), 
      //  asci_pairs.data() + asci_pairs.size(),
      //  top_k_elements, topk.data(), 
      //  asci_contrib_topk_comparator<wfn_t<N>>{},
      //  comm 
      //);

      // Strip scores
      std::vector<double> scores(asci_pairs.size());
      std::transform(asci_pairs.begin(), asci_pairs.end(), scores.begin(),
        [](const auto& p){ return std::abs(p.rv); });

      // Determine kth-ranked scores
      auto kth_score = dist_quickselect( scores.begin(), scores.end(), 
        top_k_elements, comm, std::greater<double>{}, std::equal_to<double>{} );

      // Partition local pairs into less / eq batches
      auto [g_begin, e_begin, l_begin, _end] = 
        leg_partition( asci_pairs.begin(), asci_pairs.end(), kth_score,
          [=](const auto& p, const auto& s){ return std::abs(p.rv) >  s; },
          [=](const auto& p, const auto& s){ return std::abs(p.rv) == s; } );

      // Determine local counts
      size_t n_greater = std::distance(g_begin, e_begin);
      size_t n_equal   = std::distance(e_begin, l_begin);
      size_t n_less    = std::distance(l_begin, _end   );
      const int n_geq_local = n_greater + n_equal;

      //printf("[rank %d] KTH SCORE = %.10e\n", world_rank, kth_score);
      //printf("[rank %d] G = %lu E = %lu L = %lu\n", world_rank, n_greater, n_equal, n_less);
      
      // Strip bitsrings
      std::vector<wfn_t<N>> keep_strings_local( n_geq_local );
      std::transform( g_begin, l_begin, keep_strings_local.begin(),
        [](const auto& p){ return p.state; } );

      // Gather global strings
      std::vector<int> local_sizes, displ;
      auto n_geq_global = total_gather_and_exclusive_scan( n_geq_local,
        local_sizes, displ, comm );
      //if( n_geq_global > top_k_elements ) {
      //  printf("TOPK %d %d\n", int(top_k_elements), n_geq_global );
      //  throw std::runtime_error("Houston: We Have a Problem");
      //}

      std::vector<wfn_t<N>> keep_strings_global(n_geq_global);
      auto string_dtype = mpi_traits<wfn_t<N>>::datatype();
      MPI_Allgatherv( keep_strings_local.data(), n_geq_local, string_dtype,
        keep_strings_global.data(), local_sizes.data(), displ.data(),
        string_dtype, comm );

      // Make fake strings
      topk.resize(n_geq_global);
      std::transform(keep_strings_global.begin(), keep_strings_global.end(),
        topk.begin(), [](const auto& s) { return asci_contrib<wfn_t<N>>{s, -1.0}; });

    } else {
      std::nth_element(asci_pairs.begin(), asci_pairs.begin() + top_k_elements,
        asci_pairs.end(), asci_contrib_topk_comparator<wfn_t<N>>{} );
      std::copy(asci_pairs.begin(), asci_pairs.begin() + top_k_elements, topk.begin()); 
    }
    asci_pairs = std::move(topk);
#endif
  }
  auto asci_sort_en = clock_type::now();
  logger->info("  * ASCI_SORT_DUR = {:.2e} s", 
    duration_type(asci_sort_en - asci_sort_st).count()
  );
  if(world_size > 1) {
    float dur = duration_type(asci_sort_en - asci_sort_st).count();
    auto  dmin = allreduce( dur, MPI_MIN, comm );
    auto  dmax = allreduce( dur, MPI_MAX, comm );
    auto  davg = allreduce( dur, MPI_SUM, comm ) / world_size;
    logger->info("    * ASCI_SORT_DUR_MIN = {:.2e} s, MAX = {:.2e} s, AVG = {:.2e} s",
      dmin, dmax, davg );
  }

  // Shrink to max search space
  asci_pairs.shrink_to_fit();


  // Extract new search determinants
  std::vector<std::bitset<N>> new_dets( asci_pairs.size() );
  std::transform( asci_pairs.begin(), asci_pairs.end(), new_dets.begin(),
    [](auto x){ return x.state; } );

#ifdef REMOVE_CDETS
  // Insert the CDETS back in
  new_dets.insert(new_dets.end(), cdets_begin, cdets_end);
  new_dets.shrink_to_fit();
#endif

  logger->info("  * New Dets Mem = {:.2e} GiB", to_gib(new_dets));

  // Ensure consistent ordering
#if 0
  if(world_size > 1) {
    std::sort(new_dets.begin(), new_dets.end(),
      bitset_less_comparator<N>{});

    // Not guranteed to get the SAME list of eqivalent scored
    // wfns... Sync to be sure
    bcast( new_dets.data(), new_dets.size(), 0, comm );
  }
#endif

#if 0
  if(!world_rank) {
  std::sort(new_dets.begin(), new_dets.end(),
  [](auto x, auto y){ return bitset_less(x,y); });
  std::cout << "NEW DETS " << new_dets.size() << std::endl;
  for( auto s : new_dets ) {
    std::cout << to_canonical_string(s) << std::endl;
  }
  }
  MPI_Barrier(comm);
#else
  //{
  //std::sort(new_dets.begin(), new_dets.end(),
  //  bitset_less_comparator<N>{});
  //std::ofstream wfn_file("wfn." + std::to_string(world_rank) + ".txt");
  //for( auto s : new_dets ) {
  //  wfn_file << to_canonical_string(s) << std::endl;
  //}
  //}
  //throw std::runtime_error("DIE DIE DIE");
#endif

  auto asci_search_en = clock_type::now();
  duration_type asci_search_dur = asci_search_en - asci_search_st;
  logger->info("  * ASCI_SEARCH DUR = {:.2e} s", asci_search_dur.count() );
  return new_dets;
}

} // namespace asci
