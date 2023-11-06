/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <omp.h>
#include <chrono>
#include <fstream>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/asci/determinant_sort.hpp>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <macis/util/dist_quickselect.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/mpi.hpp>

namespace macis {

template <typename WfnT>
struct asci_contrib_topk_comparator {
  using type = asci_contrib<WfnT>;
  constexpr bool operator()(const type& a, const type& b) const {
    return std::abs(a.rv()) > std::abs(b.rv());
  }
};

struct ASCISettings {
  size_t ntdets_max = 1e5;
  size_t ntdets_min = 100;
  size_t ncdets_max = 100;
  double h_el_tol = 1e-8;
  double rv_prune_tol = 1e-8;
  size_t pair_size_max = 5e8;
  bool just_singles = false;
  size_t grow_factor = 8;
  size_t max_refine_iter = 6;
  double refine_energy_tol = 1e-6;

  bool grow_with_rot = false;
  size_t rot_size_start = 1000;

  // bool dist_triplet_random = false;
  int constraint_level = 2;  // Up To Quints
};

template <size_t N>
asci_contrib_container<wfn_t<N>> asci_contributions_standard(
    ASCISettings asci_settings, wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs, HamiltonianGenerator<wfn_t<N>>& ham_gen) {
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  auto logger = spdlog::get("asci_search");

  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  asci_contrib_container<wfn_t<N>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;
  asci_pairs.reserve(asci_settings.pair_size_max);
  for(size_t i = 0; i < ncdets; ++i) {
    // Alias state data
    auto state = *(cdets_begin + i);
    auto state_alpha = wfn_traits::alpha_string(state);
    auto state_beta = wfn_traits::beta_string(state);
    auto coeff = C[i];

    // Get occupied and virtual indices
    spin_wfn_traits::state_to_occ_vir(norb, state_alpha, occ_alpha, vir_alpha);
    spin_wfn_traits::state_to_occ_vir(norb, state_beta, occ_beta, vir_beta);

    // Precompute orbital energies
    auto eps_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
    auto eps_beta = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);

    // Compute base diagonal matrix element
    double h_diag = ham_gen.matrix_element(state, state);

    const double h_el_tol = asci_settings.h_el_tol;

    // Singles - AA
    append_singles_asci_contributions<Spin::Alpha>(
        coeff, state, state_alpha, occ_alpha, vir_alpha, occ_beta,
        eps_alpha.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol,
        h_diag, E_ASCI, ham_gen, asci_pairs);

    // Singles - BB
    append_singles_asci_contributions<Spin::Beta>(
        coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
        eps_beta.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol, h_diag,
        E_ASCI, ham_gen, asci_pairs);

    if(not asci_settings.just_singles) {
      // Doubles - AAAA
      append_ss_doubles_asci_contributions<Spin::Alpha>(
          coeff, state, state_alpha, state_beta, occ_alpha, vir_alpha, occ_beta,
          eps_alpha.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
          asci_pairs);

      // Doubles - BBBB
      append_ss_doubles_asci_contributions<Spin::Beta>(
          coeff, state, state_beta, state_alpha, occ_beta, vir_beta, occ_alpha,
          eps_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
          asci_pairs);

      // Doubles - AABB
      append_os_doubles_asci_contributions(
          coeff, state, state_alpha, state_beta, occ_alpha, occ_beta, vir_alpha,
          vir_beta, eps_alpha.data(), eps_beta.data(), V_pqrs, norb, h_el_tol,
          h_diag, E_ASCI, ham_gen, asci_pairs);
    }

    // Prune Down Contributions
    if(asci_pairs.size() > asci_settings.pair_size_max) {
      // Remove small contributions
      auto it = std::partition(
          asci_pairs.begin(), asci_pairs.end(), [=](const auto& x) {
            return std::abs(x.rv()) > asci_settings.rv_prune_tol;
          });
      asci_pairs.erase(it, asci_pairs.end());
      logger->info("  * Pruning at DET = {} NSZ = {}", i, asci_pairs.size());

      // Extra Pruning if not sufficient
      if(asci_pairs.size() > asci_settings.pair_size_max) {
        logger->info("    * Removing Duplicates");
        sort_and_accumulate_asci_pairs(asci_pairs);
        logger->info("    * NSZ = {}", asci_pairs.size());
      }

    }  // Pruning
  }    // Loop over search determinants

  return asci_pairs;
}

#ifdef MACIS_ENABLE_MPI
template <size_t N>
asci_contrib_container<wfn_t<N>> asci_contributions_constraint(
    ASCISettings asci_settings, const size_t ntdets,
    wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs, HamiltonianGenerator<wfn_t<N>>& ham_gen,
    MPI_Comm comm) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  using wfn_comp = typename wfn_traits::spin_comparator;
  if(!std::is_sorted(cdets_begin, cdets_end, wfn_comp{}))
    throw std::runtime_error("ASCI Search Only Works with Sorted Wfns");

  auto logger = spdlog::get("asci_search");
  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  // For each unique alpha, create a list of beta string and store metadata
  struct beta_coeff_data {
    spin_wfn_type beta_string;
    std::vector<uint32_t> occ_beta;
    std::vector<uint32_t> vir_beta;
    std::vector<double> orb_ens_alpha;
    std::vector<double> orb_ens_beta;
    double coeff;
    double h_diag;

    beta_coeff_data(double c, size_t norb,
                    const std::vector<uint32_t>& occ_alpha, wfn_t<N> w,
                    const HamiltonianGenerator<wfn_t<N>>& ham_gen) {
      coeff = c;

      beta_string = wfn_traits::beta_string(w);

      // Compute diagonal matrix element
      h_diag = ham_gen.matrix_element(w, w);

      // Compute occ/vir for beta string
      spin_wfn_traits::state_to_occ_vir(norb, beta_string, occ_beta, vir_beta);

      // Precompute orbital energies
      orb_ens_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
      orb_ens_beta = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);
    }
  };

  auto uniq_alpha = get_unique_alpha(cdets_begin, cdets_end);
  const size_t nuniq_alpha = uniq_alpha.size();
  std::vector<wfn_t<N>> uniq_alpha_wfn(nuniq_alpha);
  std::transform(
      uniq_alpha.begin(), uniq_alpha.end(), uniq_alpha_wfn.begin(),
      [](const auto& p) { return wfn_traits::from_spin(p.first, 0); });

  using unique_alpha_data = std::vector<beta_coeff_data>;
  std::vector<unique_alpha_data> uad(nuniq_alpha);
  for(auto i = 0, iw = 0; i < nuniq_alpha; ++i) {
    std::vector<uint32_t> occ_alpha, vir_alpha;
    spin_wfn_traits::state_to_occ_vir(norb, uniq_alpha[i].first, occ_alpha,
                                      vir_alpha);

    const auto nbeta = uniq_alpha[i].second;
    uad[i].reserve(nbeta);
    for(auto j = 0; j < nbeta; ++j, ++iw) {
      const auto& w = *(cdets_begin + iw);
      uad[i].emplace_back(C[iw], norb, occ_alpha, w, ham_gen);
    }
  }


  // const auto n_occ_alpha = wfn_traits::count(uniq_alpha_wfn[0]);
  const auto n_occ_alpha = spin_wfn_traits::count(uniq_alpha[0].first);
  const auto n_vir_alpha = norb - n_occ_alpha;
  const auto n_sing_alpha = n_occ_alpha * n_vir_alpha;
  const auto n_doub_alpha = (n_sing_alpha * (n_sing_alpha - norb + 1)) / 4;

  const auto n_occ_beta = cdets_begin->count() - n_occ_alpha;
  const auto n_vir_beta = norb - n_occ_beta;
  const auto n_sing_beta = n_occ_beta * n_vir_beta;
  const auto n_doub_beta = (n_sing_beta * (n_sing_beta - norb + 1)) / 4;

  // logger->info("  * NS = {} ND = {}", n_sing_alpha, n_doub_alpha);

  // Generate mask constraints
  if(!world_rank) {
    std::string cl_string;
    switch(asci_settings.constraint_level) {
      case 0:
        cl_string = "Triplets";
        break;
      case 1:
        cl_string = "Quadruplets";
        break;
      case 2:
        cl_string = "Quintuplets";
        break;
      case 3:
        cl_string = "Hextuplets";
        break;
      default:
        cl_string = "Something I dont recognize (" +
                    std::to_string(asci_settings.constraint_level) + ")";
        break;
    }
    logger->info("  * Will Generate up to {}", cl_string);
  }

  auto gen_c_st = clock_type::now();
  auto constraints = gen_constraints_general<wfn_t<N>>(
      asci_settings.constraint_level, norb, n_sing_beta, 
      n_doub_beta, uniq_alpha, world_size);
  auto gen_c_en = clock_type::now();
  duration_type gen_c_dur = gen_c_en - gen_c_st;
  logger->info("  * GEN_DUR = {:.2e} ms", gen_c_dur.count());

  size_t max_size =
      std::min(std::min(ntdets,asci_settings.pair_size_max),
               ncdets * (n_sing_alpha + n_sing_beta +  // AA + BB
                         n_doub_alpha + n_doub_beta +  // AAAA + BBBB
                         n_sing_alpha * n_sing_beta    // AABB
                         ));

  const size_t ncon_total = constraints.size();

  // Global atomic task-id counter
  global_atomic<size_t> nxtval(comm);

  asci_contrib_container<wfn_t<N>> asci_pairs_total;
#pragma omp parallel
  {
  // Process ASCI pair contributions for each constraint
  asci_contrib_container<wfn_t<N>> asci_pairs;
  asci_pairs.reserve(max_size);

  size_t ic = 0;
  while(ic < ncon_total) {
    auto size_before = asci_pairs.size();
    const double h_el_tol = asci_settings.h_el_tol;

    // Atomically get the next task ID and increment for other
    // MPI ranks and threads
    size_t ntake = ic < 1000 ? 1 : 10;
    ic = nxtval.fetch_and_add(ntake);

    // Loop over assigned tasks
    const size_t c_end = std::min(ncon_total, ic + ntake);
    for(; ic < c_end; ++ic) {
      const auto& con = constraints[ic].first;
      //printf("[rank %4d tid:%4d] %10lu / %10lu\n", world_rank,
      //       omp_get_thread_num(), ic, ncon_total);

    for(size_t i_alpha = 0, iw = 0; i_alpha < nuniq_alpha; ++i_alpha) {
      const auto& alpha_det = uniq_alpha[i_alpha].first;
      const auto occ_alpha = bits_to_indices(alpha_det);
      const bool alpha_satisfies_con = satisfies_constraint(alpha_det, con);

      const auto& bcd = uad[i_alpha];
      const size_t nbeta = bcd.size();
      for(size_t j_beta = 0; j_beta < nbeta; ++j_beta, ++iw) {
        const auto w = *(cdets_begin + iw);
        const auto c = C[iw];
        const auto& beta_det = bcd[j_beta].beta_string;
        const auto h_diag = bcd[j_beta].h_diag;
        const auto& occ_beta = bcd[j_beta].occ_beta;
        const auto& vir_beta = bcd[j_beta].vir_beta;
        const auto& orb_ens_alpha = bcd[j_beta].orb_ens_alpha;
        const auto& orb_ens_beta = bcd[j_beta].orb_ens_beta;

        // AA excitations
        generate_constraint_singles_contributions_ss(
            c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), T_pq, norb,
            G_red, norb, V_red, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
            asci_pairs);

        // AAAA excitations
        generate_constraint_doubles_contributions_ss(
            c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), G_pqrs, norb,
            h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

        // AABB excitations
        generate_constraint_doubles_contributions_os(
            c, w, con, occ_alpha, occ_beta, vir_beta, orb_ens_alpha.data(),
            orb_ens_beta.data(), V_pqrs, norb, h_el_tol, h_diag, E_ASCI,
            ham_gen, asci_pairs);

        if(alpha_satisfies_con) {
          // BB excitations
          append_singles_asci_contributions<Spin::Beta>(
              c, w, beta_det, occ_beta, vir_beta, occ_alpha,
              orb_ens_beta.data(), T_pq, norb, G_red, norb, V_red, norb,
              h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

          // BBBB excitations
          append_ss_doubles_asci_contributions<Spin::Beta>(
              c, w, beta_det, alpha_det, occ_beta, vir_beta, occ_alpha,
              orb_ens_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI,
              ham_gen, asci_pairs);

          // No excitation (push inf to remove from list)
          asci_pairs.push_back(
              {w, std::numeric_limits<double>::infinity(), 1.0});
        }
      }

      // Prune Down Contributions
      if(asci_pairs.size() > asci_settings.pair_size_max) {
        throw std::runtime_error("DIE DIE DIE");
      }

    }  // Unique Alpha Loop


    // Local S&A for each quad
    {
      if(size_before > asci_pairs.size())
        throw std::runtime_error("DIE DIE DIE");
      auto uit = sort_and_accumulate_asci_pairs(
          asci_pairs.begin() + size_before, asci_pairs.end());
      asci_pairs.erase(uit, asci_pairs.end());

      // Remove small contributions
      uit =
          std::partition(asci_pairs.begin() + size_before, asci_pairs.end(),
                         [=](const auto& x) {
                           return std::abs(x.rv()) > asci_settings.rv_prune_tol;
                         });
      asci_pairs.erase(uit, asci_pairs.end());
    }
    } // Loc constraint loop
  }  // Constraint Loop

  // Insert into list
  #pragma omp critical
  {
  if(asci_pairs_total.size()) {
    // Preallocate space for insertion
    asci_pairs_total.reserve(asci_pairs.size() + asci_pairs_total.size());
    asci_pairs_total.insert(asci_pairs_total.end(), asci_pairs.begin(),
      asci_pairs.end());
  } else {
    asci_pairs_total = std::move(asci_pairs);
  }
  asci_contrib_container<wfn_t<N>>().swap(asci_pairs);
  }

  } // OpenMP

  return asci_pairs_total;
}
#endif

template <size_t N>
std::vector<wfn_t<N>> asci_search(
    ASCISettings asci_settings, size_t ndets_max,
    wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs,
    HamiltonianGenerator<wfn_t<N>>& ham_gen MACIS_MPI_CODE(, MPI_Comm comm)) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  // MPI Info
#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
#else
  int world_rank = 0;
  int world_size = 1;
#endif

  auto logger = spdlog::get("asci_search");
  if(!logger)
    logger = world_rank ? spdlog::null_logger_mt("asci_search")
                        : spdlog::stdout_color_mt("asci_search");

#ifdef MACIS_ENABLE_MPI
  auto print_mpi_stats = [&](auto str, auto vmin, auto vmax, auto vavg) {
    std::string fmt_string =
        "    * {0}_MIN = {1}, {0}_MAX = {2}, {0}_AVG = {3}, RATIO = {4:.2e}";
    if constexpr(std::is_floating_point_v<std::decay_t<decltype(vmin)>>)
      fmt_string =
          "    * {0}_MIN = {1:.2e}, {0}_MAX = {2:.2e}, {0}_AVG = {3:.2e}, "
          "RATIO = {4:.2e}";
    logger->info(fmt_string, str, vmin, vmax, vavg, vmax / float(vmin));
  };
#endif

  // Print Search Header to logger
  const size_t ncdets = std::distance(cdets_begin, cdets_end);
  logger->info("[ASCI Search Settings]:");
  logger->info(
      "  NCDETS = {:6}, NDETS_MAX = {:9}, H_EL_TOL = {:4e}, RV_TOL = {:4e}",
      ncdets, ndets_max, asci_settings.h_el_tol, asci_settings.rv_prune_tol);
  logger->info("  MAX_RV_SIZE = {}, JUST_SINGLES = {}",
               asci_settings.pair_size_max, asci_settings.just_singles);

  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto asci_search_st = clock_type::now();

  // Expand Search Space with Connected ASCI Contributions
  auto pairs_st = clock_type::now();
  asci_contrib_container<wfn_t<N>> asci_pairs;
  // if(world_size == 1)
  //   asci_pairs = asci_contributions_standard(
  //       asci_settings, cdets_begin, cdets_end, E_ASCI, C, norb, T_pq, G_red,
  //       V_red, G_pqrs, V_pqrs, ham_gen);
  // #ifdef MACIS_ENABLE_MPI
  // else
  asci_pairs = asci_contributions_constraint(
      asci_settings, ndets_max, cdets_begin, cdets_end, E_ASCI, C, norb, T_pq, G_red,
      V_red, G_pqrs, V_pqrs, ham_gen MACIS_MPI_CODE(, comm));
  // #endif
  auto pairs_en = clock_type::now();

  {
#ifdef MACIS_ENABLE_MPI
    size_t npairs = allreduce(asci_pairs.size(), MPI_SUM, comm);
#else
    size_t npairs = asci_pairs.size();
#endif
    logger->info("  * ASCI Kept {} Pairs", npairs);
    if(npairs < ndets_max)
      logger->info("    * WARNING: Kept ASCI pairs less than requested TDETS");

#ifdef MACIS_ENABLE_MPI
    if(world_size > 1) {
      size_t npairs_max = allreduce(asci_pairs.size(), MPI_MAX, comm);
      size_t npairs_min = allreduce(asci_pairs.size(), MPI_MIN, comm);
      print_mpi_stats("PAIRS_LOC", npairs_min, npairs_max, npairs / world_size);
    }
#endif

    if(world_size == 1) {
      logger->info("  * Pairs Mem = {:.2e} GiB", to_gib(asci_pairs));
    } else {
#ifdef MACIS_ENABLE_MPI
      float local_mem = to_gib(asci_pairs);
      float total_mem = allreduce(local_mem, MPI_SUM, comm);
      float min_mem = allreduce(local_mem, MPI_MIN, comm);
      float max_mem = allreduce(local_mem, MPI_MAX, comm);
      print_mpi_stats("PAIRS_MEM", min_mem, max_mem, total_mem / world_size);
#endif
    }
  }

#if 0
  std::cout << "ASCI PAIRS" << std::endl;
  for(auto [s,rv] : asci_pairs) {
    std::cout << to_canonical_string(s) << ", " << std::scientific << rv << std::endl;
  }
#endif

  // Accumulate unique score contributions
  // MPI + Constraint Search already does S&A
  auto bit_sort_st = clock_type::now();
  if(world_size == 1) sort_and_accumulate_asci_pairs(asci_pairs);
  auto bit_sort_en = clock_type::now();

  {
#ifdef MACIS_ENABLE_MPI
    size_t npairs = allreduce(asci_pairs.size(), MPI_SUM, comm);
#else
    size_t npairs = asci_pairs.size();
#endif
    logger->info("  * ASCI will search over {} unique determinants", npairs);

    float pairs_dur = duration_type(pairs_en - pairs_st).count();
    float bit_sort_dur = duration_type(bit_sort_en - bit_sort_st).count();

    if(world_size > 1) {
#ifdef MACIS_ENABLE_MPI
      float timings = pairs_dur;
      float timings_max, timings_min, timings_avg;
      allreduce(&timings, &timings_max, 1, MPI_MAX, comm);
      allreduce(&timings, &timings_min, 1, MPI_MIN, comm);
      allreduce(&timings, &timings_avg, 1, MPI_SUM, comm);
      timings_avg /= world_size;
      print_mpi_stats("PAIRS_DUR", timings_min, timings_max, timings_avg);
#endif
    } else {
      logger->info("  * PAIR_DUR = {:.2e} s, SORT_ACC_DUR = {:.2e} s",
                   pairs_dur, bit_sort_dur);
    }
  }

  auto keep_large_st = clock_type::now();
#if 0
  // Finalize scores
  for(auto& x : asci_pairs) {
    x.c_times_matel = -std::abs(x.c_times_matel);
    x.h_diag = std::abs(x.h_diag);
  }

  // Insert all dets with their coefficients as seeds
  for(size_t i = 0; i < ncdets; ++i) {
    auto state = *(cdets_begin + i);
    asci_pairs.push_back({state, std::abs(C[i])});
  }

  // Check duplicates (which correspond to the initial truncation),
  // and keep only the duplicate with positive coefficient.
  keep_only_largest_copy_asci_pairs(asci_pairs);

  asci_pairs.erase(std::partition(asci_pairs.begin(), asci_pairs.end(),
                                  [](const auto& p) { return p.rv() < 0.0; }),
                   asci_pairs.end());

#else
  // Remove core dets
  // XXX: This assumes the constraint search for now
  {
    auto inf_ptr = std::partition(asci_pairs.begin(), asci_pairs.end(),
                                  [](auto& p) { return !std::isinf(p.rv()); });
    asci_pairs.erase(inf_ptr, asci_pairs.end());
  }
#endif

  auto keep_large_en = clock_type::now();
  duration_type keep_large_dur = keep_large_en - keep_large_st;
  if(world_size > 1) {
#ifdef MACIS_ENABLE_MPI
    float dur = keep_large_dur.count();
    auto dmin = allreduce(dur, MPI_MIN, comm);
    auto dmax = allreduce(dur, MPI_MAX, comm);
    auto davg = allreduce(dur, MPI_SUM, comm) / world_size;
    print_mpi_stats("KEEP_LARG_DUR", dmin, dmax, davg);
#endif
  } else {
    logger->info("  * KEEP_LARG_DUR = {:.2e} s", keep_large_dur.count());
  }

  // Only do top-K on (ndets_max - ncdets) b/c CDETS will be added later
  const size_t top_k_elements = ndets_max - ncdets;

  // Do Top-K to get the largest determinant contributions
  auto asci_sort_st = clock_type::now();
  if(world_size > 1 or asci_pairs.size() > top_k_elements) {
    std::vector<asci_contrib<wfn_t<N>>> topk(top_k_elements);
    if(world_size > 1) {
#ifdef MACIS_ENABLE_MPI
      // Strip scores
      std::vector<double> scores(asci_pairs.size());
      std::transform(asci_pairs.begin(), asci_pairs.end(), scores.begin(),
                     [](const auto& p) { return std::abs(p.rv()); });

      // Determine kth-ranked scores
      auto kth_score =
          dist_quickselect(scores.begin(), scores.end(), top_k_elements, comm,
                           std::greater<double>{}, std::equal_to<double>{});

      logger->info("  * Kth Score Pivot = {:.2e}", kth_score);
      // Partition local pairs into less / eq batches
      auto [g_begin, e_begin, l_begin, _end] = leg_partition(
          asci_pairs.begin(), asci_pairs.end(), kth_score,
          [=](const auto& p, const auto& s) { return std::abs(p.rv()) > s; },
          [=](const auto& p, const auto& s) { return std::abs(p.rv()) == s; });

      // Determine local counts
      size_t n_greater = std::distance(g_begin, e_begin);
      size_t n_equal = std::distance(e_begin, l_begin);
      size_t n_less = std::distance(l_begin, _end);
      const int n_geq_local = n_greater + n_equal;

      // Strip bitsrings
      std::vector<wfn_t<N>> keep_strings_local(n_geq_local);
      std::transform(g_begin, l_begin, keep_strings_local.begin(),
                     [](const auto& p) { return p.state; });

      // Gather global strings
      std::vector<int> local_sizes, displ;
      auto n_geq_global = total_gather_and_exclusive_scan(
          n_geq_local, local_sizes, displ, comm);

      std::vector<wfn_t<N>> keep_strings_global(n_geq_global);
      auto string_dtype = mpi_traits<wfn_t<N>>::datatype();
      MPI_Allgatherv(keep_strings_local.data(), n_geq_local, string_dtype,
                     keep_strings_global.data(), local_sizes.data(),
                     displ.data(), string_dtype, comm);

      // Edge case where NGEQ > TOPK - erase equivalent elements
      if(n_geq_global > top_k_elements) {
        n_geq_global = top_k_elements;
        keep_strings_global.resize(n_geq_global);
      }

      // Make fake strings
      topk.resize(n_geq_global);
      std::transform(keep_strings_global.begin(), keep_strings_global.end(),
                     topk.begin(), [](const auto& s) {
                       return asci_contrib<wfn_t<N>>{s, -1.0};
                     });

#endif
    } else {
      std::nth_element(asci_pairs.begin(), asci_pairs.begin() + top_k_elements,
                       asci_pairs.end(),
                       asci_contrib_topk_comparator<wfn_t<N>>{});
      std::copy(asci_pairs.begin(), asci_pairs.begin() + top_k_elements,
                topk.begin());
    }
    asci_pairs = std::move(topk);
  }
  auto asci_sort_en = clock_type::now();
  if(world_size > 1) {
#ifdef MACIS_ENABLE_MPI
    float dur = duration_type(asci_sort_en - asci_sort_st).count();
    auto dmin = allreduce(dur, MPI_MIN, comm);
    auto dmax = allreduce(dur, MPI_MAX, comm);
    auto davg = allreduce(dur, MPI_SUM, comm) / world_size;
    print_mpi_stats("ASCI_SORT_DUR", dmin, dmax, davg);
#endif
  } else {
    logger->info("  * ASCI_SORT_DUR = {:.2e} s",
                 duration_type(asci_sort_en - asci_sort_st).count());
  }

  // Shrink to max search space
  asci_pairs.shrink_to_fit();

  // Extract new search determinants
  std::vector<std::bitset<N>> new_dets(asci_pairs.size());
  std::transform(asci_pairs.begin(), asci_pairs.end(), new_dets.begin(),
                 [](auto x) { return x.state; });

  // Insert the CDETS back in
  new_dets.insert(new_dets.end(), cdets_begin, cdets_end);
  new_dets.shrink_to_fit();

  logger->info("  * New Dets Mem = {:.2e} GiB", to_gib(new_dets));

  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto asci_search_en = clock_type::now();
  duration_type asci_search_dur = asci_search_en - asci_search_st;
  logger->info("  * ASCI_SEARCH DUR = {:.2e} s", asci_search_dur.count());
  return new_dets;
}

}  // namespace macis
