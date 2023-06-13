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
    return std::abs(a.rv) > std::abs(b.rv);
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
    const double* V_pqrs, HamiltonianGenerator<N>& ham_gen) {
  auto logger = spdlog::get("asci_search");

  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  asci_contrib_container<wfn_t<N>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;
  asci_pairs.reserve(asci_settings.pair_size_max);
  for(size_t i = 0; i < ncdets; ++i) {
    // Alias state data
    auto state = *(cdets_begin + i);
    auto state_alpha = bitset_lo_word(state);
    auto state_beta = bitset_hi_word(state);
    auto coeff = C[i];

    // Get occupied and virtual indices
    bitset_to_occ_vir(norb, state_alpha, occ_alpha, vir_alpha);
    bitset_to_occ_vir(norb, state_beta, occ_beta, vir_beta);

    // Precompute orbital energies
    auto eps_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
    auto eps_beta = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);

    // Compute base diagonal matrix element
    double h_diag = ham_gen.matrix_element(state, state);

    const double h_el_tol = asci_settings.h_el_tol;

    // Singles - AA
    append_singles_asci_contributions<(N / 2), 0>(
        coeff, state, state_alpha, occ_alpha, vir_alpha, occ_beta,
        eps_alpha.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol,
        h_diag, E_ASCI, ham_gen, asci_pairs);

    // Singles - BB
    append_singles_asci_contributions<(N / 2), (N / 2)>(
        coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
        eps_beta.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol, h_diag,
        E_ASCI, ham_gen, asci_pairs);

    if(not asci_settings.just_singles) {
      // Doubles - AAAA
      append_ss_doubles_asci_contributions<N / 2, 0>(
          coeff, state, state_alpha, occ_alpha, vir_alpha, occ_beta,
          eps_alpha.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
          asci_pairs);

      // Doubles - BBBB
      append_ss_doubles_asci_contributions<N / 2, N / 2>(
          coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
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
            return std::abs(x.rv) > asci_settings.rv_prune_tol;
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
    ASCISettings asci_settings, wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs, HamiltonianGenerator<N>& ham_gen, MPI_Comm comm) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  auto logger = spdlog::get("asci_search");
  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  asci_contrib_container<wfn_t<N>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;

  // Get unique alpha strings
  std::vector<wfn_t<N>> uniq_alpha_wfn(cdets_begin, cdets_end);
  std::transform(uniq_alpha_wfn.begin(), uniq_alpha_wfn.end(),
                 uniq_alpha_wfn.begin(),
                 [=](const auto& w) { return w & full_mask<N / 2, N>(); });
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
    std::vector<double> orb_ens_alpha;
    std::vector<double> orb_ens_beta;
    double coeff;
    double h_diag;

    beta_coeff_data(double c, size_t norb,
                    const std::vector<uint32_t>& occ_alpha, wfn_t<N> w,
                    const HamiltonianGenerator<N>& ham_gen) {
      coeff = c;

      // Compute Beta string
      const auto beta_shift = w >> N / 2;
      // Reduce the number of times things shift in inner loop
      beta_string = beta_shift << N / 2;

      // Compute diagonal matrix element
      h_diag = ham_gen.matrix_element(w, w);

      // Compute occ/vir for beta string
      bitset_to_occ_vir(norb, beta_shift, occ_beta, vir_beta);

      // Precompute orbital energies
      orb_ens_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
      orb_ens_beta = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);
    }
  };

  struct unique_alpha_data {
    std::vector<beta_coeff_data> bcd;
  };

  std::vector<unique_alpha_data> uad(nuniq_alpha);
  for(auto i = 0; i < nuniq_alpha; ++i) {
    const auto wfn_a = uniq_alpha_wfn[i];
    std::vector<uint32_t> occ_alpha, vir_alpha;
    bitset_to_occ_vir(norb, wfn_a, occ_alpha, vir_alpha);
    for(auto j = 0; j < ncdets; ++j) {
      const auto w = *(cdets_begin + j);
      if((w & full_mask<N / 2, N>()) == wfn_a) {
        uad[i].bcd.emplace_back(C[j], norb, occ_alpha, w, ham_gen);
      }
    }
  }

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  const auto n_occ_alpha = uniq_alpha_wfn[0].count();
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
  auto constraints =
      dist_constraint_general(asci_settings.constraint_level, norb,
                              n_sing_alpha, n_doub_alpha, uniq_alpha_wfn, comm);
  auto gen_c_en = clock_type::now();
  duration_type gen_c_dur = gen_c_en - gen_c_st;
  logger->info("  * GEN_DUR = {:.2e} ms", gen_c_dur.count());

  size_t max_size =
      std::min(asci_settings.pair_size_max,
               ncdets * (n_sing_alpha + n_sing_beta +  // AA + BB
                         n_doub_alpha + n_sing_beta +  // AAAA + BBBB
                         n_sing_alpha * n_sing_beta    // AABB
                         ));
  asci_pairs.reserve(max_size);

  // Process ASCI pair contributions for each constraint
  for(auto con : constraints) {
    auto size_before = asci_pairs.size();

    const double h_el_tol = asci_settings.h_el_tol;
    const auto& [C, B, C_min] = con;
    wfn_t<N> O = full_mask<N>(norb);

    // Loop over unique alpha strings
    for(size_t i_alpha = 0; i_alpha < nuniq_alpha; ++i_alpha) {
      const auto& det = uniq_alpha_wfn[i_alpha];
      const auto occ_alpha = bits_to_indices(det);

      // AA excitations
      for(const auto& bcd : uad[i_alpha].bcd) {
        const auto& beta = bcd.beta_string;
        const auto& coeff = bcd.coeff;
        const auto& h_diag = bcd.h_diag;
        const auto& occ_beta = bcd.occ_beta;
        const auto& orb_ens_alpha = bcd.orb_ens_alpha;
        generate_constraint_singles_contributions_ss(
            coeff, det, C, O, B, beta, occ_alpha, occ_beta,
            orb_ens_alpha.data(), T_pq, norb, G_red, norb, V_red, norb,
            h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);
      }

      // AAAA excitations
      for(const auto& bcd : uad[i_alpha].bcd) {
        const auto& beta = bcd.beta_string;
        const auto& coeff = bcd.coeff;
        const auto& h_diag = bcd.h_diag;
        const auto& occ_beta = bcd.occ_beta;
        const auto& orb_ens_alpha = bcd.orb_ens_alpha;
        generate_constraint_doubles_contributions_ss(
            coeff, det, C, O, B, beta, occ_alpha, occ_beta,
            orb_ens_alpha.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI,
            ham_gen, asci_pairs);
      }

      // AABB excitations
      for(const auto& bcd : uad[i_alpha].bcd) {
        const auto& beta = bcd.beta_string;
        const auto& coeff = bcd.coeff;
        const auto& h_diag = bcd.h_diag;
        const auto& occ_beta = bcd.occ_beta;
        const auto& vir_beta = bcd.vir_beta;
        const auto& orb_ens_alpha = bcd.orb_ens_alpha;
        const auto& orb_ens_beta = bcd.orb_ens_beta;
        generate_constraint_doubles_contributions_os(
            coeff, det, C, O, B, beta, occ_alpha, occ_beta, vir_beta,
            orb_ens_alpha.data(), orb_ens_beta.data(), V_pqrs, norb, h_el_tol,
            h_diag, E_ASCI, ham_gen, asci_pairs);
      }

      // If the alpha determinant satisfies the constraint,
      // append BB and BBBB excitations
      if(satisfies_constraint(det, C, C_min)) {
        for(const auto& bcd : uad[i_alpha].bcd) {
          const auto& beta = bcd.beta_string;
          const auto& coeff = bcd.coeff;
          const auto& h_diag = bcd.h_diag;
          const auto& occ_beta = bcd.occ_beta;
          const auto& vir_beta = bcd.vir_beta;
          const auto& eps_beta = bcd.orb_ens_beta;

          const auto state = det | beta;
          const auto state_beta = bitset_hi_word(beta);
          // BB Excitations
          append_singles_asci_contributions<(N / 2), (N / 2)>(
              coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
              eps_beta.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol,
              h_diag, E_ASCI, ham_gen, asci_pairs);

          // BBBB Excitations
          append_ss_doubles_asci_contributions<N / 2, N / 2>(
              coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
              eps_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
              asci_pairs);

        }  // Beta Loop
      }    // Triplet Check

      // Prune Down Contributions
      if(asci_pairs.size() > asci_settings.pair_size_max) {
        // Remove small contributions
        auto it = std::partition(
            asci_pairs.begin(), asci_pairs.end(), [=](const auto& x) {
              return std::abs(x.rv) > asci_settings.rv_prune_tol;
            });
        asci_pairs.erase(it, asci_pairs.end());

        auto c_indices = bits_to_indices(C);
        std::string c_string;
        for(int i = 0; i < c_indices.size(); ++i)
          c_string += std::to_string(c_indices[i]) + " ";
        logger->info("  * Pruning at CON = {}, NSZ = {}", c_string,
                     asci_pairs.size());

        // Extra Pruning if not sufficient
        if(asci_pairs.size() > asci_settings.pair_size_max) {
          logger->info("    * Removing Duplicates");
          auto uit = sort_and_accumulate_asci_pairs(
              asci_pairs.begin() + size_before, asci_pairs.end());
          asci_pairs.erase(uit, asci_pairs.end());
          logger->info("    * NSZ = {}", asci_pairs.size());
        }

      }  // Pruning
    }    // Unique Alpha Loop

    // Local S&A for each quad
    {
      auto uit = sort_and_accumulate_asci_pairs(
          asci_pairs.begin() + size_before, asci_pairs.end());
      asci_pairs.erase(uit, asci_pairs.end());
    }
  }  // Constraint Loop

  return asci_pairs;
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
    HamiltonianGenerator<N>& ham_gen MACIS_MPI_CODE(, MPI_Comm comm)) {
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
  if(world_size == 1)
    asci_pairs = asci_contributions_standard(
        asci_settings, cdets_begin, cdets_end, E_ASCI, C, norb, T_pq, G_red,
        V_red, G_pqrs, V_pqrs, ham_gen);
#ifdef MACIS_ENABLE_MPI
  else
    asci_pairs = asci_contributions_constraint(
        asci_settings, cdets_begin, cdets_end, E_ASCI, C, norb, T_pq, G_red,
        V_red, G_pqrs, V_pqrs, ham_gen MACIS_MPI_CODE(, comm));
#endif
  auto pairs_en = clock_type::now();

  {
#ifdef MACIS_ENABLE_MPI
    size_t npairs = allreduce(asci_pairs.size(), MPI_SUM, comm);
#else
    size_t npairs = asci_pairs.size();
#endif
    logger->info("  * ASCI Kept {} Pairs", npairs);

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
  // Finalize scores
  for(auto& x : asci_pairs) x.rv = -std::abs(x.rv);

  // Insert all dets with their coefficients as seeds
  for(size_t i = 0; i < ncdets; ++i) {
    auto state = *(cdets_begin + i);
    asci_pairs.push_back({state, std::abs(C[i])});
  }

  // Check duplicates (which correspond to the initial truncation),
  // and keep only the duplicate with positive coefficient.
  keep_only_largest_copy_asci_pairs(asci_pairs);

  asci_pairs.erase(std::partition(asci_pairs.begin(), asci_pairs.end(),
                                  [](const auto& p) { return p.rv < 0.0; }),
                   asci_pairs.end());

  // Only do top-K on (ndets_max - ncdets) b/c CDETS will be added later
  const size_t top_k_elements = ndets_max - ncdets;

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

  // Do Top-K to get the largest determinant contributions
  auto asci_sort_st = clock_type::now();
  if(world_size > 1 or asci_pairs.size() > top_k_elements) {
    std::vector<asci_contrib<wfn_t<N>>> topk(top_k_elements);
    if(world_size > 1) {
#ifdef MACIS_ENABLE_MPI
      // Strip scores
      std::vector<double> scores(asci_pairs.size());
      std::transform(asci_pairs.begin(), asci_pairs.end(), scores.begin(),
                     [](const auto& p) { return std::abs(p.rv); });

      // Determine kth-ranked scores
      auto kth_score =
          dist_quickselect(scores.begin(), scores.end(), top_k_elements, comm,
                           std::greater<double>{}, std::equal_to<double>{});

      // Partition local pairs into less / eq batches
      auto [g_begin, e_begin, l_begin, _end] = leg_partition(
          asci_pairs.begin(), asci_pairs.end(), kth_score,
          [=](const auto& p, const auto& s) { return std::abs(p.rv) > s; },
          [=](const auto& p, const auto& s) { return std::abs(p.rv) == s; });

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
