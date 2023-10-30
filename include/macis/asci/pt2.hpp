/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <deque>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/asci/determinant_sort.hpp>

namespace macis {

template <size_t N>
double asci_pt2_constraint(wavefunction_iterator_t<N> cdets_begin,
                           wavefunction_iterator_t<N> cdets_end,
                           const double E_ASCI, const std::vector<double>& C,
                           size_t norb, const double* T_pq, const double* G_red,
                           const double* V_red, const double* G_pqrs,
                           const double* V_pqrs,
                           HamiltonianGenerator<wfn_t<N>>& ham_gen,
                           MPI_Comm comm) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  using wfn_comp = typename wfn_traits::spin_comparator;
  if(!std::is_sorted(cdets_begin, cdets_end, wfn_comp{}))
    throw std::runtime_error("PT2 Only Works with Sorted Wfns");

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
  auto logger = spdlog::get("asci_pt2");
  if(!logger)
    logger = world_rank ? spdlog::null_logger_mt("asci_pt2")
                        : spdlog::stdout_color_mt("asci_pt2");

  const size_t ncdets = std::distance(cdets_begin, cdets_end);

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

  // if(world_rank == 0) {
  //   std::ofstream ofile("uniq_alpha.txt");
  //   for(auto [d, c] : uniq_alpha) {
  //     ofile << to_canonical_string(wfn_traits::from_spin(d,0)) << " " << c <<
  //     std::endl;
  //   }
  // }

  // const auto n_occ_alpha = wfn_traits::count(uniq_alpha_wfn[0]);
  const auto n_occ_alpha = spin_wfn_traits::count(uniq_alpha[0].first);
  const auto n_vir_alpha = norb - n_occ_alpha;
  const auto n_sing_alpha = n_occ_alpha * n_vir_alpha;
  const auto n_doub_alpha = (n_sing_alpha * (n_sing_alpha - norb + 1)) / 4;

  const auto n_occ_beta = cdets_begin->count() - n_occ_alpha;
  const auto n_vir_beta = norb - n_occ_beta;
  const auto n_sing_beta = n_occ_beta * n_vir_beta;
  const auto n_doub_beta = (n_sing_beta * (n_sing_beta - norb + 1)) / 4;

  logger->info("  * NS = {} ND = {}", n_sing_alpha, n_doub_alpha);

  auto gen_c_st = clock_type::now();
  // auto constraints = dist_constraint_general<wfn_t<N>>(
  //     5, norb, n_sing_beta, n_doub_beta, uniq_alpha, comm);
  auto constraints = gen_constraints_general<wfn_t<N>>(
      5, norb, n_sing_beta, n_doub_beta, uniq_alpha, world_size);
  auto gen_c_en = clock_type::now();
  duration_type gen_c_dur = gen_c_en - gen_c_st;
  logger->info("  * GEN_DUR = {:.2e} ms", gen_c_dur.count());

  size_t max_size = 100000000ul;

  double EPT2 = 0.0;
  size_t NPT2 = 0;
  auto pt2_st = clock_type::now();
  std::deque<size_t> print_points(100);
  for(auto i = 0; i < 100; ++i) {
    print_points[i] = constraints.size() * (i / 100.);
  }
  // std::mutex print_barrier;

  const size_t ncon_total = constraints.size();
  duration_type lock_wait_dur(0.0);
  MPI_Win window;
  // MPI_Win_create( &window_count, sizeof(size_t), sizeof(size_t),
  // MPI_INFO_NULL, comm, &window );
  size_t* window_buffer;
  MPI_Win_allocate(sizeof(size_t), sizeof(size_t), MPI_INFO_NULL, comm,
                   &window_buffer, &window);
  if(window == MPI_WIN_NULL) throw std::runtime_error("Window failed");
  MPI_Win_lock_all(MPI_MODE_NOCHECK, window);
  // Process ASCI pair contributions for each constraint
  // #pragma omp parallel reduction(+ : EPT2) reduction(+ : NPT2)
  {
    asci_contrib_container<wfn_t<N>> asci_pairs;
    asci_pairs.reserve(max_size);
    // #pragma omp for
    // for(size_t ic = 0; ic < constraints.size(); ++ic)
    size_t ic = 0;
    while(ic < ncon_total) {
      size_t ntake = 10;
      MPI_Fetch_and_op(&ntake, &ic, MPI_UINT64_T, 0, 0, MPI_SUM, window);
      MPI_Win_flush(0, window);

      // Loop over assigned tasks
      const size_t c_end = std::min(ncon_total, ic + ntake);
      for(; ic < c_end; ++ic) {
        const auto& con = constraints[ic].first;
        // if(ic >= print_points.front()) {
        //   //std::lock_guard<std::mutex> lock(print_barrier);
        //   printf("[rank %d] %.1f  done\n", world_rank,
        //   double(ic)/constraints.size()*100); print_points.pop_front();
        // }
        printf("[rank %d] %lu / %lu\n", world_rank, ic, ncon_total);
        const double h_el_tol = 1e-16;

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
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), T_pq,
                norb, G_red, norb, V_red, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, asci_pairs);

            // AAAA excitations
            generate_constraint_doubles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), G_pqrs,
                norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

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

        }  // Unique Alpha Loop

        double EPT2_local = 0.0;
        // Local S&A for each quad + update EPT2
        {
          auto uit = sort_and_accumulate_asci_pairs(asci_pairs.begin(),
                                                    asci_pairs.end());
          for(auto it = asci_pairs.begin(); it != uit; ++it) {
            // if(std::find(cdets_begin, cdets_end, it->state) == cdets_end)
            if(!std::isinf(it->c_times_matel)) {
              EPT2_local +=
                  (it->c_times_matel * it->c_times_matel) / it->h_diag;
              NPT2++;
            }
          }
          asci_pairs.clear();
        }

        EPT2 += EPT2_local;
      }  // Loc constraint loop
    }    // Constraint Loop
  }
  auto pt2_en = clock_type::now();

  EPT2 = allreduce(EPT2, MPI_SUM, comm);

  double local_pt2_dur = duration_type(pt2_en - pt2_st).count();
  if(world_size > 1) {
    double total_dur = allreduce(local_pt2_dur, MPI_SUM, comm);
    double min_dur = allreduce(local_pt2_dur, MPI_MIN, comm);
    double max_dur = allreduce(local_pt2_dur, MPI_MAX, comm);
    logger->info("* PT2_DUR MIN = {:.2e}, MAX = {:.2e}, AVG = {:.2e} ms",
                 min_dur, max_dur, total_dur / world_size);
  } else {
    logger->info("* PT2_DUR = ${:.2e} ms", local_pt2_dur);
  }
  printf("[rank %d] WAIT_DUR = %.2e\n", world_rank, lock_wait_dur.count());

  NPT2 = allreduce(NPT2, MPI_SUM, comm);
  logger->info("* NPT2 = {}", NPT2);
  MPI_Win_unlock_all(window);
  MPI_Win_free(&window);

  return EPT2;
}
}  // namespace macis
