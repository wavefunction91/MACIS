/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/asci/determinant_contributions.hpp>
#include <macis/asci/determinant_sort.hpp>

namespace macis {


template <size_t N>
double asci_pt2_constraint(
    wavefunction_iterator_t<N> cdets_begin, wavefunction_iterator_t<N> cdets_end,
    const double E_ASCI, const std::vector<double>& C, size_t norb,
    const double* T_pq, const double* G_red, const double* V_red, 
    const double* G_pqrs, const double* V_pqrs, HamiltonianGenerator<wfn_t<N>>& ham_gen,
    MPI_Comm comm) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  using wfn_traits = wavefunction_traits<wfn_t<N>>;

  auto logger = spdlog::get("asci_search");
  const size_t ncdets = std::distance(cdets_begin, cdets_end);
  //std::cout << "NDETS PT = " << ncdets <<  " " << C.size() << std::endl;
  //std::cout << "PT E0    = " << E_ASCI << std::endl;

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
                    const HamiltonianGenerator<wfn_t<N>>& ham_gen) {
      coeff = c;

      // Compute Beta string
      const auto beta_shift = w >> N / 2;
      // Reduce the number of times things shift in inner loop
      beta_string = beta_shift << N / 2;

      // Compute diagonal matrix element
      h_diag = ham_gen.matrix_element(w, w);

      // Compute occ/vir for beta string
      wfn_traits::state_to_occ_vir(norb, beta_shift, occ_beta, vir_beta);

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
    wfn_traits::state_to_occ_vir(norb, wfn_a, occ_alpha, vir_alpha);
    for(auto j = 0; j < ncdets; ++j) {
      const auto w = *(cdets_begin + j);
      if((w & full_mask<N / 2, N>()) == wfn_a) {
        uad[i].bcd.emplace_back(C[j], norb, occ_alpha, w, ham_gen);
      }
    }
  }

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  const auto n_occ_alpha = wfn_traits::count(uniq_alpha_wfn[0]);
  const auto n_vir_alpha = norb - n_occ_alpha;
  const auto n_sing_alpha = n_occ_alpha * n_vir_alpha;
  const auto n_doub_alpha = (n_sing_alpha * (n_sing_alpha - norb + 1)) / 4;

  logger->info("  * NS = {} ND = {}", n_sing_alpha, n_doub_alpha);

  auto gen_c_st = clock_type::now();
  auto constraints =
      dist_constraint_general(0, norb,
                              n_sing_alpha, n_doub_alpha, uniq_alpha_wfn, comm);
  auto gen_c_en = clock_type::now();
  duration_type gen_c_dur = gen_c_en - gen_c_st;
  logger->info("  * GEN_DUR = {:.2e} ms", gen_c_dur.count());

  size_t max_size = std::min(100000000ul,
                             ncdets * (2 * n_sing_alpha +  // AA + BB
                                       2 * n_doub_alpha +  // AAAA + BBBB
                                       n_sing_alpha * n_sing_alpha  // AABB
                                       ));
  double EPT2 = 0.0;

  // Process ASCI pair contributions for each constraint
  #pragma omp parallel 
  {
  asci_contrib_container<wfn_t<N>> asci_pairs;
  asci_pairs.reserve(max_size);
  #pragma omp for reduction(+:EPT2)
  for(size_t ic = 0; ic < constraints.size(); ++ic) {
    const auto& con = constraints[ic];
    //std::cout << std::distance(&constraints[0], &con) << "/" << constraints.size() << std::endl;
    const double h_el_tol = 1e-16;
    const auto& [C, B, C_min] = con;
    wfn_constraint<N/2> alpha_con{ wfn_traits::alpha_string(C), wfn_traits::alpha_string(B), C_min};

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
            coeff, det|beta, alpha_con, occ_alpha, occ_beta,
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
            coeff, det|beta, alpha_con, occ_alpha, occ_beta,
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
            coeff, det|beta, alpha_con, occ_alpha, occ_beta, vir_beta,
            orb_ens_alpha.data(), orb_ens_beta.data(), V_pqrs, norb, h_el_tol,
            h_diag, E_ASCI, ham_gen, asci_pairs);
      }

      // If the alpha determinant satisfies the constraint,
      // append BB and BBBB excitations
      if(satisfies_constraint(det, con)) {
        for(const auto& bcd : uad[i_alpha].bcd) {
          const auto& beta = bcd.beta_string;
          const auto& coeff = bcd.coeff;
          const auto& h_diag = bcd.h_diag;
          const auto& occ_beta = bcd.occ_beta;
          const auto& vir_beta = bcd.vir_beta;
          const auto& eps_beta = bcd.orb_ens_beta;

          const auto state = det | beta;
          const auto state_alpha = wfn_traits::alpha_string(state);
          const auto state_beta  = wfn_traits::beta_string(beta);
          // BB Excitations
          append_singles_asci_contributions<Spin::Beta>(
              coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
              eps_beta.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol,
              h_diag, E_ASCI, ham_gen, asci_pairs);

          // BBBB Excitations
          append_ss_doubles_asci_contributions<Spin::Beta>(
              coeff, state, state_beta, state_alpha, occ_beta, vir_beta, occ_alpha,
              eps_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
              asci_pairs);

          // No excition - to remove for PT2
          asci_pairs.push_back({state, std::numeric_limits<double>::infinity(), 1.0});
        }  // Beta Loop
      }    // Triplet Check

    }    // Unique Alpha Loop

    double EPT2_local = 0.0;
    // Local S&A for each quad + update EPT2
    {
      auto uit = sort_and_accumulate_asci_pairs(asci_pairs.begin(), asci_pairs.end());
      for(auto it = asci_pairs.begin(); it != uit; ++it) {
        //if(std::find(cdets_begin, cdets_end, it->state) == cdets_end)
        if(!std::isinf(it->c_times_matel))
        EPT2_local += (it->c_times_matel * it->c_times_matel) / it->h_diag;
      } 
      asci_pairs.clear();
    }

    EPT2 += EPT2_local;
  }  // Constraint Loop
  }

  EPT2 = allreduce(EPT2, MPI_SUM, comm);

  return EPT2;
}
}
