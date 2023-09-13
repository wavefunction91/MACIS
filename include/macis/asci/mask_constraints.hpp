/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>
#include <variant>

#include <macis/asci/alpha_constraint.hpp>

namespace macis {


template <size_t N, typename ConType>
bool satisfies_constraint(wfn_t<N> det, ConType C) {
  return C.satisfies_constraint(det);
}

template <size_t N, typename ConType>
auto generate_constraint_single_excitations(wfn_t<N> det, ConType constraint) {
  using spin_wfn_traits = typename ConType::spin_wfn_traits;
  const auto C = constraint.C(); const auto B = constraint.B();

  // need to have at most one different from the constraint
  if(constraint.overlap(det) < (constraint.count()-1))
    return std::make_pair(wfn_t<N>(0), wfn_t<N>(0));

  auto o = det ^ C;
  auto v = (~det) & B;

  if((o & C).count() == 1) {  
    v = o & C;
    o ^= v;
  }

  if((o & ~B).count() > 1) return std::make_pair(wfn_t<N>(0), wfn_t<N>(0));

  if((o & ~B).count() == 1) o &= ~B;

  return std::make_pair(o, v);
}

template <size_t N, typename ConType>
auto generate_constraint_double_excitations(wfn_t<N> det, ConType constraint) {
  const auto C = constraint.C(); const auto B = constraint.B();
  // Occ/Vir pairs to generate excitations
  std::vector<wfn_t<N>> O, V;

  if((det & C) == 0) return std::make_tuple(O, V);

  auto o = det ^ C;
  auto v = (~det) & B;

  if((o & C).count() >= 3) return std::make_tuple(O, V);

  // Generate Virtual Pairs
  if((o & C).count() == 2) {
    v = o & C;
    o ^= v;
  }

  const auto virt_ind = bits_to_indices(v);
  const auto o_and_t = o & C;
  switch((o & C).count()) {
    case 1:
      for(auto a : virt_ind) {
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
  if(o_and_not_b.count() > 2) return std::make_tuple(O, V);

  switch(o_and_not_b.count()) {
    case 1:
      for(auto i : bits_to_indices(o & B)) {
        O.emplace_back(o_and_not_b).flip(i);
      }
      break;
    default:
      if(o_and_not_b.count() == 2) o = o_and_not_b;
      generate_pairs(bits_to_indices(o), O);
      break;
  }

  return std::make_tuple(O, V);
}

template <size_t N, typename ConType>
void generate_constraint_singles(wfn_t<N> det, ConType constraint, 
                                 std::vector<wfn_t<N>>& t_singles) {
  auto [o, v] = generate_constraint_single_excitations(det, constraint);
  const auto oc = o.count();
  const auto vc = v.count();
  if(!oc or !vc) return;

  t_singles.clear();
  t_singles.reserve(oc * vc);
  const auto occ = bits_to_indices(o);
  const auto vir = bits_to_indices(v);
  for(auto i : occ) {
    auto temp = det;
    temp.flip(i);
    for(auto a : vir) t_singles.emplace_back(temp).flip(a);
  }
}

template <typename... Args>
unsigned count_constraint_singles(Args&&... args) {
  auto [o, v] =
      generate_constraint_single_excitations(std::forward<Args>(args)...);
  return o.count() * v.count();
}

template <size_t N, typename ConType >
void generate_constraint_doubles(wfn_t<N> det, ConType constraint, 
                                 std::vector<wfn_t<N>>& t_doubles) {
  auto [O, V] = generate_constraint_double_excitations(det, constraint);

  t_doubles.clear();
  for(auto ij : O) {
    const auto temp = det ^ ij;
    for(auto ab : V) {
      t_doubles.emplace_back(temp | ab);
    }
  }
}

/**
 *  @param[in]  det Input root determinant
 *  @param[in]  T   Triplet constraint mask
 *  @param[in]  B   B mask (?)
 */
template <size_t N, typename ConType>
unsigned count_constraint_doubles(wfn_t<N> det, ConType constraint) {
  const auto C = constraint.C(); const auto B = constraint.B();
  if((det & C) == 0) return 0;

  auto o = det ^ C;
  auto v = (~det) & B;

  if((o & C).count() >= 3) return 0;

  // Generate Virtual Pairs
  if((o & C).count() == 2) {
    v = o & C;
    o ^= v;
  }

  unsigned nv_pairs = v.count();
  const auto o_and_t = o & C;
  switch((o & C).count()) {
    case 1:
      o ^= o_and_t;
      break;
    default:
      nv_pairs = (nv_pairs * (nv_pairs - 1)) / 2;
      break;
  }

  // Generate Occupied Pairs
  const auto o_and_not_b = o & ~B;
  if(o_and_not_b.count() > 2) return 0;

  unsigned no_pairs = 0;
  switch(o_and_not_b.count()) {
    case 1:
      no_pairs = (o & B).count();
      break;
    default:
      if(o_and_not_b.count() == 2) o = o_and_not_b;
      no_pairs = o.count();
      no_pairs = (no_pairs * (no_pairs - 1)) / 2;
      break;
  }

  return no_pairs * nv_pairs;
}

template <size_t N, typename ConType>
size_t constraint_histogram(wfn_t<N> det, size_t n_os_singles,
                            size_t n_os_doubles, ConType constraint){ 
  auto ns = count_constraint_singles(det, constraint);
  auto nd = count_constraint_doubles(det, constraint);

  size_t ndet = 0;
  ndet += ns;                 // AA
  ndet += nd;                 // AAAA
  ndet += ns * n_os_singles;  // AABB
  if(satisfies_constraint(det, constraint)) {
    ndet += n_os_singles + n_os_doubles + 1;  // BB + BBBB + No Excitations
  }

  return ndet;
}

template <typename WfnType, typename ConType>
void generate_constraint_singles_contributions_ss(
    double coeff, WfnType det, ConType constraint,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& occ_othr, const double* eps,
    const double* T_pq, const size_t LDT, const double* G_kpq, const size_t LDG,
    const double* V_kpq, const size_t LDV, double h_el_tol, double root_diag,
    double E0, HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  auto [o, v] = generate_constraint_single_excitations(wfn_traits::alpha_string(det), constraint);
  const auto no = o.count();
  const auto nv = v.count();
  if(!no or !nv) return;

  const size_t LDG2 = LDG * LDG;
  const size_t LDV2 = LDV * LDV;
  for(int ii = 0; ii < no; ++ii) {
    const auto i = fls(o);
    o.flip(i);  // Disable "i"-bit so it's not used in FLS next iteration
    auto v_cpy = v;
    for(int aa = 0; aa < nv; ++aa) {
      const auto a = fls(v_cpy);
      v_cpy.flip(a);  // Disable "a"-bit so it's not used in FLS next iteration

      double h_el = T_pq[a + i * LDT];
      const double* G_ov = G_kpq + a * LDG + i * LDG2;
      const double* V_ov = V_kpq + a * LDV + i * LDV2;
      for(auto p : occ_same) h_el += G_ov[p];
      for(auto p : occ_othr) h_el += V_ov[p];

      // Early Exit
      if(std::abs(coeff * h_el) < h_el_tol) continue;

      // Calculate Excited Determinant
      auto ex_det = wfn_traits::template single_excitation_no_check<Spin::Alpha>(det, i, a );
      

      // Compute Sign in a Canonical Way
      auto sign = single_excitation_sign(det, a, i);
      h_el *= sign;

      // Compute Fast Diagonal Matrix Element
      auto h_diag = ham_gen.fast_diag_single(eps[i], eps[a], i, a, root_diag);
      // h_el /= (E0 - h_diag);

      asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});
    }
  }
}

template <typename WfnType, typename ConType>
void generate_constraint_doubles_contributions_ss(
    double coeff, WfnType det, ConType constraint,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& occ_othr, const double* eps, const double* G,
    const size_t LDG, double h_el_tol, double root_diag, double E0,
    HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_t<WfnType>>;
  auto [O, V] = generate_constraint_double_excitations(wfn_traits::alpha_string(det), constraint);
  const auto no_pairs = O.size();
  const auto nv_pairs = V.size();
  if(!no_pairs or !nv_pairs) return;

  const size_t LDG2 = LDG * LDG;
  for(int _ij = 0; _ij < no_pairs; ++_ij) {
    const auto ij = O[_ij];
    const auto i = ffs(ij) - 1;
    const auto j = fls(ij);
    const auto G_ij = G + (j + i * LDG2) * LDG;
    const auto ex_ij = wfn_traits::template single_excitation_no_check<Spin::Alpha>(det,i,j); // det ^ ij;
    for(int _ab = 0; _ab < nv_pairs; ++_ab) {
      const auto ab = V[_ab];
      const auto a = ffs(ab) - 1;
      const auto b = fls(ab);

      const auto G_aibj = G_ij[b + a * LDG2];

      // Early Exit
      if(std::abs(coeff * G_aibj) < h_el_tol) continue;

      // Calculate Excited Determinant (spin)
      const auto full_ex_spin = spin_wfn_traits::template single_excitation_no_check<Spin::Alpha>(ij,a,b);   // ij | ab;
      const auto ex_det_spin = wfn_traits::template single_excitation_no_check<Spin::Alpha>(ex_ij,a,b); // ex_ij | ab;

      // Compute Sign in a Canonical Way
      auto sign = doubles_sign(wfn_traits::alpha_string(det), wfn_traits::alpha_string(ex_det_spin), full_ex_spin);

      // Calculate Full Excited Determinant
      const auto full_ex = ex_det_spin;// | os_det;

      // Update Sign of Matrix Element
      auto h_el = sign * G_aibj;

      // Evaluate fast diagonal matrix element
      auto h_diag = ham_gen.fast_diag_ss_double(eps[i], eps[j], eps[a], eps[b],
                                                i, j, a, b, root_diag);
      // h_el /= (E0 - h_diag);

      asci_contributions.push_back({full_ex, coeff * h_el, E0 - h_diag});
    }
  }
}

template <typename WfnType, typename ConType>
void generate_constraint_doubles_contributions_os(
    double coeff, WfnType det, ConType constraint,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& occ_othr,
    const std::vector<uint32_t>& vir_othr, const double* eps_same,
    const double* eps_othr, const double* V, const size_t LDV, double h_el_tol,
    double root_diag, double E0, HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  // Generate Single Excitations that Satisfy the Constraint
  auto [o, v] = generate_constraint_single_excitations(wfn_traits::alpha_string(det), constraint);
  const auto no = o.count();
  const auto nv = v.count();
  if(!no or !nv) return;

  const size_t LDV2 = LDV * LDV;
  for(int ii = 0; ii < no; ++ii) {
    const auto i = fls(o);
    o.flip(i);
    auto v_cpy = v;
    for(int aa = 0; aa < nv; ++aa) {
      const auto a = fls(v_cpy);
      v_cpy.flip(a);

      const auto* V_ai = V + a + i * LDV;
      double sign_same = single_excitation_sign(det, a, i);

      for(auto j : occ_othr)
        for(auto b : vir_othr) {
          const auto jb = b + j * LDV;
          const auto V_aibj = V_ai[jb * LDV2];

          // Early Exist
          if(std::abs(coeff * V_aibj) < h_el_tol) continue;

          // double sign_othr = single_excitation_sign( os_det >> (N/2),  b, j
          // );
          double sign_othr =
              single_excitation_sign(wfn_traits::beta_string(det), b, j);
          double sign = sign_same * sign_othr;

          // Compute Excited Determinant
          //auto ex_det = det | os_det;
          //ex_det.flip(i).flip(a).flip(j + N / 2).flip(b + N / 2);
          auto ex_det = wfn_traits::template single_excitation_no_check<Spin::Alpha>(det,i,a);
          ex_det = wfn_traits::template single_excitation_no_check<Spin::Beta>(ex_det,j,b);

          // Finalize Matrix Element
          auto h_el = sign * V_aibj;

          auto h_diag =
              ham_gen.fast_diag_os_double(eps_same[i], eps_othr[j], eps_same[a],
                                          eps_othr[b], i, j, a, b, root_diag);
          // h_el /= (E0 - h_diag);

          asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});
        }  // BJ

    }  // A
  }    // I
}

#if 0
template <size_t N>
auto dist_triplets_all(size_t norb, size_t ns_othr, size_t nd_othr,
  const std::vector<wfn_t<N>>& unique_alpha) {

  wfn_t<N> O = full_mask<N>(norb);

  std::vector< wfn_constraint<N> > triplets; 
  triplets.reserve(norb*norb*norb);
  for(int t_i = 0; t_i < norb; ++t_i)
  for(int t_j = 0; t_j < t_i;  ++t_j)
  for(int t_k = 0; t_k < t_j;  ++t_k) {
    auto constraint = make_triplet<N>(t_i,t_j,t_k);
    const auto [T, B, _] = constraint;

    size_t nw = 0;
    for( const auto& alpha : unique_alpha ) {
       nw += 
         constraint_histogram(alpha, ns_othr, nd_othr, T, O, B );
    }

    if(nw) triplets.emplace_back(constraint);
  }
  

  return triplets;
}


template <size_t N>
auto dist_triplets_histogram(size_t norb, size_t ns_othr, size_t nd_othr,
  const std::vector<wfn_t<N>>& unique_alpha, MPI_Comm comm) {

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
  wfn_t<N> O = full_mask<N>(norb);

  // Generate triplets + heuristic
  std::vector<std::pair<wfn_constraint<N>,size_t>> triplet_sizes; 
  triplet_sizes.reserve(norb*norb*norb);
  for(int t_i = 0; t_i < norb; ++t_i)
  for(int t_j = 0; t_j < t_i;  ++t_j)
  for(int t_k = 0; t_k < t_j;  ++t_k) {
    auto constraint = make_triplet<N>(t_i,t_j,t_k);
    const auto& [T,B,_] = constraint;

    size_t nw = 0;
    for( const auto& alpha : unique_alpha ) {
       nw += 
         constraint_histogram(alpha, ns_othr, nd_othr, T, O, B );
    }
    if(nw) triplet_sizes.emplace_back(constraint, nw);
  }

  // Sort to get optimal bucket partitioning
  std::sort(triplet_sizes.begin(), triplet_sizes.end(),
    [](const auto& a, const auto& b){ return a.second > b.second;} );
  
  // Assign work
  std::vector<size_t> workloads(world_size, 0);
  std::vector< wfn_constraint<N> > triplets; 
  triplets.reserve((norb*norb*norb) / world_size);

  for( auto [trip, nw] : triplet_sizes ) {

    // Get rank with least amount of work
    auto min_rank_it = std::min_element(workloads.begin(), workloads.end());
    int min_rank = std::distance(workloads.begin(), min_rank_it);

    // Assign triplet
    *min_rank_it += nw;
    if(world_rank == min_rank) triplets.emplace_back(trip);
    
  }

  return triplets;
}
#endif


template <size_t N>
auto make_triplet(unsigned i, unsigned j, unsigned k) {
  using wfn_type = wfn_t<N>;
  using wfn_traits = wavefunction_traits<wfn_type>;
  using constraint_type = alpha_constraint<wfn_traits>;
  using string_type     = typename constraint_type::constraint_type;

  string_type C = 0;
  C.flip(i).flip(j).flip(k);
  string_type B = 1;
  B <<= k;
  B = B.to_ullong() - 1;

  return constraint_type(C,B,k);
}

#ifdef MACIS_ENABLE_MPI
template <size_t N>
auto dist_constraint_general(size_t nlevels, size_t norb, size_t ns_othr,
                             size_t nd_othr,
                             const std::vector<wfn_t<N>>& unique_alpha,
                             MPI_Comm comm) {
  using wfn_type = wfn_t<N>;
  using wfn_traits = wavefunction_traits<wfn_type>;
  using constraint_type = alpha_constraint<wfn_traits>;
  using string_type     = typename constraint_type::constraint_type;
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  //wfn_t<N> O = full_mask<N>(norb);

  // Global workloads
  std::vector<size_t> workloads(world_size, 0);

  // Generate triplets + heuristic
  std::vector<std::pair<constraint_type, size_t>> constraint_sizes;
  constraint_sizes.reserve(norb * norb * norb);
  size_t total_work = 0;
  for(int t_i = 0; t_i < norb; ++t_i)
    for(int t_j = 0; t_j < t_i; ++t_j)
      for(int t_k = 0; t_k < t_j; ++t_k) {
        auto constraint = make_triplet<N>(t_i, t_j, t_k);

        size_t nw = 0;
        for(const auto& alpha : unique_alpha) {
          nw += constraint_histogram(wfn_traits::alpha_string(alpha), ns_othr, nd_othr, constraint);
        }
        if(nw) constraint_sizes.emplace_back(constraint, nw);
        total_work += nw;
      }

  size_t local_average = (0.6 * total_work) / world_size;

  for(size_t ilevel = 0; ilevel < nlevels; ++ilevel) {
    // Select constraints larger than average to be broken apart
    std::vector<std::pair<constraint_type, size_t>> tps_to_next;
    {
      auto it = std::partition(
          constraint_sizes.begin(), constraint_sizes.end(),
          [=](const auto& a) { return a.second <= local_average; });

      // Remove constraints from full list
      tps_to_next = decltype(tps_to_next)(it, constraint_sizes.end());
      constraint_sizes.erase(it, constraint_sizes.end());
      for(auto [t, s] : tps_to_next) total_work -= s;
    }

    if(!tps_to_next.size()) break;

    // Break apart constraints
    for(auto [c, nw_trip] : tps_to_next) {
      const auto C_min = c.C_min();

      // Loop over possible constraints with one more element
      for(auto q_l = 0; q_l < C_min; ++q_l) {
        // Generate masks / counts
        string_type cn_C = c.C();
        cn_C.flip(q_l);
        string_type cn_B = c.B() >> (C_min - q_l);
        constraint_type c_next(cn_C, cn_B, q_l);

        size_t nw = 0;

        for(const auto& alpha : unique_alpha) {
          nw += constraint_histogram(wfn_traits::alpha_string(alpha), ns_othr, nd_othr, c_next); 
        }
        if(nw) constraint_sizes.emplace_back(c_next, nw);
        total_work += nw;
      }
    }
  }  // Recurse into constraints

  // if(!world_rank) {
  //   const auto ntrip = std::count_if(constraint_sizes.begin(),
  //     constraint_sizes.end(), [](auto &c){ return c.first.C.count() == 3; });
  //   printf("[rank 0] NTRIP = %lu\n", ntrip);
  //   if(nlevels > 0) {
  //     const auto nquad = std::count_if(constraint_sizes.begin(),
  //       constraint_sizes.end(), [](auto &c){ return c.first.C.count() == 4;
  //       });
  //     printf("[rank 0] NQUAD = %lu\n", nquad);
  //   }
  //   if(nlevels > 1) {
  //     const auto nquint = std::count_if(constraint_sizes.begin(),
  //       constraint_sizes.end(), [](auto &c){ return c.first.C.count() == 5;
  //       });
  //     printf("[rank 0] NQINT = %lu\n", nquint);
  //   }
  // }

  // Sort to get optimal bucket partitioning
  std::sort(constraint_sizes.begin(), constraint_sizes.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Assign work
  std::vector<constraint_type> constraints;
  constraints.reserve(constraint_sizes.size() / world_size);

  for(auto [c, nw] : constraint_sizes) {
    // Get rank with least amount of work
    auto min_rank_it = std::min_element(workloads.begin(), workloads.end());
    int min_rank = std::distance(workloads.begin(), min_rank_it);

    // Assign constraint
    *min_rank_it += nw;
    if(world_rank == min_rank) {
      constraints.emplace_back(c);
    }
  }

  // if(world_rank == 0)
  // printf("[rank %2d] AFTER LOCAL WORK = %lu TOTAL WORK = %lu\n", world_rank,
  //   workloads[world_rank], total_work);

  return constraints;
}
#endif

#if 0
template <typename Integral, size_t N>
auto dist_triplets_random(size_t norb, size_t ns_othr, size_t nd_othr,
  const std::vector<wfn_t<N>>& unique_alpha, MPI_Comm comm) {

  auto triplets = dist_triplets_all<Integral>(norb,ns_othr, nd_othr, unique_alpha);
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  std::default_random_engine g(155039);
  std::shuffle(triplets.begin(),triplets.end(),g);

  std::vector< std::tuple<Integral,Integral,Integral> > local_triplets;
  local_triplets.reserve(triplets.size() / world_size);
  for( auto i = 0; i < triplets.size(); ++i) 
  if( i % world_size == world_rank ) {
    local_triplets.emplace_back(triplets[i]);
  }
  triplets = std::move(local_triplets);

  return triplets;
}
#endif

}  // namespace macis
