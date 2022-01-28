#pragma once
#include "dbwy/hamiltonian_generator.hpp"

namespace dbwy {

template <size_t N>
class ResidueArraysHamiltonianGenerator : public HamiltonianGenerator<N> {

public:

  using base_type = HamiltonianGenerator<N>;
  using full_det_t        = typename base_type::full_det_t;
  using spin_det_t        = typename base_type::spin_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;

  template <typename index_t>
  using sparse_matrix_type = typename base_type::sparse_matrix_type<index_t>;

protected:

  struct residue_state_pair {
    size_t state_idx;
    full_det_t residue;
  };

  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block_(
    full_det_iterator bra_begin,
    full_det_iterator bra_end,
    full_det_iterator ket_begin,
    full_det_iterator ket_end,
    double H_thresh ) {

    
    const size_t nbra_dets = std::distance( bra_begin, bra_end );
    const size_t nket_dets = std::distance( ket_begin, ket_end );

    std::vector< index_t > colind, rowptr( nbra_dets + 1 );
    std::vector< double  > nzval;

    colind.reserve( nbra_dets * nbra_dets * 0.005 );
    nzval .reserve( nbra_dets * nbra_dets * 0.005 );

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    const size_t nelec = bra_begin->count();
    const size_t res_size = 
      detail::factorial(nelec) / ( detail::factorial(nelec-2) * 2 );

    std::vector<full_det_t> residues;
    std::vector<residue_state_pair> res_state_pairs;
    residues.reserve( res_size );
    res_state_pairs.reserve( nbra_dets * res_size );

    // Generate Residues for Bra determinants
    for( size_t i = 0; i < nbra_dets; ++i ) {
      residues.clear();
      const auto state = *(bra_begin + i);
      generate_residues( state, residues );
      for( auto&& res : residues ) 
        res_state_pairs.emplace_back(residue_state_pair{i, res});
    }

    std::sort( res_state_pairs.begin(), res_state_pairs.end(),
      [](auto x, auto y) -> bool {
        auto res_x = x.residue;
        auto res_y = y.residue;
        return bitset_less(res_x,res_y);
      } );

    
    //std::vector<std::pair<size_t,size_t>> edges;
    std::set<std::pair<size_t,size_t>> edges;

    auto begin = res_state_pairs.begin();
    auto end   = res_state_pairs.end();
    size_t idx = 0;
    while( begin != end ) {
      auto ref_res = begin->residue;
      decltype(begin) part_end;
      for( part_end = begin; part_end != end; ++part_end ) 
      if( part_end->residue != ref_res ) break;

      for( auto it = begin; it != part_end; ++it ) 
      for( auto jt = it; jt != part_end; ++jt ) {
        edges.emplace(it->state_idx, jt->state_idx);
      }
      begin = part_end;
    }

    std::cout << edges.size() << std::endl;

    rowptr[0] = 0;

    throw "";
    return sparse_matrix_type<index_t>( nbra_dets, nket_dets, std::move(rowptr),
      std::move(colind), std::move(nzval) );

  }

  sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
    full_det_iterator bra_begin, full_det_iterator bra_end, 
    full_det_iterator ket_begin, full_det_iterator ket_end,
    double H_thresh) override {

    std::cout << "in 32 bit" << std::endl;
    return make_csr_hamiltonian_block_<int32_t>(bra_begin, bra_end,
      ket_begin, ket_end, H_thresh );

  }

  sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
    full_det_iterator bra_begin, full_det_iterator bra_end, 
    full_det_iterator ket_begin, full_det_iterator ket_end,
    double H_thresh) override {

    return make_csr_hamiltonian_block_<int64_t>(bra_begin, bra_end,
      ket_begin, ket_end, H_thresh );

  }
  
public:

  template <typename... Args>
  ResidueArraysHamiltonianGenerator(Args&&... args) :
    HamiltonianGenerator<N>(std::forward<Args>(args)...) { }

};

} // namespace dbwy
