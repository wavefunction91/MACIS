#pragma once
#include "hamiltonian_generator.hpp"

template <size_t N>
class DoubleLoopHamiltonianGenerator : public HamiltonianGenerator<N> {

public:

  using base_type = HamiltonianGenerator<N>;
  using full_det_t        = typename base_type::full_det_t;
  using spin_det_t        = typename base_type::spin_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;

  template <typename index_t>
  using sparse_matrix_type = typename base_type::sparse_matrix_type<index_t>;

protected:

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

    rowptr[0] = 0;

    // Loop over bra determinants
    for( size_t i = 0; i < nbra_dets; ++i ) {
      const auto bra = *(bra_begin + i);

      size_t nrow = 0;
      if( bra.count() ) {

        // Separate out into alpha/beta components 
        spin_det_t bra_alpha = detail::truncate_bitset<N/2>(bra);
        spin_det_t bra_beta  = detail::truncate_bitset<N/2>(bra >> (N/2));
        
        // Get occupied indices
        detail::bits_to_indices( bra_alpha, bra_occ_alpha );
        detail::bits_to_indices( bra_beta, bra_occ_beta );

        // Loop over ket determinants
        for( size_t j = 0; j < nket_dets; ++j ) {
          const auto ket = *(ket_begin + j);
          if( ket.count() ) {
            spin_det_t ket_alpha = detail::truncate_bitset<N/2>(ket);
            spin_det_t ket_beta  = detail::truncate_bitset<N/2>(ket >> (N/2));

            full_det_t ex_total = bra ^ ket;
            if( ex_total.count() <= 4 ) {
            
              spin_det_t ex_alpha = detail::truncate_bitset<N/2>( ex_total );
              spin_det_t ex_beta  = detail::truncate_bitset<N/2>( ex_total >> (N/2) );

              // Compute Matrix Element
              const auto h_el = this->matrix_element( bra_alpha, ket_alpha,
                ex_alpha, bra_beta, ket_beta, ex_beta, bra_occ_alpha,
                bra_occ_beta );

              if( std::abs(h_el) > H_thresh ) {
                nrow++;
                colind.emplace_back(j);
                nzval.emplace_back(h_el);
              }

            } // Possible non-zero connection (Hamming distance)
            
          } // Non-zero ket determinant
        } // Loop over ket determinants
      
      } // Non-zero bra determinant

      rowptr[i+1] = rowptr[i] + nrow; // Update rowptr
    } // Loop over bra determinants 


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
  DoubleLoopHamiltonianGenerator(Args&&... args) :
    HamiltonianGenerator<N>(std::forward<Args>(args)...) { }

};
