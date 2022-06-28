#pragma once
#include "dbwy/hamiltonian_generator.hpp"
#include "dbwy/sd_operations.hpp"

namespace dbwy {

template< size_t N >
struct det_pos
{
  public:
    bitset<N> det;
    uint32_t  id;
};

template< size_t N >
bool operator<( const det_pos<N>& a, const det_pos<N>& b )
{ return dbwy::bitset_less( a.det, b.det ); }

template <size_t N>
class SDBuildHamiltonianGenerator : public HamiltonianGenerator<N> {

public:

  using base_type = HamiltonianGenerator<N>;
  using full_det_t        = typename base_type::full_det_t;
  using spin_det_t        = typename base_type::spin_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;

  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double,index_t>;

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

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    std::set<det_pos<N> > kets;
    for( full_det_iterator it = ket_begin; it != ket_end; it++ )
    { 
      det_pos<N> a;
      a.det = *it;
      a.id  = std::distance(ket_begin, it);
      kets.insert( a );
    }

    rowptr[0] = 0;

    // Loop over bra determinants
    for( size_t i = 0; i < nbra_dets; ++i ) {
      ///*if( (i%1000) == 0 )*/ std::cout << i << ", " << rowptr[i] << std::endl;
      const auto bra = *(bra_begin + i);

      size_t nrow = 0;
      if( bra.count() ) {

        // Separate out into alpha/beta components 
        spin_det_t bra_alpha = truncate_bitset<N/2>(bra);
        spin_det_t bra_beta  = truncate_bitset<N/2>(bra >> (N/2));
        
        // Get occupied indices
        bits_to_indices( bra_alpha, bra_occ_alpha );
        bits_to_indices( bra_beta, bra_occ_beta );
        
        // Get singles and doubles
        std::vector<full_det_t> excs, doubles;
        if( just_singles )
          generate_singles_spin( this->norb_, bra, excs );
        else
        {
          generate_singles_doubles_spin( this->norb_, bra, excs, doubles );
          excs.insert( excs.end(), doubles.begin(), doubles.end() );
        }

        // Diagonal term
        full_det_t ex_diag   = bra ^ bra;
        spin_det_t exd_alpha = truncate_bitset<N/2>( ex_diag );
        spin_det_t exd_beta  = truncate_bitset<N/2>( ex_diag >> (N/2) );

        // Compute Matrix Element
        const auto h_eld = this->matrix_element( bra_alpha, bra_alpha,
          exd_alpha, bra_beta, bra_beta, exd_beta, bra_occ_alpha,
          bra_occ_beta );

        if( std::abs(h_eld) > H_thresh ) {
          nrow++;
          colind.emplace_back(i);
          nzval.emplace_back(h_eld);
        }

        // Loop over ket determinants
        for( const auto pos_ket : excs ) {
          det_pos<N> pos_ket2; pos_ket2.det = pos_ket; pos_ket2.id = 0;
          auto it = kets.find( pos_ket2 );
          if( it != kets.end() ) {
            int j = it->id;
            spin_det_t ket_alpha = truncate_bitset<N/2>( pos_ket);
            spin_det_t ket_beta  = truncate_bitset<N/2>( pos_ket >> (N/2));

            full_det_t ex_total = bra ^ pos_ket;
            if( ex_total.count() <= 4 ) {
            
              spin_det_t ex_alpha = truncate_bitset<N/2>( ex_total );
              spin_det_t ex_beta  = truncate_bitset<N/2>( ex_total >> (N/2) );

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

    colind.shrink_to_fit();
    nzval.shrink_to_fit();

    return sparse_matrix_type<index_t>( nbra_dets, nket_dets, std::move(rowptr),
      std::move(colind), std::move(nzval) );

  }

  sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
    full_det_iterator bra_begin, full_det_iterator bra_end, 
    full_det_iterator ket_begin, full_det_iterator ket_end,
    double H_thresh) override {

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

  void form_rdms(
    full_det_iterator bra_begin,
    full_det_iterator bra_end,
    full_det_iterator ket_begin,
    full_det_iterator ket_end,
    double *C, double* ordm, double* trdm ) override {

    
    const size_t nbra_dets = std::distance( bra_begin, bra_end );
    const size_t nket_dets = std::distance( ket_begin, ket_end );

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    std::set<det_pos<N> > kets;
    for( full_det_iterator it = ket_begin; it != ket_end; it++ )
    { 
      det_pos<N> a;
      a.det = *it;
      a.id  = std::distance(ket_begin, it);
      kets.insert( a );
    }


    // Loop over bra determinants
    for( size_t i = 0; i < nbra_dets; ++i ) {
      const auto bra = *(bra_begin + i);

      if( bra.count() ) {

        // Separate out into alpha/beta components 
        spin_det_t bra_alpha = truncate_bitset<N/2>(bra);
        spin_det_t bra_beta  = truncate_bitset<N/2>(bra >> (N/2));
        
        // Get occupied indices
        bits_to_indices( bra_alpha, bra_occ_alpha );
        bits_to_indices( bra_beta, bra_occ_beta );
        
        // Get singles and doubles
        std::vector<full_det_t> excs, doubles;
        generate_singles_doubles_spin( this->norb_, bra, excs, doubles );
        excs.insert( excs.end(), doubles.begin(), doubles.end() );

        // Diagonal term
        full_det_t ex_diag   = bra ^ bra;
        spin_det_t exd_alpha = truncate_bitset<N/2>( ex_diag );
        spin_det_t exd_beta  = truncate_bitset<N/2>( ex_diag >> (N/2) );

        // Compute Matrix Element
        this->rdm_contributions( bra_alpha, bra_alpha,
          exd_alpha, bra_beta, bra_beta, exd_beta, bra_occ_alpha,
          bra_occ_beta, C[i] * C[i], ordm, trdm );

        // Loop over excitations
        for( const auto pos_ket : excs ) {
          det_pos<N> pos_ket2; pos_ket2.det = pos_ket; pos_ket2.id = 0;
          auto it = kets.find( pos_ket2 );
          if( it != kets.end() ) {
            int j = it->id;
            spin_det_t ket_alpha = truncate_bitset<N/2>( pos_ket );
            spin_det_t ket_beta  = truncate_bitset<N/2>( pos_ket >> (N/2) );

            full_det_t ex_total = bra ^ pos_ket;
            if( ex_total.count() <= 4 ) {
            
              spin_det_t ex_alpha = truncate_bitset<N/2>( ex_total );
              spin_det_t ex_beta  = truncate_bitset<N/2>( ex_total >> (N/2) );

              const double val = C[i] * C[j];

              // Compute Matrix Element
              this->rdm_contributions( bra_alpha, ket_alpha,
                ex_alpha, bra_beta, ket_beta, ex_beta, bra_occ_alpha,
                bra_occ_beta, val, ordm, trdm );

            } // Possible non-zero connection (Hamming distance)
            
          } // Non-zero ket determinant
        } // Loop over ket determinants
      
      } // Non-zero bra determinant
    } // Loop over bra determinants 

  }
  
  void form_rdms(
    full_det_iterator bra_begin,
    full_det_iterator bra_end,
    full_det_iterator ket_begin,
    full_det_iterator ket_end,
    double *C, double* ordm ) override {

    
    const size_t nbra_dets = std::distance( bra_begin, bra_end );
    const size_t nket_dets = std::distance( ket_begin, ket_end );

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    std::set<det_pos<N> > kets;
    for( full_det_iterator it = ket_begin; it != ket_end; it++ )
    { 
      det_pos<N> a;
      a.det = *it;
      a.id  = std::distance(ket_begin, it);
      kets.insert( a );
    }

    // Loop over bra determinants
    for( size_t i = 0; i < nbra_dets; ++i ) {
      const auto bra = *(bra_begin + i);

      if( bra.count() ) {

        // Separate out into alpha/beta components 
        spin_det_t bra_alpha = truncate_bitset<N/2>(bra);
        spin_det_t bra_beta  = truncate_bitset<N/2>(bra >> (N/2));
        
        // Get occupied indices
        bits_to_indices( bra_alpha, bra_occ_alpha );
        bits_to_indices( bra_beta, bra_occ_beta );
        
        // Get singles and doubles
        std::vector<full_det_t> singles;
        generate_singles_spin( this->norb_, bra, singles );

        // Diagonal term
        full_det_t ex_diag   = bra ^ bra;
        spin_det_t exd_alpha = truncate_bitset<N/2>( ex_diag );
        spin_det_t exd_beta  = truncate_bitset<N/2>( ex_diag >> (N/2) );

        // Compute Matrix Element
        this->rdm_contributions( bra_alpha, bra_alpha,
          exd_alpha, bra_beta, bra_beta, exd_beta, bra_occ_alpha,
          bra_occ_beta, C[i] * C[i], ordm );

        // Loop over ket determinants
        for( const auto pos_ket : singles ) {
          det_pos<N> pos_ket2; pos_ket2.det = pos_ket; pos_ket2.id = 0;
          auto it = kets.find( pos_ket2 );
          if( it != kets.end() ) {
            int j = it->id;
            spin_det_t ket_alpha = truncate_bitset<N/2>( pos_ket );
            spin_det_t ket_beta  = truncate_bitset<N/2>( pos_ket >> (N/2) );

            full_det_t ex_total = bra ^ pos_ket;
            if( ex_total.count() <= 2 ) {
            
              spin_det_t ex_alpha = truncate_bitset<N/2>( ex_total );
              spin_det_t ex_beta  = truncate_bitset<N/2>( ex_total >> (N/2) );

              const double val = C[i] * C[j];

              // Compute Matrix Element
              this->rdm_contributions( bra_alpha, ket_alpha,
                ex_alpha, bra_beta, ket_beta, ex_beta, bra_occ_alpha,
                bra_occ_beta, val, ordm );

            } // Possible non-zero connection (Hamming distance)
            
          } // Non-zero ket determinant
        } // Loop over ket determinants
      
      } // Non-zero bra determinant
    } // Loop over bra determinants 

  }
  
public:

  bool just_singles;

  template <typename... Args>
  SDBuildHamiltonianGenerator(Args&&... args) :
    HamiltonianGenerator<N>(std::forward<Args>(args)...), just_singles(false) { }

  void SetJustSingles( bool _js ) override{ just_singles = _js; }

};

} // namespace dbwy
