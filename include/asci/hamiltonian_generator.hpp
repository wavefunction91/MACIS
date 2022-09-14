#pragma once
#include <asci/bitset_operations.hpp>
#include <asci/sd_operations.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>

namespace asci {

template <size_t N = 64>
class HamiltonianGenerator {

  static_assert(N % 2 == 0, "N Must Be Even");

public:

  constexpr static size_t nbits = N;

  using full_det_t = std::bitset<N>;
  using spin_det_t = std::bitset<N/2>;

  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double,index_t>;

  using full_det_iterator = typename std::vector<full_det_t>::iterator;

public:

  inline auto gen_alpha_mask() { return asci::full_mask<N/2,N>(); }
  inline auto gen_beta_mask()  { return gen_alpha_mask() << (N/2); }

  inline spin_det_t alpha_string( full_det_t str ) {
    return asci::truncate_bitset<N/2>(str);
  }
  inline spin_det_t beta_string( full_det_t str ) {
    return asci::truncate_bitset<N/2>(str >> (N/2));
  }

#if 0
  template <size_t M>
  inline uint32_t first_occ_flipped( std::bitset<M> state, std::bitset<M> ex ) {
    return asci::ffs( state & ex ) - 1u;
  }

  template <size_t M>
  inline double single_ex_sign( std::bitset<M> state, unsigned p, unsigned q ) {
    std::bitset<M> mask = 0ul;

    if( p > q ) {
      mask = state & ( asci::full_mask<M>(p)^ asci::full_mask<M>(q+1) );
    } else {
      mask = state & ( asci::full_mask<M>(q)^ asci::full_mask<M>(p+1) );
    }
    return (mask.count() % 2) ? -1. : 1.;

  }
#endif

  size_t norb_;
  size_t norb2_;
  size_t norb3_;
  double*       V_pqrs_;
  double*       T_pq_;
  std::vector<double> G_pqrs_; // G(i,j,k,l)   = (ij|kl) - (il|kj)
  std::vector<double> G_red_;  // G_red(i,j,k) = G(i,j,k,k)
  std::vector<double> V_red_;  // V_red(i,j,k) = (ij|kk)
  std::vector<double> G2_red_; // G2_red(i,j)  = 0.5 * G(i,i,j,j)
  std::vector<double> V2_red_; // V2_red(i,j)  = (ii|jj)

  virtual sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
    full_det_iterator, full_det_iterator, full_det_iterator, full_det_iterator,
    double ) = 0;
  virtual sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
    full_det_iterator, full_det_iterator, full_det_iterator, full_det_iterator,
    double ) = 0;

public:

  HamiltonianGenerator( size_t no, double* V, double* T ); 
  virtual ~HamiltonianGenerator() noexcept = default;


  void generate_integral_intermediates(size_t no, const double* V); 


  double matrix_element_4( spin_det_t bra, spin_det_t ket, spin_det_t ex ) const; 
  double matrix_element_22( spin_det_t bra_alpha, spin_det_t ket_alpha,
    spin_det_t ex_alpha, spin_det_t bra_beta, spin_det_t ket_beta,
    spin_det_t ex_beta ) const ;

  double matrix_element_2( spin_det_t bra, spin_det_t ket, spin_det_t ex,
    const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta ) const ;

  double matrix_element_diag( const std::vector<uint32_t>& occ_alpha,
    const std::vector<uint32_t>& occ_beta ) const ;

  double matrix_element( spin_det_t bra_alpha, spin_det_t ket_alpha,
    spin_det_t ex_alpha, spin_det_t bra_beta, spin_det_t ket_beta,
    spin_det_t ex_beta, const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta ) const ;

  double single_orbital_en( uint32_t orb, const std::vector<uint32_t>& ss_occ,
    const std::vector<uint32_t>& os_occ ) const;

  double fast_diag_single( const std::vector<uint32_t>& ss_occ, 
    const std::vector<uint32_t>& os_occ, uint32_t orb_hol, uint32_t orb_par,
    double orig_det_Hii ) const;

  double fast_diag_ss_double( const std::vector<uint32_t>& ss_occ, 
    const std::vector<uint32_t>& os_occ, uint32_t orb_hol1, uint32_t orb_hol2,
    uint32_t orb_par1, uint32_t orb_par2, double orig_det_Hii ) const;

  double fast_diag_os_double( const std::vector<uint32_t>& up_occ, 
    const std::vector<uint32_t>& do_occ, uint32_t orb_holu, uint32_t orb_hold,
    uint32_t orb_paru, uint32_t orb_pard, double orig_det_Hii ) const;

  double matrix_element( full_det_t bra, full_det_t ket ) const;

  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block(
    full_det_iterator bra_begin,
    full_det_iterator bra_end,
    full_det_iterator ket_begin,
    full_det_iterator ket_end,
    double H_thresh ) {

    if constexpr ( std::is_same_v<index_t, int32_t> )
      return make_csr_hamiltonian_block_32bit_(bra_begin, bra_end,
        ket_begin, ket_end, H_thresh);
    else if constexpr ( std::is_same_v<index_t, int64_t> )
      return make_csr_hamiltonian_block_64bit_(bra_begin, bra_end,
        ket_begin, ket_end, H_thresh);
    else {
      throw std::runtime_error("Unsupported index_t");
      abort();
    }

  }


  void rdm_contributions_4( spin_det_t bra, spin_det_t ket, spin_det_t ex,
    double val, double* trdm );
  void rdm_contributions_22( spin_det_t bra_alpha, spin_det_t ket_alpha,
    spin_det_t ex_alpha, spin_det_t bra_beta, spin_det_t ket_beta,
    spin_det_t ex_beta, double val, double* trdm );
  void rdm_contributions_2( spin_det_t bra, spin_det_t ket, spin_det_t ex,
    const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta,
    double val, double* ordm, double* trdm);
  void rdm_contributions_diag( const std::vector<uint32_t>& occ_alpha,
    const std::vector<uint32_t>& occ_beta, double val, double* ordm, 
    double* trdm );
  
  void rdm_contributions( spin_det_t bra_alpha, spin_det_t ket_alpha,
    spin_det_t ex_alpha, spin_det_t bra_beta, spin_det_t ket_beta,
    spin_det_t ex_beta, const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta, double val, double* ordm,
    double* trdm);


  virtual void form_rdms( full_det_iterator, full_det_iterator, full_det_iterator,
    full_det_iterator, double* C, double* ordm, double* trdm ) = 0;


  void rotate_hamiltonian_ordm( const double* ordm ); 

  virtual void SetJustSingles( bool /*_js*/ ) {}
  virtual bool GetJustSingles( ){ return false; }
  virtual size_t GetNimp() const { return N/2; } 
};



} // namespace asci 

// Implementation
#include <asci/hamiltonian_generator/impl.hpp>
