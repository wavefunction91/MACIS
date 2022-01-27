#pragma once

namespace detail {

template <size_t N, size_t M = N>
std::bitset<M> full_mask() {
  static_assert( M >= N, "M < N" );
  std::bitset<M> mask(0ul);
  return (~mask) >> (M-N);
}

template <size_t N>
std::bitset<N> full_mask(size_t i) {
  std::bitset<N> mask(0ul);
  return (~mask) >> (N-i);
}

template <size_t N>
uint32_t ffs( std::bitset<N> str ) {
  if constexpr (N == 64) {
    return ffsll( str.to_ullong() );
  } else if constexpr (N == 32) {
    return ffsl( str.to_ulong() );
  } else {
    uint32_t ind = 0;
    for( ind = 0; ind < N; ++ind )
    if( str[ind] ) return (ind+1);
    return ind;
  }
}

template <size_t N>
void bits_to_indices( std::bitset<N> bits, std::vector<uint32_t>& indices ) {
  indices.clear();
  for( auto i = 0; i < N; ++i )
  if( bits[i] ) indices.emplace_back(i);
}


template <size_t N, size_t M>
std::bitset<N> truncate_bitset( std::bitset<M> bits ) {
  if constexpr ( M == N ) return bits;
  
  const auto mask = full_mask<N,M>();
  if constexpr ( N == 32 ) {
    return ( bits & mask ).to_ulong();
  } else if constexpr ( N == 64 ) {
    return ( bits & mask ).to_ullong();
  }
}

template <typename Iterator>
using iterator_value_type = 
  std::decay_t< typename std::iterator_traits<Iterator>::value_type >;


}

template <size_t N = 64>
class HamiltonianGenerator {

  static_assert(N%2 == 0, "N Must Be Even");

public:

  using full_det_t = std::bitset<N>;
  using spin_det_t = std::bitset<N/2>;

protected:

  inline auto gen_alpha_mask() { return detail::full_mask<N/2,N>(); }
  inline auto gen_beta_mask()  { return gen_alpha_mask() << (N/2); }

  inline spin_det_t alpha_string( full_det_t str ) {
    return detail::truncate_bitset<N/2>(str);
  }
  inline spin_det_t beta_string( full_det_t str ) {
    return detail::truncate_bitset<N/2>(str >> (N/2));
  }

  inline uint32_t first_occ_flipped( spin_det_t state, spin_det_t ex ) {
    return detail::ffs( state & ex ) - 1u;
  }

  template <size_t M>
  inline double single_ex_sign( std::bitset<M> state, unsigned p, unsigned q ) {
    std::bitset<M> mask = 0ul;

    if( p > q ) {
      mask = state & ( detail::full_mask<M>(p)^ detail::full_mask<M>(q+1) );
    } else {
      mask = state & ( detail::full_mask<M>(q)^ detail::full_mask<M>(p+1) );
    }
    return (mask.count() % 2) ? -1. : 1.;

  }

  size_t norb_;
  size_t norb2_;
  size_t norb3_;
  const double*       V_pqrs_;
  const double*       T_pq_;
  std::vector<double> G_pqrs_; // G(i,j,k,l)   = (ij|kl) - (il|kj)
  std::vector<double> G_red_;  // G_red(i,j,k) = G(i,j,k,k)
  std::vector<double> V_red_;  // V_red(i,j,k) = (ij|kk)
  std::vector<double> G2_red_; // G2_red(i,j)  = 0.5 * G(i,i,j,j)
  std::vector<double> V2_red_; // V2_red(i,j)  = (ii|jj)

  void generate_integral_intermediates_(size_t no, const double* V); 

public:

  HamiltonianGenerator( size_t no, const cmz::ed::intgrls::integrals& ints ) :
    norb_(no), norb2_(no*no), norb3_(no*no*no),
    V_pqrs_(ints.u.data()),
    T_pq_(ints.t.data()) {

    generate_integral_intermediates_(no, V_pqrs_);

  }

  double matrix_element_4( spin_det_t bra, spin_det_t ket, spin_det_t ex ); 
  double matrix_element_22( spin_det_t bra_alpha, spin_det_t ket_alpha,
    spin_det_t ex_alpha, spin_det_t bra_beta, spin_det_t ket_beta,
    spin_det_t ex_beta );

  double matrix_element_2( spin_det_t bra, spin_det_t ket, spin_det_t ex,
    const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta );

  double matrix_element_diag( const std::vector<uint32_t>& occ_alpha,
    const std::vector<uint32_t>& occ_beta );

  double matrix_element( spin_det_t bra_alpha, spin_det_t ket_alpha,
    spin_det_t ex_alpha, spin_det_t bra_beta, spin_det_t ket_beta,
    spin_det_t ex_beta, const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta );


  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double,index_t>;

  using full_det_iterator = std::vector<full_det_t>::iterator;

  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block(
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
        spin_det_t bra_alpha = alpha_string(bra);
        spin_det_t bra_beta  = beta_string(bra);
        
        // Get occupied indices
        detail::bits_to_indices( bra_alpha, bra_occ_alpha );
        detail::bits_to_indices( bra_beta, bra_occ_beta );

        // Loop over ket determinants
        for( size_t j = 0; j < nket_dets; ++j ) {
          const auto ket = *(ket_begin + j);
          if( ket.count() ) {
            spin_det_t ket_alpha = alpha_string(ket);
            spin_det_t ket_beta  = beta_string(ket);

            full_det_t ex_total = bra ^ ket;
            if( ex_total.count() <= 4 ) {
            
              spin_det_t ex_alpha = alpha_string( ex_total );
              spin_det_t ex_beta  = beta_string( ex_total );

              // Compute Matrix Element
              const auto h_el = matrix_element( bra_alpha, ket_alpha,
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


    sparse_matrix_type<index_t> H( nbra_dets, nket_dets, std::move(rowptr),
      std::move(colind), std::move(nzval) );
    return H;

  }

#if 0
  template <typename index_t, typename BraIterator, typename KetIterator>
  std::enable_if_t<
    std::is_same_v< detail::iterator_value_type<BraIterator>, slater_det> and
    std::is_same_v< detail::iterator_value_type<KetIterator>, slater_det>,
    sparse_matrix_type<index_t>
  > make_csr_hamiltonian_block(
    BraIterator bra_begin,
    BraIterator bra_end,
    KetIterator ket_begin,
    KetIterator ket_end,
    double H_thresh ) {

    size_t nbra = std::distance( bra_begin, bra_end );
    size_t nket = std::distance( ket_begin, ket_end );
    size_t norb = bra_begin->GetNorbs();

    const uint64_t sd_alpha_mask = detail::full_mask<64>(norb).to_ullong();
    const uint64_t sd_beta_mask  = sd_alpha_mask << norb;

    auto sd_to_fd = [=]( slater_det _state ) {
      auto state = _state.GetState();
      full_det_t state_alpha = state & sd_alpha_mask;
      full_det_t state_beta  = ((state & sd_beta_mask) >> norb) << N/2;
      return state_alpha | state_beta;
    };

    std::vector< full_det_t > bra_vec( nbra ), ket_vec( nket );
    std::transform( bra_begin, bra_end, bra_vec.begin(), sd_to_fd );
    std::transform( ket_begin, ket_end, ket_vec.begin(), sd_to_fd );

    return make_csr_hamiltonian_block( bra_vec.begin(), bra_vec.end(),
      ket_vec.begin(), ket_vec.end(), H_thresh );

  }
#endif
  
};


template <size_t N>
void HamiltonianGenerator<N>::generate_integral_intermediates_(
  size_t no, const double* V) {

  size_t no2 = no  * no;
  size_t no3 = no2 * no;
  size_t no4 = no3 * no;

  // G(i,j,k,l) = V(i,j,k,l) - V(i,l,k,j)
  G_pqrs_ = std::vector<double>( V, V + no4 );
  for( auto i = 0; i < no; ++i )
  for( auto j = 0; j < no; ++j )
  for( auto k = 0; k < no; ++k )
  for( auto l = 0; l < no; ++l ) {
    G_pqrs_[i + j*no + k*no2 + l*no3] -= V[i + l*no + k*no2 + j*no3];
  }

  // G_red(i,j,k) = G(i,j,k,k) = G(k,k,i,j)
  // V_red(i,j,k) = V(i,j,k,k) = V(k,k,i,j)
  G_red_.resize(no3);
  V_red_.resize(no3);
  for( auto j = 0; j < no; ++j ) 
  for( auto i = 0; i < no; ++i )
  for( auto k = 0; k < no; ++k ) {
    G_red_[k + i*no + j*no2 ] = G_pqrs_[k*(no+1) + i*no2 + j*no3];
    V_red_[k + i*no + j*no2 ] = V      [k*(no+1) + i*no2 + j*no3];
  }

  // G2_red(i,j) = 0.5 * G(i,i,j,j)
  // V2_red(i,j) = V(i,i,j,j)
  G2_red_.resize(no2);
  V2_red_.resize(no2);
  for( auto j = 0; j < no; ++j ) 
  for( auto i = 0; i < no; ++i ) {
    G2_red_[i + j*no] = 0.5 * G_pqrs_[i*(no+1) + j*(no2+no3)];
    V2_red_[i + j*no] = V[i*(no+1) + j*(no2+no3)];
  }


}

template <size_t N>
double HamiltonianGenerator<N>::matrix_element_4( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex ) {

  // Get first single excition
  const auto o1 = first_occ_flipped( ket, ex );
  const auto v1 = first_occ_flipped( bra, ex );
  auto sign     = single_ex_sign( ket, v1, o1 );

  // Apply first single excitation
  spin_det_t one = 1ul;
  ket ^= (one << v1) ^ (one << o1);
  ex = bra ^ ket;

  // Get second single excitation
  const uint64_t o2 = first_occ_flipped( ket, ex );
  const uint64_t v2 = first_occ_flipped( bra, ex );
  sign             *= single_ex_sign( ket, v2, o2 );

  return sign * G_pqrs_[v1 + o1*norb_ + v2*norb2_ + o2*norb3_];

}

template <size_t N>
double HamiltonianGenerator<N>::matrix_element_22( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta ) {

  const auto o1 = first_occ_flipped( ket_alpha, ex_alpha );
  const auto v1 = first_occ_flipped( bra_alpha, ex_alpha );
  auto sign     = single_ex_sign( ket_alpha, v1, o1 );

  const auto o2 = first_occ_flipped( ket_beta, ex_beta );
  const auto v2 = first_occ_flipped( bra_beta, ex_beta );
  sign         *= single_ex_sign( ket_beta, v2, o2 );

  return sign * V_pqrs_[v1 + o1*norb_ + v2*norb2_ + o2*norb3_];

}

template <size_t N>
double HamiltonianGenerator<N>::matrix_element_2( 
  spin_det_t bra, spin_det_t ket, spin_det_t ex,
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta ) {

  const uint64_t o1 = first_occ_flipped( bra, ex );
  const uint64_t v1 = first_occ_flipped( ket, ex );
  auto sign         = single_ex_sign( bra, v1, o1 );
  
  double h_el = T_pq_[v1 + o1*norb_];

  const double* G_red_ov = G_red_.data() + v1*norb_ + o1*norb2_;
  for( auto p : bra_occ_alpha ) {
    h_el += G_red_ov[p];
  }

  const double* V_red_ov = V_red_.data() + v1*norb_ + o1*norb2_;
  for( auto p : bra_occ_beta ) {
    h_el += V_red_ov[p];
  }

  return sign * h_el;
}

template <size_t N>
double HamiltonianGenerator<N>::matrix_element_diag( 
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta ) {

  double h_el = 0;

  // One-electron piece
  for( auto p : occ_alpha ) h_el += T_pq_[p*(norb_+1)];
  for( auto p : occ_beta  ) h_el += T_pq_[p*(norb_+1)];


  // Same-spin two-body term
  for( auto q : occ_alpha )
  for( auto p : occ_alpha ) {
    h_el += G2_red_[p + q*norb_];
  }
  for( auto q : occ_beta )
  for( auto p : occ_beta ) {
    h_el += G2_red_[p + q*norb_];
  }

  // Opposite-spin two-body term
  for( auto q : occ_beta  )
  for( auto p : occ_alpha ) {
    h_el += V2_red_[p + q*norb_];
  }

  return h_el;
}

template <size_t N>
double HamiltonianGenerator<N>::matrix_element( 
  spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha, 
  spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta, 
  const std::vector<uint32_t>& bra_occ_alpha,
  const std::vector<uint32_t>& bra_occ_beta ) {

  const uint32_t ex_alpha_count = ex_alpha.count();
  const uint32_t ex_beta_count  = ex_beta.count();

  if( (ex_alpha_count + ex_beta_count) > 4 ) return 0.;

  if( ex_alpha_count == 4 ) 
    return matrix_element_4( bra_alpha, ket_alpha, ex_alpha );

  else if( ex_beta_count == 4 )
    return matrix_element_4( bra_beta, ket_beta, ex_beta );

  else if( ex_alpha_count == 2 and ex_beta_count == 2 )
    return matrix_element_22( bra_alpha, ket_alpha, ex_alpha,
      bra_beta, ket_beta, ex_beta );

  else if( ex_alpha_count == 2 )
    return matrix_element_2( bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
      bra_occ_beta );

  else if( ex_beta_count == 2 )
    return matrix_element_2( bra_beta, ket_beta, ex_beta, bra_occ_beta,
      bra_occ_alpha );

  else return matrix_element_diag( bra_occ_alpha, bra_occ_beta );

}
