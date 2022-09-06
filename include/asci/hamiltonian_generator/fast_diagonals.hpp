#pragma once
#include <asci/hamiltonian_generator.hpp>

namespace asci {

template <size_t N>
double HamiltonianGenerator<N>::single_orbital_en( uint32_t orb,
  const std::vector<uint32_t>& ss_occ,
  const std::vector<uint32_t>& os_occ ) const {

  // One electron component
  double orb_en = T_pq_[ orb + orb*norb_ ];

  // Same-spin two-body term
  for( auto q : ss_occ )
    orb_en += G2_red_[orb + q*norb_] + G2_red_[q + orb*norb_];
  orb_en -= G2_red_[orb + orb*norb_];

  // Opposite-spin two-body term
  for( auto q : os_occ  )
    orb_en += V2_red_[orb + q*norb_];

  return orb_en;
}

template <size_t N>
double HamiltonianGenerator<N>::fast_diag_single( 
  // These refer to original determinant
  const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ, 
  uint32_t orb_hol, uint32_t orb_par, double orig_det_Hii ) const {

  return orig_det_Hii
       + single_orbital_en( orb_par, ss_occ, os_occ )  
       - single_orbital_en( orb_hol, ss_occ, os_occ )
       - G2_red_[ orb_par + norb_*orb_hol ] 
       - G2_red_[ orb_hol + norb_*orb_par ];
}

template <size_t N>
double HamiltonianGenerator<N>::fast_diag_ss_double( 
  // These refer to original determinant
  const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ, 
  uint32_t orb_hol1, uint32_t orb_hol2, uint32_t orb_par1, uint32_t orb_par2,
  double orig_det_Hii ) const {

  return orig_det_Hii
       + single_orbital_en( orb_par1, ss_occ, os_occ ) 
       + single_orbital_en( orb_par2, ss_occ, os_occ )
       - single_orbital_en( orb_hol1, ss_occ, os_occ ) 
       - single_orbital_en( orb_hol2, ss_occ, os_occ )
       + G2_red_[ orb_hol1 + norb_*orb_hol2 ] 
       + G2_red_[ orb_hol2 + norb_*orb_hol1 ]
       + G2_red_[ orb_par1 + norb_*orb_par2 ] 
       + G2_red_[ orb_par2 + norb_*orb_par1 ]
       - G2_red_[ orb_par1 + norb_*orb_hol1 ] 
       - G2_red_[ orb_hol1 + norb_*orb_par1 ]
       - G2_red_[ orb_par2 + norb_*orb_hol1 ] 
       - G2_red_[ orb_hol1 + norb_*orb_par2 ]
       - G2_red_[ orb_par1 + norb_*orb_hol2 ] 
       - G2_red_[ orb_hol2 + norb_*orb_par1 ]
       - G2_red_[ orb_par2 + norb_*orb_hol2 ] 
       - G2_red_[ orb_hol2 + norb_*orb_par2 ];
}

template <size_t N>
double HamiltonianGenerator<N>::fast_diag_os_double( 
  // These refer to original determinant
  const std::vector<uint32_t>& up_occ, const std::vector<uint32_t>& do_occ,
  uint32_t orb_holu, uint32_t orb_hold, uint32_t orb_paru, uint32_t orb_pard,
  double orig_det_Hii ) const {

  return orig_det_Hii
       + single_orbital_en( orb_paru, up_occ, do_occ ) 
       + single_orbital_en( orb_pard, do_occ, up_occ )
       - single_orbital_en( orb_holu, up_occ, do_occ ) 
       - single_orbital_en( orb_hold, do_occ, up_occ )
       + V2_red_[ orb_holu + norb_*orb_hold ] 
       + V2_red_[ orb_paru + norb_*orb_pard ]
       - G2_red_[ orb_paru + norb_*orb_holu ] 
       - G2_red_[ orb_holu + norb_*orb_paru ]
       - G2_red_[ orb_pard + norb_*orb_hold ] 
       - G2_red_[ orb_hold + norb_*orb_pard ]
       - V2_red_[ orb_paru + norb_*orb_hold ] 
       - V2_red_[ orb_holu + norb_*orb_pard ];
}

}

