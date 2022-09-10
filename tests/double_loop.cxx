#include "ut_common.hpp"
#include <asci/fcidump.hpp>
#include <asci/read_wavefunction.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <iostream>
#include <iomanip>

const std::string ref_fcidump = "/home/dbwy/devel/casscf/ASCI-CI/tests/ref_data/h2o.ccpvdz.fci.dat";
const std::string ref_wfn     = "/home/dbwy/devel/casscf/ASCI-CI/tests/ref_data/ch4.wfn.dat";

TEST_CASE("Double Loop") {

  ROOT_ONLY(MPI_COMM_WORLD);

  auto norb         = asci::read_fcidump_norb(ref_fcidump);
  const auto norb2  = norb  * norb;
  const auto norb3  = norb2 * norb;
  const size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = asci::read_fcidump_core(ref_fcidump);
  asci::read_fcidump_1body(ref_fcidump, T.data(), norb);
  asci::read_fcidump_2body(ref_fcidump, V.data(), norb);

  using generator_type = asci::DoubleLoopHamiltonianGenerator<64>;

  generator_type ham_gen(norb, V.data(), T.data());
  const auto hf_det = asci::canonical_hf_determinant<64>(nocc, nocc);

  std::vector<double> eps(norb);
  for( auto p = 0ul; p < norb; ++p ) {
    double tmp = 0.;
    for( auto i = 0ul; i < nocc; ++i ) {
      tmp += 2.*V[p*(norb + 1) + i*(norb2 + norb3)] 
              - V[p*(1 + norb3) + i*(norb+norb2)];
    }      
    eps[p] = T[p*(norb+1)] + tmp;
  }
  const auto EHF = ham_gen.matrix_element(hf_det, hf_det);

  SECTION("HF Energy") {
    REQUIRE(EHF+E_core == Approx(-76.0267803489191));
  }

  SECTION("Excited Diagonals") {
    auto state = hf_det;
    std::vector<uint32_t> occ = {0,1,2,3,4};

    SECTION("Singles") {
      state.flip(0).flip(nocc);
      const auto ES = ham_gen.matrix_element(state, state);
      REQUIRE(ES == Approx(-6.488097259228e+01)); 

      auto fast_ES = ham_gen.fast_diag_single(occ, occ, 0, nocc, EHF);
      REQUIRE(ES == Approx(fast_ES)); 
    }

    SECTION("Doubles - Same Spin") {
      state.flip(0).flip(nocc).flip(1).flip(nocc+1);
      const auto ED = ham_gen.matrix_element(state, state);
      REQUIRE(ED == Approx(-6.314093508151e+01)); 

      auto fast_ED =
        ham_gen.fast_diag_ss_double(occ, occ, 0, 1, nocc, nocc+1, EHF);
      REQUIRE(ED == Approx(fast_ED)); 
    }

    SECTION("Doubles - Opposite Spin") {
      state.flip(0).flip(nocc).flip(1+32).flip(nocc+1+32);
      const auto ED = ham_gen.matrix_element(state, state);
      REQUIRE(ED == Approx(-6.304547887231e+01)); 

      auto fast_ED =
        ham_gen.fast_diag_os_double(occ, occ, 0, 1, nocc, nocc+1, EHF);
      REQUIRE(ED == Approx(fast_ED)); 
    }
  }


  SECTION("Brilloin") {

    // Alpha -> Alpha
    for( size_t i = 0;    i < nocc; ++i )
    for( size_t a = nocc; a < norb; ++a ) {

      // Generate excited determinant
      std::bitset<64> state = hf_det; state.flip(i).flip(a);
      auto el_1 = ham_gen.matrix_element(hf_det,state);
      auto el_2 = ham_gen.matrix_element(state,hf_det);
      REQUIRE(std::abs(el_1) < 1e-6);
      REQUIRE(el_1 == Approx(el_2));

    }
    
    // Beta -> Beta
    for( size_t i = 0;    i < nocc; ++i )
    for( size_t a = nocc; a < norb; ++a ) {

      // Generate excited determinant
      std::bitset<64> state = hf_det; state.flip(i+32).flip(a+32);
      auto el_1 = ham_gen.matrix_element(hf_det,state);
      auto el_2 = ham_gen.matrix_element(state,hf_det);
      REQUIRE(std::abs(el_1) < 1e-6);
      REQUIRE(el_1 == Approx(el_2));

    }

  }

  SECTION("MP2") {

    double EMP2 = 0.;
    for( size_t a = nocc; a < norb; ++a )
    for( size_t b = a+1;  b < norb; ++b )
    for( size_t i = 0;    i < nocc; ++i )
    for( size_t j = i+1;  j < nocc; ++j ) {

      auto state = hf_det;
      state.flip(i).flip(j).flip(a).flip(b);
      auto h_el = ham_gen.matrix_element(hf_det, state);
      double diag = eps[a] + eps[b] - eps[i] - eps[j];

      EMP2 += (h_el * h_el) / diag;

      REQUIRE( ham_gen.matrix_element(state,hf_det) == Approx(h_el));

    }

    for( size_t a = nocc; a < norb; ++a )
    for( size_t b = a+1;  b < norb; ++b )
    for( size_t i = 0;    i < nocc; ++i )
    for( size_t j = i+1;  j < nocc; ++j ) {

      auto state = hf_det;
      state.flip(i+32).flip(j+32).flip(a+32).flip(b+32);
      auto h_el = ham_gen.matrix_element(hf_det, state);
      double diag = eps[a] + eps[b] - eps[i] - eps[j];

      EMP2 += (h_el * h_el) / diag;

      REQUIRE( ham_gen.matrix_element(state,hf_det) == Approx(h_el));
    }

    for( size_t a = nocc; a < norb; ++a )
    for( size_t b = nocc; b < norb; ++b )
    for( size_t i = 0;    i < nocc; ++i )
    for( size_t j = 0;    j < nocc; ++j ) {

      auto state = hf_det;
      state.flip(i).flip(j+32).flip(a).flip(b+32);
      auto h_el = ham_gen.matrix_element(hf_det, state);
      double diag = eps[a] + eps[b] - eps[i] - eps[j];

      EMP2 += (h_el * h_el) / diag;

      REQUIRE( ham_gen.matrix_element(state,hf_det) == Approx(h_el));
    }


    REQUIRE((-EMP2) == Approx(-0.203989305096243));
    
  }


  SECTION("RDM") {

    std::vector<double> ordm(norb*norb,0.0), trdm(norb3 * norb,0.0);
    std::vector<std::bitset<64>> dets = { 
      asci::canonical_hf_determinant<64>(nocc,nocc) 
    };
  
    std::vector<double> C = { 1. };
  
    ham_gen.form_rdms( dets.begin(), dets.end(), dets.begin(), dets.end(), 
      C.data(), ordm.data(), trdm.data() );

    auto E_tmp = blas::dot(norb2, ordm.data(),1, T.data(),1) + 
                 blas::dot(norb3*norb, trdm.data(),1, V.data(),1);
    REQUIRE(E_tmp == Approx(EHF));
  }
}

TEST_CASE("RDMS") {

  ROOT_ONLY(MPI_COMM_WORLD);

  auto norb         = 34;
  const auto norb2  = norb  * norb;
  const auto norb3  = norb2 * norb;
  const size_t nocc = 5;

  std::vector<double> T(norb*norb, 0.0);
  std::vector<double> V(norb*norb*norb*norb, 0.0);
  std::vector<double> ordm(norb*norb,0.0), trdm(norb3 * norb,0.0);

  using generator_type = asci::DoubleLoopHamiltonianGenerator<128>;
  generator_type ham_gen(norb, V.data(), T.data());
  auto abs_sum = [](auto a, auto b){ return a + std::abs(b); };

  SECTION("HF") {
    std::vector<std::bitset<128>> dets = { 
      asci::canonical_hf_determinant<128>(nocc,nocc) 
    };
  
    std::vector<double> C = { 1. };
  
    ham_gen.form_rdms( dets.begin(), dets.end(), dets.begin(), dets.end(), 
      C.data(), ordm.data(), trdm.data() );


   #define TRDM(i,j,k,l) trdm[i + j*norb + k*norb2 + l*norb3] 
   #define ORDM(i,j) ordm[i+j*norb]
    for( auto i = 0ul; i < nocc; ++i )
    for( auto j = 0ul; j < nocc; ++j )
    for( auto k = 0ul; k < nocc; ++k )
    for( auto l = 0ul; l < nocc; ++l ) {
      TRDM(i,j,l,k) -= 0.5  * ORDM(i,j) * ORDM(k,l);
      TRDM(i,j,l,k) += 0.25 * ORDM(i,l) * ORDM(k,j);
    }
    #undef TRDM
    #undef ORDM
    auto sum = std::accumulate(trdm.begin(), trdm.end(), 0.0, abs_sum );
    REQUIRE( sum < 1e-15 );
  
    for( auto i = 0ul; i < nocc; ++i ) ordm[i*(norb+1)] -= 2.0;
    sum = std::accumulate(ordm.begin(), ordm.end(), 0.0, abs_sum );
    REQUIRE(sum < 1e-15);

    
  }

  SECTION("CI") {

    std::vector<std::bitset<128>> states; std::vector<double> coeffs;
    asci::read_wavefunction<128>( ref_wfn, states, coeffs );


    coeffs.resize(5000);
    states.resize(5000);

    // Renormalize C for trace computation
    auto c_nrm = blas::nrm2(coeffs.size(), coeffs.data(), 1);
    blas::scal(coeffs.size(), 1./c_nrm, coeffs.data(), 1);

    ham_gen.form_rdms( states.begin(), states.end(), states.begin(), states.end(), 
      coeffs.data(), ordm.data(), trdm.data());
    auto sum_ordm = std::accumulate(ordm.begin(), ordm.end(), 0.0, abs_sum );
    auto sum_trdm = std::accumulate(trdm.begin(), trdm.end(), 0.0, abs_sum );
    REQUIRE(sum_ordm == Approx(1.038559618650e+01));
    REQUIRE(sum_trdm == Approx(9.928269867561e+01));

    double trace_ordm = 0.;
    for( auto p = 0; p < norb; ++p ) trace_ordm += ordm[p*(norb+1)];
    REQUIRE(trace_ordm == Approx(2.0*nocc));

    // Check symmetries
    for( auto p = 0; p < norb; ++p )
    for( auto q = p; q < norb; ++q ) {
      REQUIRE( ordm[p + q*norb] == Approx(ordm[q + p*norb]) );
    }
    
#if 0 
    #define TRDM(i,j,k,l) trdm[i + j*norb + k*norb2 + l*norb3] 
    for( auto p = 0; p < norb; ++p )
    for( auto q = 0; q < norb; ++q ) 
    for( auto r = 0; r < norb; ++r )
    for( auto s = 0; s < norb; ++s ) {
      REQUIRE( TRDM(p,q,r,s) == Approx(-TRDM(p,q,s,r)) );
      REQUIRE( TRDM(p,q,r,s) == Approx(-TRDM(q,p,r,s)) );
      REQUIRE( TRDM(p,q,r,s) == Approx(-TRDM(s,r,p,q)) );
      REQUIRE( TRDM(p,q,r,s) == Approx(-TRDM(r,s,q,p)) );

      REQUIRE( TRDM(p,q,r,s) == Approx(TRDM(q,p,s,r)) );
      REQUIRE( TRDM(p,q,r,s) == Approx(TRDM(s,r,q,p)) );
      REQUIRE( TRDM(p,q,r,s) == Approx(TRDM(r,s,p,q)) );
    }
    #undef TRDM
#endif

  }

}