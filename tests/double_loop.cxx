#include "catch2/catch.hpp"
#include <asci/fcidump.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <iostream>
#include <iomanip>

const std::string ref_file = "/home/dbwy/devel/casscf/ASCI-CI/tests/ref_data/h2o.ccpvdz.fci.dat";

TEST_CASE("Double Loop") {

  auto norb         = asci::read_fcidump_norb(ref_file);
  const auto norb2  = norb  * norb;
  const auto norb3  = norb2 * norb;
  const size_t nocc = 5;

  std::vector<double> T(norb*norb);
  std::vector<double> V(norb*norb*norb*norb);
  auto E_core = asci::read_fcidump_core(ref_file);
  asci::read_fcidump_1body(ref_file, T.data(), norb);
  asci::read_fcidump_2body(ref_file, V.data(), norb);

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
    for( auto a = nocc; a < norb; ++a )
    for( auto b = a+1;  b < norb; ++b )
    for( auto i = 0;    i < nocc; ++i )
    for( auto j = i+1;  j < nocc; ++j ) {

      auto state = hf_det;
      state.flip(i).flip(j).flip(a).flip(b);
      auto h_el = ham_gen.matrix_element(hf_det, state);
      double diag = eps[a] + eps[b] - eps[i] - eps[j];

      EMP2 += (h_el * h_el) / diag;

      REQUIRE( ham_gen.matrix_element(state,hf_det) == Approx(h_el));

    }

    for( auto a = nocc; a < norb; ++a )
    for( auto b = a+1;  b < norb; ++b )
    for( auto i = 0;    i < nocc; ++i )
    for( auto j = i+1;  j < nocc; ++j ) {

      auto state = hf_det;
      state.flip(i+32).flip(j+32).flip(a+32).flip(b+32);
      auto h_el = ham_gen.matrix_element(hf_det, state);
      double diag = eps[a] + eps[b] - eps[i] - eps[j];

      EMP2 += (h_el * h_el) / diag;

      REQUIRE( ham_gen.matrix_element(state,hf_det) == Approx(h_el));
    }

    for( auto a = nocc; a < norb; ++a )
    for( auto b = nocc; b < norb; ++b )
    for( auto i = 0;    i < nocc; ++i )
    for( auto j = 0;    j < nocc; ++j ) {

      auto state = hf_det;
      state.flip(i).flip(j+32).flip(a).flip(b+32);
      auto h_el = ham_gen.matrix_element(hf_det, state);
      double diag = eps[a] + eps[b] - eps[i] - eps[j];

      EMP2 += (h_el * h_el) / diag;

      REQUIRE( ham_gen.matrix_element(state,hf_det) == Approx(h_el));
    }


    REQUIRE((-EMP2) == Approx(-0.203989305096243));
    
  }

}
