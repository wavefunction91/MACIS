#include "ut_common.hpp"
#include <asci/fcidump.hpp>
#include <asci/util/orbital_energies.hpp>
#include <iostream>
#include <iomanip>

const std::string ref_file = "/home/dbwy/devel/casscf/ASCI-CI/tests/ref_data/h2o.ccpvdz.fci.dat";
TEST_CASE("FCIDUMP") {

  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("READ") {

    size_t norb_ref = 24;
    SECTION("NORB") {
      auto norb = asci::read_fcidump_norb(ref_file);
      REQUIRE( norb == norb_ref );
    }

    SECTION("Core") {
      auto coreE = asci::read_fcidump_core(ref_file);
      REQUIRE(coreE == Approx(9.191200742618042));
    }

    SECTION("OneBody") {
      std::vector<double> T(norb_ref*norb_ref);
      asci::read_fcidump_1body(ref_file, T.data(), norb_ref);
      double sum = std::accumulate(T.begin(), T.end(), 0.0);
      REQUIRE( sum == Approx(-1.095432762653e+02) );
    }

    SECTION("TwoBody") {
      std::vector<double> V(norb_ref*norb_ref*norb_ref*norb_ref);
      asci::read_fcidump_2body(ref_file, V.data(), norb_ref);
      double sum = std::accumulate(V.begin(), V.end(), 0.0);
      REQUIRE( sum == Approx(2.701609068389e+02) );
    }

    SECTION("Validity Checks") {
      auto norb = norb_ref;
      size_t nocc = 5;
      const auto norb2 = norb*norb;
      const auto norb3 = norb2 * norb;
      std::vector<double> T(norb*norb);
      std::vector<double> V(norb*norb*norb*norb);
      asci::read_fcidump_1body(ref_file, T.data(), norb);
      asci::read_fcidump_2body(ref_file, V.data(), norb);

      std::vector<double> eps(norb);
      asci::canonical_orbital_energies(asci::NumOrbital(norb),
        asci::NumInactive(nocc), T.data(), norb, V.data(), norb,
        eps.data());

      // Check orbital energies
      std::vector<double> ref_eps = {
        -20.5504959651472, -1.33652308180416, -0.699084807015877,
        -0.566535827115135, -0.493126779151562, 0.185508268977809,
        0.25620321890592, 0.78902083505165, 0.854064342891826,
        1.16354210561965, 1.20037835222836, 1.25333579878846,
        1.44452900701373, 1.4762147730592, 1.67453827449385,
        1.86734472965445, 1.93460967763627, 2.4520550848955,
        2.48955724627828, 3.28543361094139, 3.33853354817945,
        3.5101570353257, 3.86543012937399, 4.14719831587618,
      };

      for( auto i = 0ul; i < norb; ++i ) 
        REQUIRE(eps[i] == Approx(ref_eps[i]));


      // MP2
      double EMP2 = 0.;
      for( size_t i = 0;    i < nocc; ++i )
      for( size_t a = nocc; a < norb; ++a )
      for( size_t j = 0;    j < nocc; ++j )
      for( size_t b = nocc; b < norb; ++b ) {
        double den = eps[a] + eps[b] - eps[i] - eps[j];
        double dir = V[a + i*norb + b*norb2 + j*norb3];
        double exh = V[b + i*norb + a*norb2 + j*norb3];

        EMP2 -= (dir*(2*dir - exh)) / den;
      }
      REQUIRE( EMP2 == Approx(-0.203989305096243) );
    }

  }
}
