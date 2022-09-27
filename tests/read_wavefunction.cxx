#include "ut_common.hpp"
#include <asci/read_wavefunction.hpp>
#include <iostream>
#include <iomanip>

TEST_CASE("Read Wavefunction") {

  ROOT_ONLY(MPI_COMM_WORLD);

  std::vector<std::bitset<64>> states;
  std::vector<double>          coeffs;
  asci::read_wavefunction( ch4_wfn_fname, states, coeffs );

  std::vector<std::bitset<64>> ref_states = {
    0x1f0000001f,0x100f0000100f,0x401b0000401b,0x201700002017,0x4f0000004f,
    0x9700000097,0x11b0000011b,0x1001b0001001b,0x401b0000011b,0x11b0000401b,
    0x100f0000004f,0x4f0000100f,0x9700002017,0x201700000097,0x2000f0002000f,
    0x20170000401b,0x401b00002017,0x20170000100f,0x100f00002017,0x100f0000401b,
    0x401b0000100f,0x81b0000081b,0x41b0000041b,0x21700000217,0x20f0000020f,
    0x81700000817,0x40f0000040f,0x2001700020017,0x100f0000801d,0x801d0000100f,
    0x801d0000401b,0x401b0000801d,0x801d00002017,0x20170000801d,0x11b00000097,
    0x970000011b,0x11b0000004f,0x4f0000011b,0x4f00000097,0x970000004f,
    0x1f00020207,0x202070000001f,0x801700008017,0x801b0000801b,0x800f0000800f,
    0x3d0000011b,0x11b0000003d,0x4f0000003d,0x3d0000004f,0x970000003d,
    0x3d00000097,0x3d0000401b,0x401b0000003d,0x3d0000100f,0x100f0000003d,
    0x20170000003d,0x3d00002017,0x41b00000217,0x2170000041b,0x40f00000817,
    0x8170000040f,0x20f0000081b,0x81b0000020f,0x801d0000801d,0x41700000417,
    0x80f0000080f,0x21b0000021b,0x11b00002017,0x20170000011b,0x4f0000401b,
    0x401b0000004f,0x11b0000100f,0x100f0000011b,0x4f00002017,0x20170000004f,
    0x970000401b,0x401b00000097,0x100f00000097,0x970000100f,0x1f00010813,
    0x108130000001f,0x3d0000003d,0x1f0001040b,0x1040b0000001f,0x801d00000097,
    0x970000801d,0x4f0000801d,0x801d0000004f,0x801d0000011b,0x11b0000801d,
    0x201d00008017,0x80170000201d,0x101d0000800f,0x800f0000101d,0x401d0000801b,
    0x801b0000401d,0x2000f0000041b,0x41b0002000f,0x40f0001001b,0x1001b0000040f
  };

  std::vector<double> ref_coeffs = {  
    -9.6397542632e-01, 2.4710768189e-02, 2.4710245430e-02, 2.4705391588e-02,
     2.3894278451e-02, 2.3893099930e-02, 2.3893095611e-02, 2.3033503767e-02,
     2.0632389046e-02, 2.0632389046e-02, 2.0629385763e-02, 2.0629385763e-02,
     2.0622975217e-02, 2.0622975217e-02, 2.0169827492e-02, 1.8461998224e-02,
     1.8461998224e-02, 1.8460438222e-02, 1.8460438222e-02, 1.8456064768e-02,
     1.8456064768e-02, 1.7828442105e-02, 1.7823247565e-02, 1.7821364349e-02,
     1.7820151654e-02, 1.7810033038e-02, 1.7807034061e-02, 1.7303467679e-02,
     1.7144664128e-02, 1.7144664128e-02, 1.7142476624e-02, 1.7142476624e-02,
     1.7136827667e-02, 1.7136827667e-02, 1.7020096720e-02, 1.7020096720e-02,
     1.7018430076e-02, 1.7018430076e-02, 1.7015224624e-02, 1.7015224624e-02,
     1.6040520699e-02, 1.6040520699e-02, 1.6023489248e-02, 1.6017928853e-02,
     1.6007902212e-02, 1.5590988226e-02, 1.5590988226e-02, 1.5590561289e-02,
     1.5590561289e-02, 1.5589226456e-02, 1.5589226456e-02, 1.5216669145e-02,
     1.5216669145e-02, 1.5205395275e-02, 1.5205395275e-02, 1.5201208443e-02,
     1.5201208443e-02,-1.5066379915e-02,-1.5066379915e-02,-1.5062392610e-02,
    -1.5062392610e-02, 1.5060038513e-02, 1.5060038513e-02, 1.4900289781e-02,
     1.4893431408e-02, 1.4863672808e-02, 1.4858283972e-02, 1.4833559416e-02,
     1.4833559416e-02, 1.4827860109e-02, 1.4827860109e-02, 1.4827753501e-02,
     1.4827753501e-02, 1.4827064665e-02, 1.4827064665e-02, 1.4819502053e-02,
     1.4819502053e-02, 1.4818950449e-02, 1.4818950449e-02, 1.4643040141e-02,
     1.4643040141e-02, 1.3139936019e-02,-1.3085096586e-02,-1.3085096586e-02,
     1.2726313527e-02, 1.2726313527e-02, 1.2718672959e-02, 1.2718672959e-02,
     1.2709716017e-02, 1.2709716017e-02, 1.2690423417e-02, 1.2690423417e-02,
     1.2681250240e-02, 1.2681250240e-02, 1.2679634286e-02, 1.2679634286e-02,
     1.2124316530e-02, 1.2124316530e-02,-1.1837009243e-02,-1.1837009243e-02
  };

  size_t ncheck = ref_coeffs.size();
  for( auto i = 0ul; i < ncheck; ++i ) {
    REQUIRE( states[i] == ref_states[i] );
    REQUIRE( coeffs[i] == Approx(coeffs[i]) );
  }

}
