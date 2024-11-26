/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include "macis/gf/gf.hpp"

namespace macis {

std::vector<std::complex<double> > GetGFFreqGrid(const GFSettings &settings) {
  double wmin = settings.wmin;
  double wmax = settings.wmax;
  double eta = settings.eta;
  double beta = settings.beta;
  size_t nws = settings.nws;
  bool real_g = settings.real_g;

  std::complex<double> fact(0., 0.);
  if(real_g)
    fact = std::complex<double>(1., 0.);
  else {
    fact = std::complex<double>(0., 1.);
    eta = 0.;
  }

  std::vector<std::complex<double> > ws(nws, std::complex<double>(0., 0.));

  std::string scale = settings.w_scale;
  if(scale == "lin") {
    for(int i = 0; i < nws; i++)
      ws[i] = fact * (wmin + (wmax - wmin) / double(nws - 1.) * double(i) +
                      std::complex<double>(0., eta));
  } else if(scale == "matsubara") {
    if(real_g == true)
      throw(
          std::runtime_error("Error in GetGFFreqGrid! Asked for 'real' "
                             "Matsubara frequency grid."));
    for(int i = 0; i < nws; i++)
      ws[i] = std::complex<double>(0., (2. * double(i) + 1.) * M_PI / beta);
  } else if(scale == "log") {
    if((wmin < 0. && wmax > 0.) || (wmin > 0. && wmax < 0.))
      throw(std::runtime_error(
          "Error in GetGFFreqGrid! Requested grid touches or passes by 0."));
    for(int i = 0; i < nws; i++) {
      double step = std::log(wmin) + (std::log(wmax) - std::log(wmin)) /
                                         double(nws - 1.) * double(i);
      ws[i] = fact * std::exp(step) + std::complex<double>(0., eta);
    }
  } else
    throw(std::runtime_error(
        "Error in GetGFFreqGrid! Frequency scale passed is not supported. "
        "Options are 'lin', 'log' and 'matsubara'."));

  return ws;
}

void write_GF(const std::vector<std::vector<std::complex<double> > > &GF,
              const std::vector<std::complex<double> > &ws,
              const std::vector<int> &GF_orbs, const std::vector<int> &todelete,
              const bool is_part) {
  using dbl = std::numeric_limits<double>;
  size_t nfreqs = ws.size();
  int GFmat_size = GF_orbs.size() - todelete.size();

  if(GF_orbs.size() > 1) {
    std::string fname = is_part ? "LanGFMatrix_ADD.dat" : "LanGFMatrix_SUB.dat";
    std::ofstream ofile(fname);
    ofile.precision(dbl::max_digits10);
    for(int iii = 0; iii < nfreqs; iii++) {
      ofile << std::scientific << real(ws[iii]) << " " << imag(ws[iii]) << " ";
      for(int jjj = 0; jjj < GFmat_size; jjj++) {
        for(int lll = 0; lll < GFmat_size; lll++)
          ofile << std::scientific << real(GF[iii][jjj * GFmat_size + lll])
                << " " << imag(GF[iii][jjj * GFmat_size + lll]) << " ";
      }
      ofile << std::endl;
    }

    std::string fname2 = is_part ? "GFMatrix_OrbitalIndices_ADD.dat"
                                 : "GFMatrix_OrbitalIndices_SUB.dat";
    std::ofstream ofile2(fname2);
    for(int iii = 0; iii < GF_orbs.size(); iii++) {
      if(std::find(todelete.begin(), todelete.end(), iii) != todelete.end())
        continue;
      ofile2 << GF_orbs[iii] << std::endl;
    }
  } else {
    std::string fname = is_part ? "LanGF_ADD_" : "LanGF_SUB_";
    fname += std::to_string(GF_orbs[0] + 1) + "_" +
             std::to_string(GF_orbs[0] + 1) + ".dat";
    std::ofstream ofile(fname);
    ofile.precision(dbl::max_digits10);
    for(int iii = 0; iii < nfreqs; iii++)
      ofile << std::scientific << real(ws[iii]) << " " << imag(ws[iii]) << " "
            << real(GF[iii][0]) << " " << imag(GF[iii][0]) << std::endl;
  }
}

}  // namespace macis
