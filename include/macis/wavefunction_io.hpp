/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <string>
#include <vector>
#include <bitset>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <macis/sd_operations.hpp>

namespace macis {

template <size_t N>
void read_wavefunction( std::string fname, std::vector<std::bitset<N>>& states,
  std::vector<double>& coeffs) {

  states.clear();
  coeffs.clear();

  std::ifstream file(fname);
  std::string line;

  size_t nstate, norb, nalpha, nbeta;
  {
    std::getline(file, line);
    std::stringstream ss{line};
    std::string nstate_, norb_, nalpha_, nbeta_;
    ss >> nstate_ >> norb_ >> nalpha_ >> nbeta_;
    nstate = std::stoi(nstate_);
    norb   = std::stoi(norb_);
    nalpha = std::stoi(nalpha_);
    nbeta  = std::stoi(nbeta_);
  }
  (void)nstate;
  (void)norb;
  (void)nalpha;
  (void)nbeta;

  states.reserve(nstate);
  coeffs.reserve(nstate);
  while( std::getline(file, line) ) {
    std::stringstream ss{line};
    std::string c, d;
    ss >> c >> d;
    states.emplace_back( from_canonical_string<N>(d) );
    coeffs.emplace_back( std::stod(c) );
  }

}

template <size_t N>
void write_wavefunction( std::string fname, size_t norb, const std::vector<std::bitset<N>>& states,
  const std::vector<double>& coeffs) {

  if( states.size() != coeffs.size() ) throw std::runtime_error("Invalid Wave Function Dimensions");

  if( !states.size()  ) return;

  const auto nstates = states.size();
  const auto nalpha  = (states[0] << N/2).count();
  const auto nbeta   = (states[0] >> N/2).count();
  

  std::ofstream file(fname);
  file << nstates << " " << norb << " " << nalpha << " " << nbeta << std::endl;
  file << std::scientific << std::setprecision(16);
  for(size_t i = 0; i < nstates; ++i) {
    file << std::setw(30) << coeffs[i] << " " << to_canonical_string(states[i]) << " " << std::endl;
  }

}

}
