#pragma once
#include <string>
#include <vector>
#include <bitset>
#include <fstream>
#include <sstream>
#include <iostream>
#include <asci/sd_operations.hpp>

namespace asci {

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

  states.reserve(nstate);
  coeffs.reserve(nstate);
  while( std::getline(file, line) ) {
    std::stringstream ss{line};
    std::string c, d;
    ss >> c >> d;
    states.emplace_back( asci::from_canonical_string<N>(d) );
    coeffs.emplace_back( std::stod(c) );
  }

}

}
