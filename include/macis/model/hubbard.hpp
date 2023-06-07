#pragma once

#include <macis/types.hpp>

namespace macis {

/**
 *  @brief Generate Hamiltonian Data for 1D Hubbard
 */
void hubbard_1d(size_t nsites, double t, double U, std::vector<double>& T,
                std::vector<double>& V);

}  // namespace macis
