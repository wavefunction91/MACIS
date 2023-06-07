/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/types.hpp>

namespace macis {

/**
 *  @brief Generate Hamiltonian Data for 1D Hubbard
 */
void hubbard_1d(size_t nsites, double t, double U, std::vector<double>& T,
                std::vector<double>& V);

}  // namespace macis
