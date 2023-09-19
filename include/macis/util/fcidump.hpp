/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>
#include <string>

namespace macis {

/**
 *  @brief Extract the number of orbitals from a FCIDUMP file
 *
 *  @param[in] fname Filename of FCIDUMP file
 *  @returns The number of orbitals represented in `fname`
 */
uint32_t read_fcidump_norb(std::string fname);

/**
 *  @brief Extract the "core" energy from a FCIDUMP file
 *
 *  @param[in] fname Filename of FCIDUMP file
 *  @returns The "core" energy of the Hamiltonian in `fname`
 */
double read_fcidump_core(std::string fname);

/**
 *  @brief Extract the one-body Hamiltonian from a FCIDUMP file
 *
 *  Raw memory variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] T The one-body Hamiltonian contained in `filename` (col major)
 *  @param[in]  LDT The leading dimension of `T`
 */
void read_fcidump_1body(std::string fname, double* T, size_t LDT);

/**
 *  @brief Extract the two-body Hamiltonian from a FCIDUMP file
 *
 *  Raw memory variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] V The two-body Hamiltonian contained in `filename` (col major)
 *  @param[in]  LDV The leading dimension of `V`
 */
void read_fcidump_2body(std::string fname, double* V, size_t LDV);

/**
 *  @brief Extract the one-body Hamiltonian from a FCIDUMP file
 *
 *  mdspan variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] T The one-body Hamiltonian contained in `filename` (col major)
 */
void read_fcidump_1body(std::string fname, col_major_span<double, 2> T);

/**
 *  @brief Extract the two-body Hamiltonian from a FCIDUMP file
 *
 *  mdspan variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] V The two-body Hamiltonian contained in `filename` (col major)
 */
void read_fcidump_2body(std::string fname, col_major_span<double, 4> V);

/**
 * @brief Check whether the 2-body contribution of the Hamiltonian is
 * exclusively diagonal.
 *
 * @param[in] fname: Filename of FCIDUMP file
 *
 * @returns bool: Is the 2-body contribution to the Hamiltonian exclusively
 *                diagonal?
 */
bool is_2body_diagonal(std::string fname);

/**
 *  @brief Write an FCIDUMP file from a 2-body hamiltonian
 *
 *  @param[in] fname Name of the FCIDUMP file to write
 *  @param[in] norb  Numeber of orbitals for the Hamiltonian
 *  @param[in] T     The one-body Hamiltonian
 *  @param[in] LDT   The leading dimension of `T`
 *  @param[in] V     The two-body Hamiltonian
 *  @param[in] LDV   Vhe leading dimension of `V`
 *  @param[in] E_core The "core" energy of the Hamiltonian
 */
void write_fcidump(std::string fname, size_t norb, const double* T, size_t LDT,
                   const double* V, size_t LDV, double E_core);

}  // namespace macis
