#pragma once
/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include <trexio.h>
#include <stdexcept>

namespace macis {

class trexio_exception : public std::exception {
  std::string msg_;
  const char* what() const noexcept override; 

public:
  trexio_exception(std::string func_name, trexio_exit_code rc);
};

class TREXIOFile {
  trexio_t * file_handle_ = nullptr;

public:

  TREXIOFile() noexcept = default;

  TREXIOFile(std::string name, char mode, int backend); 
  ~TREXIOFile() noexcept;

  TREXIOFile(const TREXIOFile&) = delete;
  TREXIOFile(TREXIOFile&& other) noexcept; 


  int64_t read_mo_num() const;
  int64_t read_mo_2e_int_eri_size() const;
  double read_nucleus_repulsion() const;
  void read_mo_1e_int_core_hamiltonian(double* h) const;
  void read_mo_2e_int_eri(double* V) const;

  void write_mo_num(int64_t nmo);
  void write_nucleus_repulsion(double E);
  void write_mo_1e_int_core_hamiltonian(const double* h);
  void write_mo_2e_int_eri(const double* V);
  
};

}
