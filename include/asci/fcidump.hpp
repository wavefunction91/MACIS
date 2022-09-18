#pragma once
#include <string>
#include <asci/types.hpp>

namespace asci {

uint32_t read_fcidump_norb( std::string fname );

double read_fcidump_core( std::string fname );
void read_fcidump_1body( std::string fname, double* T, size_t LDT );
void read_fcidump_2body( std::string fname, double* V, size_t LDV );

void read_fcidump_1body( std::string fname, col_major_span<double,2> T);
void read_fcidump_2body( std::string fname, col_major_span<double,4> V);

void write_fcidump( std::string fname, size_t norb, const double *T, size_t LDT, 
  const double* V, size_t LDV, double E_core);

void read_rdms_binary(std::string fname, size_t norb, double* ORDM, size_t LDD1,
  double* TRDM, size_t LDD2);
void write_rdms_binary(std::string fname, size_t norb, const double* ORDM, 
  size_t LDD1, const double* TRDM, size_t LDD2); 

}
