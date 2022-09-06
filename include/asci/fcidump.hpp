#pragma once
#include <string>
//#include <span>

namespace asci {

uint32_t read_fcidump_norb( std::string fname );

double read_fcidump_core( std::string fname );
void read_fcidump_1body( std::string fname, double* T, size_t LDT );
void read_fcidump_2body( std::string fname, double* V, size_t LDV );

}
