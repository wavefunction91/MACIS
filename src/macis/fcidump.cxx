#include <macis/fcidump.hpp>
#include <fstream>
#include <string>
#include <regex>
#include <iostream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>


std::vector<std::string> split(const std::string str, const std::string regex_str) {
    std::regex regexz(regex_str);
    std::vector<std::string> list(std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  std::sregex_token_iterator());
    std::vector<std::string> clean_list;
    std::copy_if( list.begin(), list.end(), std::back_inserter(clean_list),
      [](auto& s){ return s.size() > 0; } );
    return clean_list;
}

bool is_float( const std::string& str ) {
  return std::any_of(str.begin(), str.end(), [](auto c){ return std::isalpha(c) or c == '.'; });
}


auto fcidump_line( const std::vector<std::string>& tokens ) {
  auto idx_first = is_float( tokens.back()  );
  auto int_first = is_float( tokens.front() );

  if( idx_first and int_first ) throw std::runtime_error("Invalid FCIDUMP Line");

  int32_t p,q,r,s;
  double integral;

  if(idx_first) {
    p = std::stoi(tokens[0]);
    q = std::stoi(tokens[1]);
    r = std::stoi(tokens[2]);
    s = std::stoi(tokens[3]);
    integral = std::stod(tokens[4]);
  } else { 
    p = std::stoi(tokens[1]);
    q = std::stoi(tokens[2]);
    r = std::stoi(tokens[3]);
    s = std::stoi(tokens[4]);
    integral = std::stod(tokens[0]);
  }

  if(p < 0 or q < 0 or r < 0 or s < 0) throw std::runtime_error("Invalid Orb Idx");

  return std::make_tuple(p,q,r,s,integral);

}


enum LineClassification {
  Core,
  OneBody,
  TwoBody
};

LineClassification line_classification(int p, int q, int r, int s) {
  if( !(p or q or r or s) )      return LineClassification::Core;
  else if( p and q and r and s ) return LineClassification::TwoBody;
  else                           return LineClassification::OneBody;
}

namespace macis {

uint32_t read_fcidump_norb( std::string fname ) {

  std::ifstream file(fname);
  std::string line;
  int32_t max_idx = 0;
  while(std::getline(file, line)) {
    auto tokens = split(line, " ");
    if( tokens.size() != 5 ) continue; // not a valid FCIDUMP line

    auto [p,q,r,s,integral] = fcidump_line(tokens);

    max_idx = std::max(max_idx, std::max(p, std::max(q, std::max(r, s))));
  }

  return max_idx;

}



double read_fcidump_core( std::string fname ) {

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = split(line, " ");
    if( tokens.size() != 5 ) continue; // not a valid FCIDUMP line

    auto [p,q,r,s,integral] = fcidump_line(tokens);
    auto lc = line_classification(p,q,r,s);
    if( lc == LineClassification::Core ) {
      return integral;
    }
  }
  return 0.0;

}


void read_fcidump_1body( std::string fname, col_major_span<double,2> T ) {

  if(T.extent(0) != T.extent(1)) throw std::runtime_error("T must be square");

  auto norb = read_fcidump_norb(fname);
  if( T.extent(0) != norb ) throw std::runtime_error("T is of improper dimension");

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = split(line, " ");
    if( tokens.size() != 5 ) continue; // not a valid FCIDUMP line

    auto [p,q,r,s,integral] = fcidump_line(tokens);
    auto lc = line_classification(p,q,r,s);
    if( lc == LineClassification::OneBody ) {
      p--; q--;
      T(p,q) = integral;
      T(q,p) = integral;
    }
  }

}

void read_fcidump_1body( std::string fname, double* T, size_t LDT ) {

  auto norb = read_fcidump_norb(fname);
  col_major_span<double,2> T_map(T, LDT, norb);
  read_fcidump_1body(fname, 
    KokkosEx::submdspan(T_map, std::pair{0,norb}, Kokkos::full_extent));

}




void read_fcidump_2body( std::string fname, col_major_span<double,4> V ) {

  if(V.extent(0) != V.extent(1)) throw std::runtime_error("V must be square");
  if(V.extent(0) != V.extent(2)) throw std::runtime_error("V must be square");
  if(V.extent(0) != V.extent(3)) throw std::runtime_error("V must be square");

  auto norb = read_fcidump_norb(fname);
  if( V.extent(0) != norb ) throw std::runtime_error("V is of improper dimension");

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = split(line, " ");
    if( tokens.size() != 5 ) continue; // not a valid FCIDUMP line

    auto [p,q,r,s,integral] = fcidump_line(tokens);
    auto lc = line_classification(p,q,r,s);
    if( lc == LineClassification::TwoBody ) {
      p--; q--; r--; s--;
      V(p,q,r,s) = integral; // (pq|rs)
      V(p,q,s,r) = integral; // (pq|sr)
      V(q,p,r,s) = integral; // (qp|rs)
      V(q,p,s,r) = integral; // (qp|sr)

      V(r,s,p,q) = integral; // (rs|pq)
      V(s,r,p,q) = integral; // (sr|pq)
      V(r,s,q,p) = integral; // (rs|qp)
      V(s,r,q,p) = integral; // (sr|qp)
    }
  }
}

void read_fcidump_2body( std::string fname, double* V, size_t LDV ) {

  auto norb = read_fcidump_norb(fname);
  col_major_span<double,4> V_map(V, LDV, LDV, LDV, norb);
  auto sl = std::pair{0,norb};
  read_fcidump_2body(fname, 
    KokkosEx::submdspan(V_map, sl, sl ,sl, Kokkos::full_extent));

}


void write_fcidump( std::string fname, size_t norb, const double *T, size_t LDT, 
  const double* V, size_t LDV, double E_core) {

  auto logger = spdlog::basic_logger_mt("fcidump", fname); 
  logger->set_pattern("%v");
  const std::string fmt_string = "{:8} {:8} {:8} {:8} {:25.16e}";

  // Write two body
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j)
  for(size_t k = 0; k < norb; ++k)
  for(size_t l = 0; l < norb; ++l) {
    logger->info(fmt_string, i+1,j+1,k+1,l+1, 
      V[i + j*LDV + k*LDV*LDV + l*LDV*LDV*LDV]);
  }

  // Write one body
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) {
    logger->info(fmt_string, i+1,j+1,0,0, T[i + j*LDT]);
  }

  // Write core
  logger->info(fmt_string, 0,0,0,0, E_core);

}











void read_rdms_binary(std::string fname, size_t norb, double* ORDM, size_t LDD1,
  double* TRDM, size_t LDD2) {

  // Zero out rdms in case of sparse data
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) {
    ORDM[i + j*LDD1] = 0;
  }

  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) 
  for(size_t k = 0; k < norb; ++k) 
  for(size_t l = 0; l < norb; ++l) {
    TRDM[i + j*LDD2 + k*LDD2*LDD2 + l*LDD2*LDD2*LDD2] = 0;
  }

  std::ifstream in_file(fname, std::ios::binary);
  if(!in_file) throw std::runtime_error( fname + " not available" );

  int _norb_read;
  in_file.read((char*)&_norb_read, sizeof(int));
  if( _norb_read != (int)norb ) 
    throw std::runtime_error("NORB in RDM file doesn't match " + std::to_string(norb) + " " + std::to_string(_norb_read));

  std::vector<double> raw(norb*norb*norb*norb);

  // Read 1RDM
  in_file.read( (char*)raw.data(), norb*norb * sizeof(double) );
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) {
    ORDM[i + j*LDD1] = raw[i + j*norb];
  }

  // Read 2RDM
  in_file.read( (char*)raw.data(), norb*norb*norb*norb * sizeof(double) );
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) 
  for(size_t k = 0; k < norb; ++k) 
  for(size_t l = 0; l < norb; ++l) {
    TRDM[i + j*LDD2 + k*LDD2*LDD2 + l*LDD2*LDD2*LDD2] = 
      raw[i + j*norb + k*norb*norb + l*norb*norb*norb]; 
  }

}



void write_rdms_binary(std::string fname, size_t norb, const double* ORDM, 
  size_t LDD1, const double* TRDM, size_t LDD2) {

  std::ofstream out_file(fname, std::ios::binary);
  if(!out_file) throw std::runtime_error( fname + " not available" );

  int _norb_write = norb;
  out_file.write((char*)&_norb_write, sizeof(int));


  std::vector<double> raw(norb*norb*norb*norb);

  // Pack and Write 1RDM
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) {
    raw[i + j*norb] = ORDM[i + j*LDD1];
  }
  out_file.write( (char*)raw.data(), norb*norb * sizeof(double) );

  // Pack and Write 2RDM
  for(size_t i = 0; i < norb; ++i)
  for(size_t j = 0; j < norb; ++j) 
  for(size_t k = 0; k < norb; ++k) 
  for(size_t l = 0; l < norb; ++l) {
    raw[i + j*norb + k*norb*norb + l*norb*norb*norb] = 
      TRDM[i + j*LDD2 + k*LDD2*LDD2 + l*LDD2*LDD2*LDD2];
  }
  out_file.write( (char*)raw.data(), norb*norb*norb*norb * sizeof(double) );

}


}
