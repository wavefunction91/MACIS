#include <asci/fcidump.hpp>
#include <fstream>
//#include <ranges>
//#include <string_view>
#include <string>
#include <regex>
#include <iostream>
#include <iomanip>


std::vector<std::string> split(const std::string str, const std::string regex_str) {
    std::regex regexz(regex_str);
    std::vector<std::string> list(std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  std::sregex_token_iterator());
    return list;
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
    integral = std::stod(tokens[1]);
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

namespace asci {

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

void read_fcidump_1body( std::string fname, double* T, size_t LDT ) {

  auto norb = read_fcidump_norb(fname);
  if( LDT < norb ) throw std::runtime_error("LDT < NORB");

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = split(line, " ");
    if( tokens.size() != 5 ) continue; // not a valid FCIDUMP line

    auto [p,q,r,s,integral] = fcidump_line(tokens);
    auto lc = line_classification(p,q,r,s);
    if( lc == LineClassification::OneBody ) {
      p--; q--;
      T[p + q*LDT] = integral;
      T[q + p*LDT] = integral;
    }
  }

}

void read_fcidump_2body( std::string fname, double* V, size_t LDV ) {

  auto norb = read_fcidump_norb(fname);
  if( LDV < norb ) throw std::runtime_error("LDV < NORB");
  size_t LDV2 = LDV  * LDV;
  size_t LDV3 = LDV2 * LDV;

  std::ifstream file(fname);
  std::string line;
  #define V_pqrs(p,q,r,s) V[p + q*LDV + r*LDV2 + s*LDV3]
  while(std::getline(file, line)) {
    auto tokens = split(line, " ");
    if( tokens.size() != 5 ) continue; // not a valid FCIDUMP line

    auto [p,q,r,s,integral] = fcidump_line(tokens);
    auto lc = line_classification(p,q,r,s);
    if( lc == LineClassification::TwoBody ) {
      p--; q--; r--; s--;
      V_pqrs(p,q,r,s) = integral; // (pq|rs)
      V_pqrs(p,q,s,r) = integral; // (pq|sr)
      V_pqrs(q,p,r,s) = integral; // (qp|rs)
      V_pqrs(q,p,s,r) = integral; // (qp|sr)

      V_pqrs(r,s,p,q) = integral; // (rs|pq)
      V_pqrs(s,r,p,q) = integral; // (sr|pq)
      V_pqrs(r,s,q,p) = integral; // (rs|qp)
      V_pqrs(s,r,q,p) = integral; // (sr|qp)
    }
  }

}

}
