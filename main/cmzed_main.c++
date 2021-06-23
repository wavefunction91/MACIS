// Empty main skeletton. Just prints version
#include<iostream>
#include<string>
#include "cmz_ed/CMZedConfig.h"
#include "cmz_ed/utils.h++"

using namespace cmz::ed;

int main( int argn, char* argv[] )
{
  std::cout << argv[0] << ": CMZed -- Version " << CMZed_VERSION_MAJOR << "."
                       << CMZed_VERSION_MINOR << std::endl;

  return 1;
}
