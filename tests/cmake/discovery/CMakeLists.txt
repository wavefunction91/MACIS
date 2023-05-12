cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( macis_cmake_discovery LANGUAGES C CXX )

find_package( macis REQUIRED )
add_executable( macis_link_tester macis_link_tester.cxx )
target_link_libraries( macis_link_tester PUBLIC macis::macis )
