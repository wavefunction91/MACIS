cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( asci_cmake_discovery LANGUAGES C CXX )

find_package( asci REQUIRED )
add_executable( asci_link_tester asci_link_tester.cxx )
target_link_libraries( asci_link_tester PUBLIC asci::asci )
