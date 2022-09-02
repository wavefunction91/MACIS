# CMZed
Basic ED (FCI) implementation, not based on efficiency, but rather as a simple code base for inexpensive calculations and to test other implementations. 

## Dependencies
1) Lapack for generalized eigenvalue solver.
2) Eigen library for matrix and vector classes.
3) CMake >= 3.14
4) OpenMP (optional)
5) doxygen (optional, to build documentation)

## Installation
1) mkdir build
2) cd build
3) cmake /path/to/CMZed/ -DBUILD_SHARED_LIBS=ON (for static library, this flag is not needed)  
4) cmake --build .
5) cmake --install . --prefix "/path/to/CMZmcscf/install"
6) Enjoy! CMZed is installed as shared library. The executable main (from main/main.c++) just displays the current version of the code. It can be used as template for applications.

Optional: In step (3), one can further specify

- Include -DBUILD_TESTS=ON to build the test cases in test/. Running ctest in the build directory after step (4) will check running some of these. You can use the test cases to get an idea of how to use the library.

Further, if doxygen is available, a documentation can be built after step (4) following

1) cd /path/to/CMZed/build/
2) doxygen doc_config

## Input
The required input for the test programs is an input file, which will be used to set a (string,string) dictionary to control the program flow. Each test directory includes a sample input file.

## Output
The CMZed library implements a basic Hamiltonian class, with a Lanczos algorithm to compute ground state energies, vector and 1- and 2-rdms.

The basis input for any ED/FCI method is an FCIDUMP file defining the many-Fermion Hamiltonian in 2nd quantization. Further, one has to specify how many electrons of each spin to populate the system with. 
