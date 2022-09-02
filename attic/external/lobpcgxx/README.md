TODO....


Install Deps
============

LOBPCGXX depends on two libraries: blaspp and lapackpp from the ICL.

These may be obtained:

```
hg clone https://bitbucket.org/icl/blaspp
hg clone https://bitbucket.org/icl/lapackpp
```

and built

```
mkdir -p $PWD/build_icl/{blas,lapack}pp

cmake -Hblaspp -Bbuild_icp/blaspp -DBLASPP_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/install_icl \
      -DCMAKE_PREFIX_PATH=$PWD/install_icl 

make -C build_icl/blaspp -j <cores> install

cmake -Hlapackpp -Bbuild_icp/lapackpp -DBUILD_LAPACKPP_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/install_icl \
      -DCMAKE_PREFIX_PATH=$PWD/install_icl 

make -C build_icl/lapackpp -j <cores> install
```


Build LOBPCGXX
==============

```
cmake -H/path/to/lobpcgxx -Bbuild_lobpcgxx \
      -DCMAKE_PREFIX_PATH=/path/to/icl/install \
      -DCMAKE_INSTALL_PREFIX=/path/to/where/you/want/it/installed

make -C build_lobpcgxx -j <cores> install
````


Link to LOBPCGXX
================

Currently, the easiest way to link to LOBPCGXX is through CMake
```
# CMakeLists.txt

find_package( lobpcgxx )
target_link_libraries( my_target PUBLIC lobpcgxx::lobpcgxx )
```

Ensure that `CMAKE_PREFIX_PATH` contains the LOBPCGXX install directory
(and the ICL install directory) on CMake invocation

Testing LOBPCGXX
================


To run a test (Laplacian) problem
```
/path/to/lobpcg_build/lobpcg_tester
```
