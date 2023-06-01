# About

Many-Body Adaptive Configuration Interaction Suite (MACIS) Copyright (c) 2023,
The Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

# Synopsis

The Many-Body Adaptive Configuration Interaction Suite (MACIS) is a modern,
modular C++ library for high-performance quantum many-body methods based on
configuration interaction (CI). MACIS currently provides a reuseable and
extentible interface for the development of full CI (FCI), complete
active-space (CAS) and selected-CI (sCI) methods for quantum chemistry.
Efforts have primarily been focused on the development of distributed memory
variants of the adaptive sampling CI (ASCI) method on CPU architectures,
and work is underway to extend the functionality set to other methods 
commonly encountered in quantum chemistry and to accelerator architectures
targeting exascale platforms.


MACIS is a work in progress. Its development has been funded by the
Computational Chemical Sciences (CCS) program of the U.S.  Department of Energy
Office of Science, Office of Basic Energy Science (BES). It was originally
developed under the Scalable Predictive Methods for Excitations and Correlated
Phenomena [(SPEC)](https://spec.labworks.org/home) Center.

# Dependencies

* CMake (3.14+)
* BLAS / LAPACK
* [BLAS++](https://github.com/icl-utk-edu/blaspp) / [LAPACK++](https://github.com/icl-utk-edu/lapackpp)
* MPI 
* [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) with [Kokkos](https://github.com/kokkos/mdspan) extensions
* spdlog
* OpenMP (Optional)
* Boost (Optional)
* Catch2 (Testing)

# Publications

# Build Instructions

## Influential CMake Variables

| Variable Name              | Description                                               | Default  |
|----------------------------|-----------------------------------------------------------|----------|
| `MACIS_ENABLE_OPENMP`      | Enable OpenMP Bindings                                    |  `ON`    |
| `MACIS_ENABLE_BOOST`       | Enable Boost Bindings                                     |  `ON`    |
| `BLAS_LIBRARIES`           | Full BLAS linker.                                         |  --      |
| `LAPACK_LIBRARIES`         | Full LAPACK linker.                                       |  --      |
| `BUILD_TESTING`            | Whether to build unit tests                               |  `ON`    |

# Example Usage

Coming Soon.... See `test/standalone_driver.cxx` for an example end-to-end
invocation of MACIS for various integrands.

# License

MACIS is made freely available under the terms of a modified 3-Clause BSD
license. See LICENSE.txt for details.

# Acknowledgments

The development of MACIS has ben supported by the Center for Scalable
Predictive methods for Excitations and Correlated phenomena (SPEC), which is
funded by the U.S.  Department of Energy (DoE), Office of Science, Office of
Basic Energy Sciences, Division of Chemical Sciences, Geosciences and
Biosciences as part of the Computational Chemical Sciences (CCS) program at
Lawrence Berkeley National Laboratory under FWP 12553.
