#pragma once
#include "catch2/catch.hpp"
#include <mpi.h>

#define ROOT_ONLY(comm) \
  int mpi_rank; \
  MPI_Comm_rank(comm,&mpi_rank); \
  if(mpi_rank > 0 ) return;
