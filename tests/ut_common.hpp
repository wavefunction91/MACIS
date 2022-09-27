#pragma once
#include "catch2/catch.hpp"
#include <mpi.h>

#define ROOT_ONLY(comm) \
  int mpi_rank; \
  MPI_Comm_rank(comm,&mpi_rank); \
  if(mpi_rank > 0 ) return;


#define REF_DATA_PREFIX "/home/dbwy/devel/casscf/ASCI-CI/tests/ref_data"
const std::string water_ccpvdz_fcidump = 
  REF_DATA_PREFIX "/h2o.ccpvdz.fci.dat";
const std::string water_ccpvdz_rdms_fname = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cas88.rdms.bin";
const std::string water_ccpvdz_rowptr_fname = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cisd.rowptr.bin";
const std::string water_ccpvdz_colind_fname = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cisd.colind.bin";
const std::string water_ccpvdz_nzval_fname  = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cisd.nzval.bin";
const std::string ch4_wfn_fname =
  REF_DATA_PREFIX "/ch4.wfn.dat";
