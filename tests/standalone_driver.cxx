#include <asci/fcidump.hpp>
#include <asci/davidson.hpp>
#include <asci/hamiltonian_generator/double_loop.hpp>
#include <asci/csr_hamiltonian.hpp>
#include <asci/util/selected_ci_diag.hpp>
#include <asci/util/fock_matrices.hpp>

#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "ini_input.hpp"

int main(int argc, char** argv) {

  constexpr size_t nwfn_bits = 64;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  {

  // Read Input Options
  std::vector< std::string > opts( argc );
  for( int i = 0; i < argc; ++i ) opts[i] = argv[i];

  auto input_file = opts.at(1);
  INIFile input(input_file);

  // Required Keywords
  auto fcidump_fname = input.getData<std::string>("CI.FCIDUMP");
  auto nalpha        = input.getData<size_t>("CI.NALPHA");
  auto nbeta         = input.getData<size_t>("CI.NBETA");

  if( nalpha != nbeta ) throw std::runtime_error("NALPHA != NBETA");

  // Read FCIDUMP File
  size_t norb  = asci::read_fcidump_norb(fcidump_fname);
  size_t norb2 = norb  * norb;
  size_t norb3 = norb2 * norb;
  size_t norb4 = norb2 * norb2;

  if( norb > nwfn_bits/2 ) throw std::runtime_error("Not Enough Bits");

  std::vector<double> T(norb2), V(norb4);
  auto E_core = asci::read_fcidump_core(fcidump_fname);
  asci::read_fcidump_1body(fcidump_fname, T.data(), norb);
  asci::read_fcidump_2body(fcidump_fname, V.data(), norb);
  
  // Set up active space
  size_t n_inactive = 0;
  if(input.containsData("CI.NINACTIVE")) {
    n_inactive = input.getData<size_t>("CI.NINACTIVE");
  }

  if( n_inactive >= norb )
    throw std::runtime_error("NINACTIVE >= NORB");

  size_t n_active = norb - n_inactive;
  if(input.containsData("CI.NACTIVE")) {
    n_active = input.getData<size_t>("CI.NACTIVE");
  }

  if( n_inactive + n_active > norb )
    throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

  size_t n_virtual = norb - n_active - n_inactive;

  if( !world_rank ) {
    std::cout << "Wavefunction Data:" << std::endl
              << "  * FCIDUMP = " << fcidump_fname << std::endl
              << "  * N_ALPHA = " << nalpha        << std::endl
              << "  * N_BETA  = " << nbeta         << std::endl
              << std::endl
              << "Active Space: " << std::endl
              << "  * N_ORB      = " << norb       << std::endl
              << "  * N_INACTIVE = " << n_inactive << std::endl
              << "  * N_ACTIVE   = " << n_active   << std::endl
              << "  * N_VIRTUAL  = " << n_virtual  << std::endl
              << std::endl;
  }


  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);


  #define T_ACTIVE(i,j) T_active[i + j*n_active]
  #define T_FULL(i,j)   T[i+n_inactive + (j+n_inactive)*norb]
  for( auto i = 0ul; i < n_active; ++i )
  for( auto j = 0ul; j < n_active; ++j ) {
    T_ACTIVE(i,j) = T_FULL(i,j);
  }
  #undef T_ACTIVE
  #undef T_FULL
  
  #define V_ACTIVE(i,j,k,l) V_active[i + j*n_active + k*n_active2 + l*n_active3]
  #define V_FULL(i,j,k,l)   \
    V[i+n_inactive + (j+n_inactive)*norb + (k+n_inactive)*norb2 + (l+n_inactive)*norb3]
  size_t n_active2 = n_active  * n_active;
  size_t n_active3 = n_active2 * n_active;
  for( auto i = 0ul; i < n_active; ++i )
  for( auto j = 0ul; j < n_active; ++j ) 
  for( auto k = 0ul; k < n_active; ++k ) 
  for( auto l = 0ul; l < n_active; ++l ) {
    V_ACTIVE(i,j,k,l) = V_FULL(i,j,k,l);
  }
  #undef V_ACTIVE
  #undef V_FULL

  // Compute inactive Fock
#if 0
  std::vector<double> F_inactive(T);
  for( auto i = 0ul; i < norb; ++i )
  for( auto j = 0ul; j < norb; ++j ) 
  for( auto p = 0ul; p < n_inactive;   ++p ) {
    F_inactive[i + j*norb] +=
      2. * V[i + j*norb   + p*(norb2 + norb3)] -
           V[i + j*norb3  + p*(norb  + norb2)];
  }
#else
  std::vector<double> F_inactive(norb2);
  asci::inactive_fock_matrix( norb, n_inactive, T.data(),
    norb, V.data(), norb, F_inactive.data(), norb );
#endif




  // Replace T_active with F_inactive
  for( auto i = 0ul; i < n_active; ++i )
  for( auto j = 0ul; j < n_active; ++j ) {
    T_active[i + j*n_active] = F_inactive[(i+n_inactive) + (j+n_inactive)*norb];
  }

  // Compute Inactive energy and increment E_core
  double E_inactive = 0.;
  for( auto i = 0ul; i < n_inactive; ++i )
    E_inactive += T[i*(norb+1)] + F_inactive[i*(norb+1)];
  std::cout << std::scientific << std::setprecision(12);
  std::cout << "E(inactive) = " << E_inactive << std::endl;
  E_core += E_inactive;
  

  
  // Hamiltonian Matrix Element Generator
  asci::DoubleLoopHamiltonianGenerator<nwfn_bits>
    ham_gen( n_active, V_active.data(), T_active.data() );

  // Compute HF Energy
  const auto hf_det = asci::canonical_hf_determinant<nwfn_bits>(nalpha, nbeta);
  const auto EHF    = ham_gen.matrix_element(hf_det, hf_det);  
  if(world_rank == 0) {
    std::cout << "E(HF) = " << EHF + E_core << " Eh" << std::endl;
    //std::cout << "E(HF) = " << EHF  << " Eh" << std::endl;
  }


  // Compute Lowest Energy Eigenvalue (ED) 
  auto dets = asci::generate_hilbert_space<nwfn_bits>(n_active, nalpha, nbeta);
  std::vector<double> X_local;
  auto E0 = asci::selected_ci_diag(dets.begin(), dets.end(), ham_gen, 1e-16,
    15, 1e-8, X_local, MPI_COMM_WORLD, true );
  
  if(world_rank == 0) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(CI)   = " << E0 + E_core << " Eh" << std::endl;
  }

  // Compute RDMs
  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm( active_ordm.size() * active_ordm.size() );
  ham_gen.form_rdms( dets.begin(), dets.end(), dets.begin(), dets.end(),
    X_local.data(), active_ordm.data(), active_trdm.data() );

  // Recompute CI energy
  double ERDM = blas::dot( active_ordm.size(), active_ordm.data(), 1, T_active.data(), 1 );
  ERDM += blas::dot( active_trdm.size(), active_trdm.data(), 1, V_active.data(), 1 );
  std::cout << "E(RDM)  = " << ERDM + E_core << std::endl;




  // Compute active Fock
  std::vector<double> F_active(norb2, 0.0);
#if 0
  for( auto m = 0ul; m < norb; ++m )
  for( auto n = 0ul; n < norb; ++n ) 
  for( auto v = 0ul; v < n_active; ++v ) 
  for( auto w = 0ul; w < n_active; ++w ) {
    F_active[m + n*norb] +=
      active_ordm[v + w*n_active] * (
            V[m + n*norb   + (v+n_inactive)*norb2 + (w+n_inactive)*norb3] -
        0.5*V[m + (w+n_inactive)*norb   + (v+n_inactive)*norb2 + n*norb3]
      );
  }
#else
  asci::active_fock_matrix( norb, n_inactive, n_active,
    V.data(), norb, active_ordm.data(), n_active, 
    F_active.data(), norb );
#endif

  // Compute Q
  std::vector<double> Q(n_active * norb, 0.0);
#if 0
  #define TRDM(v,q,x,y) active_trdm[v + w*n_active + x*n_active2 + y*n_active3]
  #define V_full(m,x,y,z) \
    V[m + (w+n_inactive)*norb + (x+n_inactive)*norb2 + (y+n_inactive)*norb3]
  for( auto v = 0ul; v < n_active; ++v )
  for( auto m = 0ul; m < norb;     ++m )
  for( auto w = 0ul; w < n_active; ++w )
  for( auto x = 0ul; x < n_active; ++x )
  for( auto y = 0ul; y < n_active; ++y ) {
    Q[v + m*n_active] += 2. * TRDM(v,w,x,y) * V_full(m,w,x,y);
  }
  #undef TRDM
  #undef V
#else
  asci::aux_q_matrix( n_active, norb, n_inactive, V.data(),
    norb, active_trdm.data(), n_active, Q.data(), n_active );
#endif

  // Compute full generalized Fock matrix
  std::vector<double> F(norb*norb,0.0);

#if 0
  // Inactive - General
  for( auto i = 0ul; i < n_inactive; ++i )
  for( auto n = 0ul; n < norb;       ++n ) {
    F[i + n*norb] = 2.*( F_inactive[n + i*norb] + F_active[n + i*norb] );
  }

  // Active - General
  std::vector<double> FIDA(norb * n_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
    norb, n_active, n_active, 1.0, F_inactive.data() + n_inactive*norb, norb,
    active_ordm.data(), n_active, 0.0, FIDA.data(), norb );
  for( auto v = 0ul; v < n_active; ++v )
  for( auto n = 0ul; n < norb;     ++n ) {
    F[v+n_inactive + n*norb] = FIDA[n + v*norb] + Q[v + n*n_active];
  }
#else
  asci::generalized_fock_matrix(norb, n_inactive, n_active,
    F_inactive.data(), norb, F_active.data(), norb,
    active_ordm.data(), n_active, Q.data(), n_active,
    F.data(), norb);
#endif

  // Recompute Energy again
#if 0
  std::vector<double> full_ordm(norb2,0.0);
  for( auto i = 0ul; i < n_inactive; ++i ) full_ordm[i*(norb+1)] = 2;
  for( auto x = 0ul; x < n_active; ++x )
  for( auto y = 0ul; y < n_active; ++y ) {
    full_ordm[x+n_inactive + (y+n_inactive)*norb] = active_ordm[x + y*n_active];
  }

  double E_FOCK = 0;
  for( auto p = 0ul; p < norb; ++p )
  for( auto q = 0ul; q < norb; ++q ) {
    E_FOCK += full_ordm[p + q*norb] * T[p + q*norb];
    if( p == q ) E_FOCK += F[p*(norb+1)];
  }
  E_FOCK *= 0.5;
#else
  auto E_FOCK = asci::energy_from_generalized_fock(
    n_inactive, n_active, T.data(), norb, active_ordm.data(),
    n_active, F.data(), norb);
#endif
  std::cout << "E(FOCK) = " << E_FOCK << std::endl;


  
  } // MPI Scope

  MPI_Finalize();

}
