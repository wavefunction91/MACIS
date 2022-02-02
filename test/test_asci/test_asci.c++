#include <mpi.h>
#include <iostream>
#include "cmz_ed/utils.h++"
#include "dbwy/csr_hamiltonian.hpp"
#include "dbwy/davidson.hpp"
#include "cmz_ed/slaterdet.h++"
#include "cmz_ed/integrals.h++"
#include "cmz_ed/hamil.h++"
#include "cmz_ed/lanczos.h++"
#include "cmz_ed/rdms.h++"

#include <bitset>

#include "dbwy/asci_util.hpp"

template <size_t N>
std::vector<std::bitset<N>> build_combs( uint64_t nbits, uint64_t nset ) {

  std::vector<bool> v(nbits, false);
  std::fill_n( v.begin(), nset, true );
  std::vector<std::bitset<N>> store;

  do {

    std::bitset<N> temp = 0ul;
    std::bitset<N> one  = 1ul;
    for( uint64_t i = 0; i < nbits; ++i )
    if( v[i] ) {
      temp = temp | (one << i);
    }
    store.emplace_back(temp);

  } while ( std::prev_permutation( v.begin(), v.end() ) );

  return store;

}

template <size_t N>
std::vector<std::bitset<N>> build_hilbert_space(
  size_t norbs, size_t nalpha, size_t nbeta
) {

  // Get all alpha and beta combs
  auto alpha_dets = build_combs<N>( norbs, nalpha );
  auto beta_dets  = build_combs<N>( norbs, nbeta  );

  std::vector<std::bitset<N>> states;
  states.reserve( alpha_dets.size() * beta_dets.size() );
  for( auto alpha_det : alpha_dets )
  for( auto beta_det  : beta_dets  ) {
    std::bitset<N> state = alpha_det | (beta_det << (N/2));
    states.emplace_back( state );
  }

  return states;
 
}

int main( int argc, char* argv[] ) {

  MPI_Init(NULL,NULL);
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  { // MPI Scope

  if( argc != 2 ) {
    std::cout << "Must Specify Input" << std::endl;
    return 1;
  }

  // Read Input
  std::string in_file = argv[1];
  cmz::ed::Input_t input;
  cmz::ed::ReadInput( in_file, input );

  // Get Parameters
  size_t norb = cmz::ed::getParam<int>( input, "norbs" );
  size_t nalpha  = cmz::ed::getParam<int>( input, "nups"  );
  size_t nbeta  = cmz::ed::getParam<int>( input, "ndos"  );
  size_t norb_eff = 
    cmz::ed::getParam<int>( input, "norbseff"  );
  std::string fcidump = 
    cmz::ed::getParam<std::string>( input, "fcidump_file" );

  bool do_fci = cmz::ed::getParam<bool>( input, "fci" );

  std::string wfn_file; 
  if(!do_fci) {
    try {
      wfn_file = cmz::ed::getParam<std::string>( input, "wfn_file" );
    } catch(...) {
      std::cout << "Must Specify Wfn File for non-FCI" << std::endl;
      throw;
    }
  }


#if 0
  // Read-In FCI file
  size_t norb2 = norb * norb;
  size_t norb3 = norb * norb2;
  size_t norb4 = norb * norb3;
  std::vector<double> V_pqrs(norb4), T_pq(norb2);
  double core_energy = 0.;
  {
    std::ifstream ifile(fcidump);
    std::string line;
    while( std::getline( ifile, line ) ) {
      if( line.find("ORB") != std::string::npos ) continue;
      if( line.find("SYM") != std::string::npos ) continue;
      if( line.find("END") != std::string::npos ) continue;
      // Split the line
      std::stringstream ss(line);

      // Read the line
      uint64_t p,q,r,s;
      double matel;
      ss >> p >> q >> r >> s >> matel;
      if(!ss) throw "Bad Read";

      if( p > norb or q > norb or r > norb or s > norb )
        throw "Bad Orbital Index";


      // Zero Matrix Element
      if( std::abs(matel) < 
        std::numeric_limits<double>::epsilon() ) continue;

      // Read Core Energy
      if( p == 0 and q == 0 and r == 0 and s == 0 ) {
        core_energy = matel;
      } else if( r == 0 and s == 0 ) {
        //std::cout << "T " << p << ", " << q << std::endl;
        T_pq[(p-1) + (q-1)*norb] = matel;
      } else {
        //std::cout << "V " << p << ", " << q << ", " << r << ", " << s << std::endl;
        V_pqrs[(p-1) + (q-1)*norb + (r-1)*norb2 + (s-1)*norb3] = matel;
      }
    }
  }
#endif

  cmz::ed::intgrls::integrals ints(norb, fcidump);
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr size_t nbits = 128;

#if 0
  std::vector<std::bitset<nbits>> dets;
  if( do_fci ) {
    // do FCI
    dets = build_hilbert_space<nbits>( norb_eff, nalpha, nbeta );
  } else {
    // Read in ASCI wfn
    std::ifstream ifile( wfn_file );
    std::string line;
    std::getline(ifile, line);
    while( std::getline(ifile, line) ) {
      std::stringstream ss(line);
      double c; size_t i,j;
      ss >> c >> i >> j;
      std::bitset<nbits> alpha = i;
      std::bitset<nbits> beta  = j;
      dets.emplace_back( alpha | (beta << (nbits/2)) );
    }
  }

  if(world_rank == 0)
    std::cout << "NDETS = " << dets.size() << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  dbwy::DoubleLoopHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );
  const std::bitset<nbits> hf_det = dbwy::canonical_hf_determinant(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if(world_rank == 0) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }


  auto H = dbwy::make_dist_csr_hamiltonian<int32_t>(MPI_COMM_WORLD,
    dets.begin(), dets.end(), ham_gen, 1e-12 );

  std::vector<double> X_local( H.local_row_extent() );
  auto E0 = p_davidson( 100, H, 1e-8, X_local.data() );
  if(world_rank == 0) {
    std::cout << "\nE(CI)   = " <<  E0 + ints.core_energy << std::endl;
    std::cout << "E(CORR) = " << (E0 - EHF) << std::endl;
  }

  // Gather Eigenvector
  std::vector<double> X( world_rank ? 0 : dets.size() );
  {
  std::vector<int> recvcounts(world_size), recvdisp(world_size);
  recvdisp[0] = 0;
  recvcounts[0] = H.row_extent(0);
  for( auto i = 1; i < world_size; ++i ) {
    auto rcnt = H.row_extent(i);
    recvcounts[i] = rcnt;
    recvdisp[i] = recvdisp[i-1] + recvcounts[i-1];
  }
  MPI_Gatherv( X_local.data(), X_local.size(), MPI_DOUBLE,
    X.data(), recvcounts.data(), recvdisp.data(), MPI_DOUBLE,
    0, MPI_COMM_WORLD );
  }

  double print_tol = 1e-2;
  if( world_rank == 0 ) {
    std::cout << "Psi0 Eigenvector (tol = " << print_tol << "):" << std::endl;
    for( auto i = 0; i < dets.size(); ++i ) 
    if( std::abs(X[i]) >= print_tol ) {
      std::cout << "  " << std::setw(5) << std::left << i << " " << std::setw(20) << std::right << X[i] << std::endl;
    }
  }
#else

  // Hamiltonian Matrix Element Generator
  dbwy::DoubleLoopHamiltonianGenerator<nbits> 
    ham_gen( norb, ints.u.data(), ints.t.data() );

  // Compute HF Energy
  const std::bitset<nbits> hf_det = 
    dbwy::canonical_hf_determinant<nbits>(nalpha,nbeta);
  const double EHF = ham_gen.matrix_element(hf_det, hf_det);
  if(world_rank == 0) {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "E(HF) = " << EHF + ints.core_energy << std::endl;
  }

#if 0
  // Compute CISD Energy
  double ECISD = 0.;
  if(world_rank == 0) {
    std::cout << "\nPerforming CISD Calculation" << std::endl;
  }
  {
    auto cisd_dets = dbwy::generate_cisd_hilbert_space<nbits>( norb_eff, hf_det );
    if(world_rank == 0) {
      std::cout << "NDETS_CISD = " << cisd_dets.size() << std::endl;
    }
    auto H_cisd = dbwy::make_dist_csr_hamiltonian<int32_t>(MPI_COMM_WORLD,
      cisd_dets.begin(), cisd_dets.end(), ham_gen, 1e-12 );
    ECISD = p_davidson( 100, H_cisd, 1e-8 );
  }
  if(world_rank == 0) {
    std::cout << "E(CISD)   = " << ECISD + ints.core_energy << std::endl;
    std::cout << "E_c(CISD) = " << ECISD - EHF << std::endl;
  }
#endif

#if 0
  // Compute Full CI Energy
  double EFCI = 0.;
  if(world_rank == 0) {
    std::cout << "\nPerforming FCI Calculation" << std::endl;
  }
  {
    auto fci_dets = build_hilbert_space<nbits>( norb_eff, nalpha, nbeta );
    if(world_rank == 0) {
      std::cout << "NDETS_FCI = " << fci_dets.size() << std::endl;
    }
    auto H_fci = dbwy::make_dist_csr_hamiltonian<int32_t>(MPI_COMM_WORLD,
      fci_dets.begin(), fci_dets.end(), ham_gen, 1e-12 );
    EFCI = p_davidson( 100, H_fci, 1e-8 );
  }
  if(world_rank == 0) {
    std::cout << "E(FCI)   = " << EFCI + ints.core_energy << std::endl;
    std::cout << "E_c(FCI) = " << EFCI - EHF << std::endl;
    std::cout << "E(FCI) - E(CISD) = " << (EFCI - ECISD) << std::endl;
  }
#endif


#if 1
  // ASCI
  size_t ndets_max = 100000;
  double metric_thresh = 1e-6;
  double coeff_thresh  = 1e-6;
  {

  auto bitset_comp = [](auto x, auto y){ return dbwy::bitset_less(x,y); };

  // First do CISD
  auto dets = dbwy::generate_cisd_hilbert_space<nbits>( norb, hf_det );
  std::sort(dets.begin(),dets.end(), bitset_comp );

  double EASCI = 0.;
  std::vector<double> X_local;
  {
    auto H = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD,
      dets.begin(), dets.end(), ham_gen, 1e-12 );
    
    X_local.resize( H.local_row_extent() );
    EASCI = p_davidson(100, H, 1e-8, X_local.data() );
  }
  if(world_rank == 0) {
    std::cout << "E(ASCI)   = " << EASCI + ints.core_energy << std::endl;
    std::cout << "E_c(ASCI) = " << EASCI - EHF << std::endl;
  }

  if(world_size != 1) throw "NO MPI";

  // Reorder the dets / coefficients
  {
  std::vector<uint32_t> idx( X_local.size() );
  std::iota( idx.begin(), idx.end(), 0 );
  std::sort(idx.begin(), idx.end(), [&](auto i, auto j) {
    return std::abs(X_local[i]) > std::abs(X_local[j]);
  });

  std::vector<double> tmp( X_local.size() );
  std::vector<std::bitset<nbits>> tmp_dets(dets.size());
  for( auto i = 0; i < dets.size(); ++i ) {
    tmp_dets[i] = dets[idx[i]];
    tmp[i]      = X_local[idx[i]];
  }

  dets = std::move(tmp_dets);
  X_local = std::move(tmp);
  }

  // Find det cutoff
  size_t nkeep = 0;
  {
    auto it = std::partition_point( X_local.begin(), X_local.end(),
      [&](auto x){ return std::abs(x) > coeff_thresh; } );
    nkeep = std::distance( X_local.begin(), it );
  }

  std::cout << "NKEEP COEFF = " << nkeep << std::endl;

  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;

  auto st = std::chrono::high_resolution_clock::now();
  // Loop over kept determinants and expand det space 
  std::vector<std::pair<std::bitset<nbits>,double>> singles_v;
  for( auto i = 0ul; i < nkeep; ++i ) {
    
    // Get occupied nad virtual indices
    auto state       = dets[i];
    auto state_alpha = dbwy::truncate_bitset<nbits/2>(state);
    auto state_beta  = dbwy::truncate_bitset<nbits/2>(state >> (nbits/2));
    auto coeff       = X_local[i];

    dbwy::bitset_to_occ_vir( norb, state_alpha, occ_alpha, vir_alpha ); 
    dbwy::bitset_to_occ_vir( norb, state_beta,  occ_beta,  vir_beta  ); 

    // Generate single excitations
    const std::bitset<nbits>   one   = 1ul;
    const std::bitset<nbits/2> one_h = 1ul;

    // Alpha
#if 0
    for( auto i : occ_alpha )
    for( auto a : vir_alpha ) {

      double h_el = ham_gen.T_pq_[a + i*norb];

      const double* G_red_ov = ham_gen.G_red_.data() + a*norb + i*norb*norb;
      const double* V_red_ov = ham_gen.V_red_.data() + a*norb + i*norb*norb;
      for( auto p : occ_alpha ) h_el += G_red_ov[p];
      for( auto p : occ_beta  ) h_el += V_red_ov[p];

      if( std::abs(h_el) < 1e-12 ) continue;

      // Get excited determinant
      auto ex = state ^ (one << i) ^ (one << a);

      double sign = dbwy::single_excitation_sign( state_alpha, a, i );
      h_el *= sign;

      singles_v.push_back( {ex, coeff*h_el} );

    }
#else
    dbwy::append_singles_asci_contributions<(nbits/2),0>( coeff, state, state_alpha,
      occ_alpha, vir_alpha, occ_beta, ham_gen.T_pq_, ham_gen.G_red_.data(),
      ham_gen.V_red_.data(), norb, 1e-12, singles_v );
#endif

    // Beta 
#if 0
    for( auto i : occ_beta )
    for( auto a : vir_beta ) {

      double h_el = ham_gen.T_pq_[a + i*norb];

      const double* G_red_ov = ham_gen.G_red_.data() + a*norb + i*norb*norb;
      const double* V_red_ov = ham_gen.V_red_.data() + a*norb + i*norb*norb;
      for( auto p : occ_beta  ) h_el += G_red_ov[p];
      for( auto p : occ_alpha ) h_el += V_red_ov[p];

      if( std::abs(h_el) < 1e-12 ) continue;

      // Get excited determinant
      auto ex = state ^ (((one << i) ^ (one << a)) << (nbits/2));

      double sign = dbwy::single_excitation_sign( state_beta, a, i );
      h_el *= sign;

      singles_v.push_back( {ex, coeff*h_el} );

    }
#else
    dbwy::append_singles_asci_contributions<(nbits/2),(nbits/2)>( coeff, state, 
      state_beta, occ_beta, vir_beta, occ_alpha, ham_gen.T_pq_, 
      ham_gen.G_red_.data(), ham_gen.V_red_.data(), norb, 1e-12, singles_v );
#endif

    // Doubles
    const size_t nocc = occ_alpha.size();
    const size_t nvir = vir_alpha.size();

    // All Alpha
#if 0
    for( auto ii = 0; ii < nocc; ++ii )
    for( auto aa = 0; aa < nvir; ++aa ) {
      const auto i = occ_alpha[ii];
      const auto a = vir_alpha[aa];
      const auto G_ai = ham_gen.G_pqrs_.data() + a + i*norb;

      for( auto jj = ii + 1; jj < nocc; ++jj )
      for( auto bb = aa + 1; bb < nvir; ++bb ) {
        const auto j = occ_alpha[jj];
        const auto b = vir_alpha[bb];
        const auto jb = b + j*norb;
        const auto G_aibj = G_ai[jb*norb*norb];

        if( std::abs(G_aibj) < 1e-12 ) continue;

        // Calculate excited determinant string (alpha)
        const auto full_ex_alpha = 
          (one_h << i) ^ (one_h << j) ^ (one_h << a) ^ (one_h << b);
        auto ex_det_alpha = state_alpha ^ full_ex_alpha;

        // Calculate the sign in a canonical way
        double sign = 1.;
        {
          auto ket = state_alpha;
          auto bra = ex_det_alpha;
          const auto _o1 = ham_gen.first_occ_flipped( ket, full_ex_alpha );
          const auto _v1 = ham_gen.first_occ_flipped( bra, full_ex_alpha );
          sign = ham_gen.single_ex_sign( ket, _v1, _o1 );

          ket ^= (one_h << _o1) ^ (one_h << _v1);
          const auto fx = bra ^ ket;
          const auto _o2 = ham_gen.first_occ_flipped( ket, fx );
          const auto _v2 = ham_gen.first_occ_flipped( bra, fx );
          sign *= ham_gen.single_ex_sign( ket, _v2, _o2 );
        }

        // Calculate full excited determinant
        const auto full_ex = dbwy::expand_bitset<nbits>(full_ex_alpha);
        auto ex_det = state ^ full_ex;

        // Update sign of matrix element
        auto h_el = sign * G_aibj;

#if 0
        auto ref = ham_gen.matrix_element(ex_det,state);
        if( std::abs( h_el - ref ) > 1e-14 )
          std::cout << "WRONG AAAA " << ref << ", " << h_el << std::endl;
#endif

        // Append {det, c*h_el}
        singles_v.push_back( {ex_det, coeff*h_el} );
      }
    }
#else
    dbwy::append_ss_doubles_asci_contributions<nbits/2,0>( coeff, state, 
      state_alpha, occ_alpha, vir_alpha, ham_gen.G_pqrs_.data(), norb,
      1e-12, singles_v);
#endif

    // All Beta 
#if 0
    for( auto ii = 0; ii < nocc; ++ii )
    for( auto aa = 0; aa < nvir; ++aa ) {
      const auto i = occ_beta[ii];
      const auto a = vir_beta[aa];
      const auto G_ai = ham_gen.G_pqrs_.data() + a + i*norb;
      for( auto jj = ii + 1; jj < nocc; ++jj )
      for( auto bb = aa + 1; bb < nvir; ++bb ) {
        const auto j = occ_beta[jj];
        const auto b = vir_beta[bb];
        const auto jb = b + j*norb;
        const auto G_aibj = G_ai[jb*norb*norb];

        if( std::abs(G_aibj) < 1e-12 ) continue;

        // Calculate excited determinant string (beta)
        const auto full_ex_beta = 
          (one_h << i) ^ (one_h << j) ^ (one_h << a) ^ (one_h << b);
        auto ex_det_beta = state_beta ^ full_ex_beta;

        // Calculate the sign in a canonical way
        double sign = 1.;
        {
          auto ket = state_beta;
          auto bra = ex_det_beta;
          const auto _o1 = ham_gen.first_occ_flipped( ket, full_ex_beta );
          const auto _v1 = ham_gen.first_occ_flipped( bra, full_ex_beta );
          sign = ham_gen.single_ex_sign( ket, _v1, _o1 );

          ket ^= (one_h << _o1) ^ (one_h << _v1);
          const auto fx = bra ^ ket;
          const auto _o2 = ham_gen.first_occ_flipped( ket, fx );
          const auto _v2 = ham_gen.first_occ_flipped( bra, fx );
          sign *= ham_gen.single_ex_sign( ket, _v2, _o2 );
        }

        // Calculate full excited determinant
        const auto full_ex = dbwy::expand_bitset<nbits>(full_ex_beta) << (nbits/2);
        auto ex_det = state ^ full_ex;

        // Update sign of matrix element
        auto h_el = sign * G_aibj;

#if 0
        auto ref = ham_gen.matrix_element(ex_det,state);
        if( std::abs( h_el - ref ) > 1e-14 )
          std::cout << "WRONG BBBB " << ref << ", " << h_el << std::endl;
#endif

        // Append {det, c*h_el}
        singles_v.push_back( {ex_det, coeff*h_el} );
      
      }
    }
#else
    dbwy::append_ss_doubles_asci_contributions<nbits/2,nbits/2>( coeff, state, 
      state_beta, occ_beta, vir_beta, ham_gen.G_pqrs_.data(), norb,
      1e-12, singles_v);
#endif

    // Mixed Alpha/Beta
#if 0
    for( auto i : occ_alpha )
    for( auto a : vir_alpha ) {
      const auto V_ai = ham_gen.V_pqrs_ + a + i*norb;

      double sign_alpha = ham_gen.single_ex_sign( state_alpha, a, i );
      for( auto j : occ_beta )
      for( auto b : vir_beta ) {
        const auto jb = b + j*norb;
        const auto V_aibj = V_ai[jb*norb*norb];

        if( std::abs(V_aibj) < 1e-12 ) continue;

        double sign_beta = ham_gen.single_ex_sign( state_beta,  b, j );
        double sign = sign_alpha * sign_beta;
        auto ex_det = state ^ (one << i) ^ (one << a) ^
                            (((one << j) ^ (one << b)) << (nbits/2));
        auto h_el = sign * V_aibj;
#if 0
        auto ref = ham_gen.matrix_element(ex_det,state);
        if( std::abs( h_el - ref ) > 1e-14 )
          std::cout << "WRONG AABB " << ref << ", " << h_el << std::endl;
#endif
        singles_v.push_back( {ex_det, coeff*h_el} );
      }
    }
#else
    dbwy::append_os_doubles_asci_contributions( coeff, state, state_alpha,
      state_beta, occ_alpha, occ_beta, vir_alpha, vir_beta, ham_gen.V_pqrs_,
      norb, 1e-12, singles_v );
#endif
      
  }

  std::cout << singles_v.size() << std::endl;
  std::sort( singles_v.begin(), singles_v.end(), 
    []( auto x, auto y ) {
      return dbwy::bitset_less(x.first, y.first);
    });

  
  auto cur_it = singles_v.begin();
  for( auto it = singles_v.begin() + 1; it != singles_v.end(); ++it ) {
    if( it->first != cur_it->first ) {
      cur_it = it;
    } else {
      cur_it->second += it->second;
      it->second = 0;
    }
  }

  auto uit = std::unique( singles_v.begin(), singles_v.end(), 
    []( auto x, auto y ) {
      return x.first == y.first;
    });
  singles_v.erase(uit,singles_v.end());
  std::cout << "UNIQ = " << singles_v.size() << std::endl;

  for( auto i = 0; i < singles_v.size(); ++i ) {
    auto det = singles_v[i].first;
    auto diag_element = ham_gen.matrix_element(det,det);
    singles_v[i].second /= EASCI - diag_element;
  }

  std::sort( singles_v.begin(), singles_v.end(), 
  [](auto x, auto y){ return std::abs(x.second) > std::abs(y.second); });

  singles_v.erase(singles_v.begin() + ndets_max, singles_v.end());
  singles_v.shrink_to_fit();

  auto en = std::chrono::high_resolution_clock::now();
  std::cout << "DUR  = " << std::chrono::duration<double>(en-st).count() << std::endl;

  for( auto [x,y] : singles_v ) {
    std::cout << x << ", " << y << ", " << std::endl;
  }
  // Compute new energy



#if 0

  // Compute weights of determinants already in space

  // Compute Y = H * X = E * X
  std::vector<double> Y_local( X_local );
  std::transform( Y_local.begin(), Y_local.end(), Y_local.begin(), 
    [=](auto x) { return ESCI*x; });

  // Compute A(i) = H(i,j) * X(j) / (H(i,i) - E)
  //              = Y(i) / (H(i,i) - E)
  for( auto i = 0; i < dets.size(); ++i ) {
    auto d = dets[i];
    Y_local[i] = Y_local[i] / ( ham_gen.matrix_element(d,d) - ESCI );
  }

#if 0
  for( auto i = 0; i < dets.size(); ++i ) {
    std::cout << i << ", " << X_local[i] << ", " << Y_local[i] << std::endl;
  }
#endif

#endif


#if 0
  // Expand Search Space
  std::vector<std::bitset<nbits>> new_dets, sd_i;
  for( auto i = 1; i < dets.size(); ++i ) {

    // Get all SD states connects to D[i]
    dbwy::generate_cisd_hilbert_space<nbits>( norb, dets[i], sd_i );

  }
#endif


#if 0
  // Get doubly connected states
  std::vector<std::bitset<nbits>> new_dets;
  for( auto i = 0; i < dets.size(); ++i ) {
    auto new_dets_i = dbwy::generate_cisd_hilbert_space<nbits>( norb, dets[i] );
    new_dets.insert(new_dets.end(), new_dets_i.begin()+1, new_dets_i.end());
  }

  std::sort(new_dets.begin(), new_dets.end(),
    [](auto x, auto y){ return dbwy::bitset_less(x,y); });
  {
    auto it = std::unique(new_dets.begin(), new_dets.end());
    new_dets.erase(it, new_dets.end());
    new_dets.shrink_to_fit();
  }

  
  {
    std::vector<std::bitset<nbits>> uniq_dets;
    std::set_difference( 
      new_dets.begin(), new_dets.end(),
      dets.begin(), dets.end(),
      std::back_inserter(uniq_dets),
      [](auto x, auto y){ return dbwy::bitset_less(x,y); }
    );
    new_dets = std::move( uniq_dets );
  }

  std::cout << "NEW_NDETS = " << new_dets.size() << std::endl; 

  // Compute PT2 Contributions
  std::vector<double> PT2(new_dets.size());
  for( auto i = 0; i < new_dets.size(); ++i ) {
    double numerator = 0.;
    for( auto j = 0; j < dets.size(); ++j )
      numerator += ham_gen.matrix_element(dets[j], new_dets[i]) * X_local[j];
    numerator *= numerator;

    PT2[i] = numerator / (ESCI - ham_gen.matrix_element(new_dets[i],new_dets[i]));
  }

  std::vector<uint32_t> new_det_indices( new_dets.size() );
  std::iota( new_det_indices.begin(), new_det_indices.end(), 0 );
  std::sort( new_det_indices.begin(), new_det_indices.end(), [&](auto i, auto j) {
    return std::abs(PT2[i]) > std::abs(PT2[j]);
  });

  for( auto i : new_det_indices ) {
    if( dets.size() > ndets_max ) break;
    if( std::abs(PT2[i]) < pt2_thresh ) break;

    dets.emplace_back( new_dets[i] );
  }
  std::sort(dets.begin(),dets.end(),
    [](auto x, auto y){ return dbwy::bitset_less(x,y); });


  auto H_new = dbwy::make_dist_csr_hamiltonian<int32_t>( MPI_COMM_WORLD,
    dets.begin(), dets.end(), ham_gen, 1e-12 );
  
  ESCI = p_davidson(100, H_new, 1e-8, nullptr );

  std::cout << "E(SCI)   = " << ESCI  + ints.core_energy << std::endl;
  std::cout << "E_c(SCI) = " << ESCI - EHF << std::endl;
  //std::cout << "E(FCI) - E(SCI) = " << (EFCI - ESCI) << std::endl;
#endif
  } // CIPSI
#endif

#endif

  } // MPI Scope
  MPI_Finalize();

}
