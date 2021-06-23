#include "cmz_ed/hamil.h++"

namespace cmz
{
  namespace ed
  {

    
    bool CompTripltByVal(const Eigen::Triplet<double> &t1, const Eigen::Triplet<double> &t2)
    {
      return abs(t1.value()) > abs(t2.value());
    }

    double FermionHamil::GetHmatel( const slater_det &L_st, const slater_det &R_st ) const
    {
      // |R_st> defines occupied orbitals in excitation,
      // |L_st> the virtual orbitals.
      double res = 0.;

      // Copy Slater dets, for manipulations
      slater_det bra(L_st), ket(R_st);
      // Store occupied orbitals by spin
      std::vector<unsigned short> occs_up = ket.GetOccOrbsUp();
      std::vector<unsigned short> occs_do = ket.GetOccOrbsDo();
      // First, determine number of excitations between both determinants.
      unsigned short exc_up = bra.CountExcUp( ket );
      unsigned short exc_do = bra.CountExcDo( ket );
      // Double up excitation
      if( exc_up == 4 && exc_do == 0 )
      {
        unsigned short o1 = ket.GetFlippedOccIndx( bra );
        unsigned short v1 = bra.GetFlippedOccIndx( ket );
        double sign = ket.SingleExcUp( v1, o1 );
        unsigned short o2 = ket.GetFlippedOccIndx( bra );
        unsigned short v2 = bra.GetFlippedOccIndx( ket );
        sign *= ket.SingleExcUp( v2, o2 );
        res = sign * (pints->getChem(v1,o1,v2,o2) 
                      - pints->getChem(v1,o2,v2,o1));
      }
      // Double down excitation
      else if( exc_up == 0 && exc_do == 4 )
      {
        unsigned short o1 = ket.GetFlippedOccIndx( bra );
        unsigned short v1 = bra.GetFlippedOccIndx( ket );
        double sign = ket.SingleExcDo( v1, o1 );
        unsigned short o2 = ket.GetFlippedOccIndx( bra );
        unsigned short v2 = bra.GetFlippedOccIndx( ket );
        sign *= ket.SingleExcDo( v2, o2 );
        res = sign * (pints->getChem(v1,o1,v2,o2) 
                      - pints->getChem(v1,o2,v2,o1));
      }
      // Mixed double excitation
      else if( exc_up == 2 && exc_do == 2 )
      {
        unsigned short oup = ket.GetFlippedOccIndxUp( bra );
        unsigned short vup = bra.GetFlippedOccIndxUp( ket );
        double sign = ket.SingleExcUp( vup, oup );
        unsigned short odo = ket.GetFlippedOccIndxDo( bra );
        unsigned short vdo = bra.GetFlippedOccIndxDo( ket );
        sign *= ket.SingleExcDo( vdo, odo );
        res = sign * pints->getChem(vup,oup,vdo,odo);
      }
      // Single up excitation
      else if( exc_up == 2 && exc_do == 0)
      {
        unsigned short o1 = ket.GetFlippedOccIndx( bra );
        unsigned short v1 = bra.GetFlippedOccIndx( ket );
        double sign = ket.SingleExcUp( v1, o1 );

        res = pints->get(v1, o1);
        for( auto b : occs_up )
          res += pints->getChem(v1, o1, b, b)
                 - pints->getChem(v1, b, b, o1);
        for( auto b : occs_do )
          res += pints->getChem(v1, o1, b, b);
        res *= sign;
      }
      // Single do excitation
      else if( exc_up == 0 && exc_do == 2 )
      {
        unsigned short o1 = ket.GetFlippedOccIndx( bra );
        unsigned short v1 = bra.GetFlippedOccIndx( ket );
        double sign = ket.SingleExcDo( v1, o1 );

        res = pints->get(v1, o1);
        for( auto b : occs_up )
          res += pints->getChem(v1, o1, b, b);
        for( auto b : occs_do )
          res += pints->getChem(v1, o1, b, b)
                 - pints->getChem(v1, b, b, o1);
        res *= sign;
      }
      // Diagonal term
      else if( exc_up == 0 && exc_do == 0 )
      {
        // One-body terms
        for( auto o : occs_up )
          res += pints->get(o,o);
        for( auto o : occs_do )
          res += pints->get(o,o);

        // Two-body terms
        // Up-Up
        for( auto o1 : occs_up )
          for( auto o2 : occs_up )
            res += 0.5 * (pints->getChem(o1,o1,o2,o2)
                          - pints->getChem(o1,o2,o2,o1));
        // Down-Down
        for( auto o1 : occs_do )
          for( auto o2 : occs_do )
            res += 0.5 * (pints->getChem(o1,o1,o2,o2)
                          - pints->getChem(o1,o2,o2,o1));
        // Up-Down
        for( auto o1 : occs_up )
          for( auto o2 : occs_do )
            res += pints->getChem(o1,o1,o2,o2);
      }
      
      return res;
    }
    
    std::vector<std::pair<size_t, size_t> > FermionHamil::GetHpairs( const SetSlaterDets &stts ) const
    {
      // Returns a list of pairs of indices, indicating
      // the pairs of states in stts that have
      // non-vanishing Hamiltonian matrix elements.
      std::vector<std::pair<size_t, size_t> > pairs;
    
      // Iterate over set, and find possible pairs
      for( SetSlaterDets_It curr_it = stts.begin(); curr_it != stts.end(); curr_it++ )
      {
        // Essentially, go through all electrons and pairs,
        // proposing single and double excitations.
        size_t curr_indx = std::distance( stts.cbegin(), curr_it );
        std::vector<slater_det> s_and_ds;
        s_and_ds = curr_it->GetSinglesAndDoubles( pints );
        for( auto exc_det : s_and_ds )
        {
          SetSlaterDets_It part_it = stts.find( exc_det );
          if( part_it != stts.end() )
          {
            size_t part_indx = std::distance( stts.cbegin(), part_it );
            pairs.push_back( std::pair<size_t, size_t>(curr_indx, part_indx) );
          }
        }
      }
      return pairs;
    }
    
    SpMatD GetHmat( const FermionHamil* H, const SetSlaterDets & stts, bool print ) 
    {
      // Return sparse matrix of fermionic Hamiltonian in
      // the basis of the states stored in stts.
      size_t nelems = stts.size();
      SpMatD Hmat( nelems, nelems );
    
      // Find pairs of states with non-vanishing 
      // matrix elements.
      std::vector<std::pair<size_t, size_t> > pairs;
      pairs = H->GetHpairs( stts );
     
      if( print )
        std::cout << "Possible NNZ pairs: " << pairs.size() + nelems << ", worst sparsity : " << double(pairs.size() + nelems) / double(nelems * nelems) * 100. << "%" << std::endl;
    
      // Run over pairs, computing matrix elements
      VecT tripletList( pairs.size() + nelems, T(0,0,0.) );
      #pragma omp parallel for
      for( size_t ipair = 0; ipair < pairs.size(); ipair++ )
      {
        size_t l_indx = pairs[ ipair ].first;
        size_t r_indx = pairs[ ipair ].second;
        SetSlaterDets_It l_it = std::next( stts.begin(), l_indx );
        SetSlaterDets_It r_it = std::next( stts.begin(), r_indx );
        double Hmatel = H->GetHmatel( *l_it, *r_it );
        tripletList[ ipair ] = T(l_indx, r_indx, Hmatel);
      }
      // Prune list from matrix elements below
      // 1.E-9 
      std::sort(tripletList.begin(), 
                tripletList.begin() + pairs.size(), 
                CompTripltByVal);
      T thres(0,0,1.E-9);
      std::vector<T>::iterator todelete = std::upper_bound(tripletList.begin(), 
                                                           tripletList.begin() + pairs.size(), 
                                                           thres, CompTripltByVal);
      size_t n_off_diags = todelete - tripletList.begin();
      tripletList.erase(todelete, 
                        tripletList.begin() + pairs.size());
      // Run over diagonal elements
      #pragma omp parallel for
      for( size_t ist = 0; ist < stts.size(); ist++ )
      {
        SetSlaterDets_It it = std::next( stts.begin(), ist );
        double Hmatel = H->GetHmatel( *it, *it );
        tripletList[ ist + n_off_diags ] = T(ist, ist, Hmatel);
      }
      // Prune list from matrix elements below
      // 1.E-9 
      std::sort(tripletList.begin() + n_off_diags, 
                tripletList.begin() + n_off_diags + stts.size(), 
                CompTripltByVal);
      todelete = std::upper_bound(tripletList.begin() + n_off_diags, 
                                  tripletList.begin() + n_off_diags + stts.size(), 
                                  thres, CompTripltByVal);
      tripletList.erase(todelete, 
                        tripletList.begin() + n_off_diags + stts.size());
    
      // Build sparse matrix
      Hmat.setFromTriplets( tripletList.begin(), tripletList.end() );
      Hmat.makeCompressed();
      if( print )
        std::cout << "Hamiltonian NNZ   : " << Hmat.nonZeros() << ", actual sparsity: " << double(Hmat.nonZeros()) / double(nelems * nelems) * 100. << "%" << std::endl;
      return Hmat;
    }
    
    SpMatD GetHmat_FromPairs( const FermionHamil* H, const SetSlaterDets & stts, const std::vector<std::pair<size_t, size_t> > &pairs ) 
    {
      // Return sparse matrix with FA Hamiltonian in
      // the basis of the states stored in stts, 
      // considering only as off-diagonals the
      // pairs of states given in pairs.
      size_t nelems = stts.size();
      SpMatD Hmat( nelems, nelems );
     
      // Run over pairs, computing matrix elements
      VecT tripletList( pairs.size() + nelems, T(0,0,0.) );
      #pragma omp parallel for
      for( size_t ipair = 0; ipair < pairs.size(); ipair++ )
      {
        size_t l_indx = pairs[ ipair ].first;
        size_t r_indx = pairs[ ipair ].second;
        SetSlaterDets_It l_it = std::next( stts.begin(), l_indx );
        SetSlaterDets_It r_it = std::next( stts.begin(), r_indx );
        double Hmatel = H->GetHmatel( *l_it, *r_it );
        tripletList[ ipair ] = T(l_indx, r_indx, Hmatel);
      }
      // Prune list from matrix elements below
      // 1.E-9 
      std::sort(tripletList.begin(), 
                tripletList.begin() + pairs.size(), 
                CompTripltByVal);
      T thres(0,0,1.E-9);
      std::vector<T>::iterator todelete = std::upper_bound(tripletList.begin(), 
                                                           tripletList.begin() + pairs.size(), 
                                                           thres, CompTripltByVal);
      size_t n_off_diags = todelete - tripletList.begin();
      tripletList.erase(todelete, 
                        tripletList.begin() + pairs.size());
    
      // Run over diagonal elements
      #pragma omp parallel for
      for( size_t ist = 0; ist < stts.size(); ist++ )
      {
        SetSlaterDets_It it = std::next( stts.begin(), ist );
        double Hmatel = H->GetHmatel( *it, *it );
        tripletList[ ist + n_off_diags ] = T(ist, ist, Hmatel);
      }
      // Prune list from matrix elements below
      // 1.E-9 
      std::sort(tripletList.begin() + n_off_diags, 
                tripletList.begin() + n_off_diags + stts.size(), 
                CompTripltByVal);
      todelete = std::upper_bound(tripletList.begin() + n_off_diags, 
                                  tripletList.begin() + n_off_diags + stts.size(), 
                                  thres, CompTripltByVal);
      tripletList.erase(todelete, 
                        tripletList.begin() + n_off_diags + stts.size());
    
      // Build sparse matrix
      Hmat.setFromTriplets( tripletList.begin(), tripletList.end() );
      Hmat.makeCompressed();
      std::cout << "Hamiltonian NNZ   : " << Hmat.nonZeros() << ", actual sparsity: " << double(Hmat.nonZeros()) / double(nelems * nelems) * 100. << "%" << std::endl;
      return Hmat;
    }

  }// namespace ed
}// namespace cmz
