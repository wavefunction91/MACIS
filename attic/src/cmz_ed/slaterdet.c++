#include "cmz_ed/slaterdet.h++"

namespace cmz
{
  namespace ed
  {

    double slater_det::SingleExcUp( uint64_t i, uint64_t a ) 
    { 
      uint64_t omax = std::max(i,a);
      uint64_t omin = std::min(i,a);
      // Bitmask to check for Fermionic sign change.
      uint64_t mask = ((1<<omax) - 1) ^ ((1<<omin+1) - 1);
      mask = state & mask;
      Flip(i); Flip(a);
      return (std::popcount(mask) % 2) == 1 ? -1. : 1.; 
    }

    double slater_det::SingleExcDo( uint64_t i, uint64_t a ) 
    { 
      uint64_t omax = std::max(i,a) + Norbs;
      uint64_t omin = std::min(i,a) + Norbs;
      // Bitmask to check for Fermionic sign change.
      uint64_t mask = ((1<<omax) - 1) ^ ((1<<omin+1) - 1);
      mask = state & mask;
      Flip(i+Norbs); Flip(a+Norbs);
      return (std::popcount(mask) % 2) == 1 ? -1. : 1.; 
    }

    slater_det slater_det::GetSingExUpSt( uint64_t i, uint64_t a, double &sign ) const
    {
      slater_det cp(*this);
      sign = cp.SingleExcUp( i, a );
      return cp;
    }

    slater_det slater_det::GetSingExDoSt( uint64_t i, uint64_t a, double &sign ) const
    {
      slater_det cp(*this);
      sign = cp.SingleExcDo( i, a );
      return cp;
    }

    string slater_det::ToStr( ) const
    {
      string str = "|";
      for( short i = 0; i < Norbs; i++ )
      {
        bool up = IsOccUp( i );
        bool dn = IsOccDo( i );
        if(up && dn)
          str += "2";
        else if(up)
          str += "u";
        else if(dn)
          str += "d";
        else
          str += "0";
      }
      str += ">";
      return str;
    }

    string slater_det::ToStrBra( ) const
    {
      string str = "<";
      for( short i = 0; i < Norbs; i++ )
      {
        bool up = IsOccUp( i );
        bool dn = IsOccDo( i );
        if(up && dn)
          str += "2";
        else if(up)
          str += "u";
        else if(dn)
          str += "d";
        else
          str += "0";
      }
      str += "|";
      return str;
    }

    std::vector<uint64_t> slater_det::GetOccOrbsUp() const
    {
      std::vector<uint64_t> occs_up(Nup, 0);
      uint64_t indx = 0;
      for(uint64_t i = 0; i < Norbs; i++)
        if( IsOccUp(i) )
        {
//	  std::cout << 'checking up indx ' << indx << std::endl;
          occs_up[indx] = i; indx++;
        }
      return occs_up;
    }

    std::vector<uint64_t> slater_det::GetOccOrbsDo() const
    {
      std::vector<uint64_t> occs_do(Ndo, 0);
      uint64_t indx = 0;
      for(uint64_t i = 0; i < Norbs; i++)
      {
//	std::cout << 'checking down ' << i << std::endl;
        if( IsOccDo(i) )
        {
//	  std::cout << 'checking indx ' << indx << std::endl;
          occs_do[indx] = i; indx++;
        }
      }
      return occs_do;
    }

    void slater_det::GetOccsAndVirtsUp( std::vector<uint64_t> &occs, std::vector<uint64_t> &virts ) const
    {
      occs.resize(Nup, 0); virts.resize(Norbs - Nup, 0);
      uint64_t iocc = 0, ivirt = 0;
      for(uint64_t i = 0; i < Norbs; i++)
      {
        if( IsOccUp(i) )
        {
          occs[iocc] = i; iocc++;
        }
        else
        {
          virts[ivirt] = i; ivirt++;
        }
      }
    }

    void slater_det::GetOccsAndVirtsDo( std::vector<uint64_t> &occs, std::vector<uint64_t> &virts ) const
    {
      occs.resize(Ndo, 0); virts.resize(Norbs - Ndo, 0);
      uint64_t iocc = 0, ivirt = 0;
      for(uint64_t i = 0; i < Norbs; i++)
      {
        if( IsOccDo(i) )
        {
          occs[iocc] = i; iocc++;
        }
        else
        {
          virts[ivirt] = i; ivirt++;
        }
      }
    }

    std::vector<slater_det> slater_det::GetSinglesAndDoubles() const
    {
      std::vector<slater_det> excs;
      // Store occupied and empty orbitals of each spin
      std::vector<uint64_t> occs_up, occs_do;
      std::vector<uint64_t> virts_up, virts_do;
      GetOccsAndVirtsUp( occs_up, virts_up ); 
      GetOccsAndVirtsDo( occs_do, virts_do ); 

      double sign = 1.; // For function calling purposes
      // Build single excitation.
      // Single - Up
      for( auto ou : occs_up )
        for( auto vu : virts_up )
          excs.push_back( GetSingExUpSt( vu, ou, sign ) );
      // Single - Down
      for( auto od : occs_do )
        for( auto vd : virts_do )
          excs.push_back( GetSingExDoSt( vd, od, sign ) );
 
      // Build double excitations.
      // Double - Up/Up
      for( size_t io1 = 0; io1 < occs_up.size(); io1++ )
        for( size_t iv1 = 0; iv1 < virts_up.size(); iv1++ )
        {
          slater_det tmp = GetSingExUpSt( virts_up[iv1], occs_up[io1], sign );
          for( size_t io2 = io1+1; io2 < occs_up.size(); io2++ )
            for( size_t iv2 = iv1+1; iv2 < virts_up.size(); iv2++ )
              excs.push_back( tmp.GetSingExUpSt( virts_up[iv2], occs_up[io2], sign ) );
         }
      // Double - Do/Do
      for( size_t io1 = 0; io1 < occs_do.size(); io1++ )
        for( size_t iv1 = 0; iv1 < virts_do.size(); iv1++ )
        {
          slater_det tmp = GetSingExDoSt( virts_do[iv1], occs_do[io1], sign );
          for( size_t io2 = io1+1; io2 < occs_do.size(); io2++ )
            for( size_t iv2 = iv1+1; iv2 < virts_do.size(); iv2++ )
              excs.push_back( tmp.GetSingExDoSt( virts_do[iv2], occs_do[io2], sign ) );
         }
      // Double - Up/Do
      for( size_t io1 = 0; io1 < occs_up.size(); io1++ )
        for( size_t iv1 = 0; iv1 < virts_up.size(); iv1++ )
        {
          slater_det tmp = GetSingExUpSt( virts_up[iv1], occs_up[io1], sign );
          for( size_t io2 = 0; io2 < occs_do.size(); io2++ )
            for( size_t iv2 = 0; iv2 < virts_do.size(); iv2++ )
              excs.push_back( tmp.GetSingExDoSt( virts_do[iv2], occs_do[io2], sign ) );
         }

      // Sort list and delete duplicates
      std::sort( excs.begin(), excs.end() );
      std::vector<slater_det>::iterator it;
      it = std::unique( excs.begin(), excs.end() );
      excs.resize( std::distance(excs.begin(),it) );

      return excs;
    }

    std::vector<slater_det> slater_det::GetSinglesAndDoubles( const intgrls::integrals *pint ) const
    {
      std::vector<slater_det> excs;
      // Store occupied and empty orbitals of each spin
      std::vector<uint64_t> occs_up, occs_do;
      std::vector<uint64_t> virts_up, virts_do;
      GetOccsAndVirtsUp( occs_up, virts_up ); 
      GetOccsAndVirtsDo( occs_do, virts_do ); 
  
      double thres = 1.E-9; //Threshold to keep excitation.

      double sign = 1.; // For function calling purposes
      // Build single excitation.
      // Single - Up
      for( auto ou : occs_up )
        for( auto vu : virts_up )
          if( std::abs(pint->get(vu, ou)) >= thres )
            excs.push_back( GetSingExUpSt( vu, ou, sign ) );
      // Single - Down
      for( auto od : occs_do )
        for( auto vd : virts_do )
          if( std::abs(pint->get(vd , od)) >= thres )
            excs.push_back( GetSingExDoSt( vd, od, sign ) );
 
      // Build double excitations.
      // Double - Up/Up
      for( size_t io1 = 0; io1 < occs_up.size(); io1++ )
        for( size_t iv1 = 0; iv1 < virts_up.size(); iv1++ )
        {
          slater_det tmp = GetSingExUpSt( virts_up[iv1], occs_up[io1], sign );
          for( size_t io2 = io1+1; io2 < occs_up.size(); io2++ )
            for( size_t iv2 = iv1+1; iv2 < virts_up.size(); iv2++ )
              if( std::abs(pint->getChem(virts_up[iv1], virts_up[iv2], occs_up[io1], occs_up[io2])) >= thres 
               || std::abs(pint->getChem(virts_up[iv1], virts_up[iv2], occs_up[io2], occs_up[io1])) >= thres )
                excs.push_back( tmp.GetSingExUpSt( virts_up[iv2], occs_up[io2], sign ) );
         }
      // Double - Do/Do
      for( size_t io1 = 0; io1 < occs_do.size(); io1++ )
        for( size_t iv1 = 0; iv1 < virts_do.size(); iv1++ )
        {
          slater_det tmp = GetSingExDoSt( virts_do[iv1], occs_do[io1], sign );
          for( size_t io2 = io1+1; io2 < occs_do.size(); io2++ )
            for( size_t iv2 = iv1+1; iv2 < virts_do.size(); iv2++ )
              if( std::abs(pint->getChem(virts_do[iv1], virts_do[iv2], occs_do[io1], occs_do[io2])) >= thres 
               || std::abs(pint->getChem(virts_do[iv1], virts_do[iv2], occs_do[io2], occs_do[io1])) >= thres )
                excs.push_back( tmp.GetSingExDoSt( virts_do[iv2], occs_do[io2], sign ) );
         }
      // Double - Up/Do
      for( size_t io1 = 0; io1 < occs_up.size(); io1++ )
        for( size_t iv1 = 0; iv1 < virts_up.size(); iv1++ )
        {
          slater_det tmp = GetSingExUpSt( virts_up[iv1], occs_up[io1], sign );
          for( size_t io2 = 0; io2 < occs_do.size(); io2++ )
            for( size_t iv2 = 0; iv2 < virts_do.size(); iv2++ )
              if( std::abs(pint->getChem(virts_up[iv1], virts_do[iv2], occs_up[io1], occs_do[io2])) >= thres )
                excs.push_back( tmp.GetSingExDoSt( virts_do[iv2], occs_do[io2], sign ) );
         }

      // Sort list and delete duplicates
      std::sort( excs.begin(), excs.end() );
      std::vector<slater_det>::iterator it;
      it = std::unique( excs.begin(), excs.end() );
      excs.resize( std::distance(excs.begin(),it) );

      return excs;
    }


    SetSlaterDets BuildShiftHilbertSpace( uint64_t Norbs,uint64_t Neff, uint64_t Nups, uint64_t Ndos )
    {
      // Builds full Hilbert space for a system of Norbs
      // orbitals, Nups up electrons and Ndos down electrons.
      // Returns corresponding set of Slater determinants
      SetSlaterDets basis;
     
      // Make all possible states of Norbs with Nups and Ndos
      // bits set.
      std::vector<uint64_t> up_stts = BuildCombs( Neff, Nups );
      std::vector<uint64_t> do_stts = BuildCombs( Neff, Ndos );
      // Combine into possible Hilbert space
      for( size_t iup = 0; iup < up_stts.size(); iup++ )
        for( size_t ido = 0; ido < do_stts.size(); ido++ )
        {
          uint64_t st = (do_stts[ido] << Norbs) + up_stts[iup];
//	  cout << "adding in " << st << endl;
          basis.insert( slater_det( st, Norbs, Nups, Ndos ) );
        }

      return basis;
    }


    SetSlaterDets BuildFullHilbertSpace( uint64_t Norbs, uint64_t Nups, uint64_t Ndos )
    {
      // Builds full Hilbert space for a system of Norbs
      // orbitals, Nups up electrons and Ndos down electrons.
      // Returns corresponding set of Slater determinants
      SetSlaterDets basis;
     
      // Make all possible states of Norbs with Nups and Ndos
      // bits set.
      std::vector<uint64_t> up_stts = BuildCombs( Norbs, Nups );
      std::vector<uint64_t> do_stts = BuildCombs( Norbs, Ndos );

      // Combine into possible Hilbert space
      for( size_t iup = 0; iup < up_stts.size(); iup++ )
        for( size_t ido = 0; ido < do_stts.size(); ido++ )
        {
          uint64_t st = (do_stts[ido] << Norbs) + up_stts[iup];
          basis.insert( slater_det( st, Norbs, Nups, Ndos ) );
        }

      return basis;
    }

    std::vector<std::pair<size_t, size_t> > GetSingDoublPairs( const SetSlaterDets &stts ) 
    {
      // Returns a list of pairs of indices, indicating
      // the pairs of states in stts that are connected
      // by either a single or double excitation.
      std::vector<std::pair<size_t, size_t> > pairs;
    
      // Iterate over set, and find possible pairs
      for( SetSlaterDets_It curr_it = stts.begin(); curr_it != stts.end(); curr_it++ )
      {
        // Essentially, go through all electrons and pairs,
        // proposing single and double excitations.
        size_t curr_indx = std::distance( stts.cbegin(), curr_it );
        std::vector<slater_det> s_and_ds;
        s_and_ds = curr_it->GetSinglesAndDoubles();
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
    
  }// namespace ed
}// namespace cmz
