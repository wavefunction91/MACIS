#include "cmz_ed/rdms.h++"

namespace cmz
{
  namespace ed
  {
    namespace rdm
    {

      rdms::rdms( int norbitals, const VectorXd &vec, const SetSlaterDets &stts ) : rdms(norbitals)
      {
        // Constructure using a many-Fermion state and Slater
        // determinant list.
        // Get lists of pairs with possible non-zero rdm contributions.
        std::vector<std::pair<size_t, size_t> > pairs;
        pairs = GetSingDoublPairs( stts );
        // Go over pairs, filling out rdms
        for( size_t ipair = 0; ipair < pairs.size(); ipair++ )
        {
          size_t l_indx = pairs[ ipair ].first;
          size_t r_indx = pairs[ ipair ].second;
          double contrib = vec(l_indx) * vec(r_indx);
          SetSlaterDets_It l_it = std::next( stts.begin(), l_indx );
          SetSlaterDets_It r_it = std::next( stts.begin(), r_indx );
          slater_det bra(*l_it);
          slater_det ket(*r_it);
          // Store occupied orbitals by spin
          std::vector<unsigned short> occs_up = ket.GetOccOrbsUp();
          std::vector<unsigned short> occs_do = ket.GetOccOrbsDo();
          // Determine number of excitations between both determinants.
          unsigned short exc_up = bra.CountExcUp( ket );
          unsigned short exc_do = bra.CountExcDo( ket );
          // Double Up-Up
          if( exc_up == 4 && exc_do == 0 )
          {
            unsigned short o1 = ket.GetFlippedOccIndx( bra );
            unsigned short v1 = bra.GetFlippedOccIndx( ket );
            double sign = bra.SingleExcUp( o1, v1 );
            sign = ket.SingleExcUp( v1, o1 );
            unsigned short o2 = ket.GetFlippedOccIndx( bra );
            unsigned short v2 = bra.GetFlippedOccIndx( ket );
            sign *= ket.SingleExcUp( v2, o2 );
            rdm2[ indexer(v1,v2,o1,o2) ] += sign * contrib * 0.5;
            rdm2[ indexer(v1,v2,o2,o1) ] -= sign * contrib * 0.5;
            rdm2[ indexer(v2,v1,o2,o1) ] += sign * contrib * 0.5;
            rdm2[ indexer(v2,v1,o1,o2) ] -= sign * contrib * 0.5;
          }
          // Double Do-Do
          else if( exc_do == 4 && exc_do == 0 )
          {
            unsigned short o1 = ket.GetFlippedOccIndx( bra );
            unsigned short v1 = bra.GetFlippedOccIndx( ket );
            double sign = bra.SingleExcDo( o1, v1 );
            sign = ket.SingleExcDo( v1, o1 );
            unsigned short o2 = ket.GetFlippedOccIndx( bra );
            unsigned short v2 = bra.GetFlippedOccIndx( ket );
            sign *= ket.SingleExcDo( v2, o2 );
            rdm2[ indexer(v1,v2,o1,o2) ] += sign * contrib * 0.5;
            rdm2[ indexer(v1,v2,o2,o1) ] -= sign * contrib * 0.5;
            rdm2[ indexer(v2,v1,o2,o1) ] += sign * contrib * 0.5;
            rdm2[ indexer(v2,v1,o1,o2) ] -= sign * contrib * 0.5;
          }
          // Double Up-Do
          else if( exc_up == 2 && exc_do == 2 )
          {
            unsigned short oup = ket.GetFlippedOccIndxUp( bra );
            unsigned short vup = bra.GetFlippedOccIndxUp( ket );
            double sign = bra.SingleExcUp( oup, vup );
            sign = ket.SingleExcUp( vup, oup );
            unsigned short odo = ket.GetFlippedOccIndxDo( bra );
            unsigned short vdo = bra.GetFlippedOccIndxDo( ket );
            sign *= ket.SingleExcDo( vdo, odo );
            rdm2[ indexer(vup,vdo,oup,odo) ] += sign * contrib * 0.5;
            rdm2[ indexer(vdo,vup,odo,oup) ] += sign * contrib * 0.5;
          }
          // Single Up
          else if( exc_up == 2 && exc_do == 0 )
          {
            unsigned short o1 = ket.GetFlippedOccIndx( bra );
            unsigned short v1 = bra.GetFlippedOccIndx( ket );
            double sign = ket.SingleExcUp( v1, o1 );

            // One-body contribution
            rdm1[ indexer(v1, o1) ] += sign * contrib;
            // Two-body contribution
            // Up-Up
            for( auto b : occs_up )
            {
              rdm2[ indexer(v1, b, o1, b) ] += sign * contrib * 0.5;
              rdm2[ indexer(v1, b, b, o1) ] -= sign * contrib * 0.5;
              rdm2[ indexer(b, v1, b, o1) ] += sign * contrib * 0.5;
              rdm2[ indexer(b, v1, o1, b) ] -= sign * contrib * 0.5;
            }
            // Up-Do
            for( auto b : occs_do )
            {
              rdm2[ indexer(v1, b, o1, b) ] += sign * contrib * 0.5;
              rdm2[ indexer(b, v1, b, o1) ] += sign * contrib * 0.5;
            }
          }
          // Single Do
          else if( exc_up == 0 && exc_do == 2 )
          {
            unsigned short o1 = ket.GetFlippedOccIndx( bra );
            unsigned short v1 = bra.GetFlippedOccIndx( ket );
            double sign = ket.SingleExcDo( v1, o1 );

            // One-body contribution
            rdm1[ indexer(v1, o1) ] += sign * contrib;
            // Two-body contribution
            // Do-Up
            for( auto b : occs_up )
            {
              rdm2[ indexer(v1, b, o1, b) ] += sign * contrib * 0.5;
              rdm2[ indexer(b, v1, b, o1) ] += sign * contrib * 0.5;
            }
            // Up-Do
            for( auto b : occs_do )
            {
              rdm2[ indexer(v1, b, o1, b) ] += sign * contrib * 0.5;
              rdm2[ indexer(v1, b, b, o1) ] -= sign * contrib * 0.5;
              rdm2[ indexer(b, v1, b, o1) ] += sign * contrib * 0.5;
              rdm2[ indexer(b, v1, o1, b) ] -= sign * contrib * 0.5;
            }
          }
        }
        // Diagonal contribution
        for( size_t ist = 0; ist < stts.size(); ist++ )
        {
          SetSlaterDets_It it = std::next( stts.begin(), ist );
          double contrib = vec(ist) * vec(ist); 
          // Store occupied orbitals by spin
          std::vector<unsigned short> occs_up = it->GetOccOrbsUp();
          std::vector<unsigned short> occs_do = it->GetOccOrbsDo();
          // One-body terms
          for( auto o : occs_up )
            rdm1[ indexer(o,o) ] += contrib;
          for( auto o : occs_do )
            rdm1[ indexer(o,o) ] += contrib;
          
          // Two-body terms
          // Up-Up
          for( auto o1 : occs_up )
            for( auto o2 : occs_up )
            {
              rdm2[ indexer(o1,o2,o1,o2) ] += 0.5 * contrib;
              rdm2[ indexer(o1,o2,o2,o1) ] -= 0.5 * contrib;
            }
          // Down-Down
          for( auto o1 : occs_do )
            for( auto o2 : occs_do )
            {
              rdm2[ indexer(o1,o2,o1,o2) ] += 0.5 * contrib;
              rdm2[ indexer(o1,o2,o2,o1) ] -= 0.5 * contrib;
            }
          // Up-Down
          for( auto o1 : occs_up )
            for( auto o2 : occs_do )
            {
              rdm2[ indexer(o1,o2,o1,o2) ] += 0.5 * contrib;
              rdm2[ indexer(o2,o1,o2,o1) ] += 0.5 * contrib;
            }
        }
      }

    }// namespace rdm
  }// namespace ed
}// namespace cmz
