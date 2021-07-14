#include "cmz_ed/ed.h++"

namespace cmz
{
  namespace ed
  {

    void RunED( const Input_t &input, const intgrls::integrals &ints, double &E0, std::vector<double> &rdm1, std::vector<double> &rdm2 )
    {
      // Performs simple ED calculation
      uint64_t Norbs = getParam<int>( input, "norbs" );
      uint64_t Nups  = getParam<int>( input, "nups"  );
      uint64_t Ndos  = getParam<int>( input, "ndos"  );
      bool print;
      try{ print = getParam<bool>( input, "print" );} catch (...) { print = false; }
    
      SetSlaterDets stts = BuildFullHilbertSpace( Norbs, Nups, Ndos );
    
      FermionHamil Hop(ints);
      
      SpMatD Hmat = GetHmat( &Hop, stts, print );
    
      VectorXd psi0;
    
      SpMatDOp Hwrap( Hmat );
    
      GetGS( Hwrap, E0, psi0, input );
      E0 += ints.get_core_energy();
    
      rdm::rdms rdms( Norbs, psi0, stts );
     
      rdm1 = rdms.rdm1;
      rdm2 = rdms.rdm2;
    }

  }// namespace ed
}// namespace cmz
