#include "cmz_ed/combins.h++"

namespace cmz
{
  namespace ed
  {

    std::vector<unsigned long> BuildCombs( unsigned short Nbits, unsigned short Nset )
    {
      // Returns all bitstrings of Nbits bits with
      // Nset bits set.
      if( Nbits > 16 )
        throw("cmz::ed code is not ready for more than 16 orbitals!!");
      // lookup table
      vector<unsigned long> DP[Nbits+1][Nbits+1];
  
      // DP[k][0] will store all k-bit numbers  
      // with 0 bits set (All bits are 0's)
      for (int len = 0; len <= Nbits; len++) 
          DP[len][0].push_back(0);
        
      // fill DP lookup table in bottom-up manner
      // DP[k][n] will store all k-bit numbers  
      // with n-bits set
      for (int len = 1; len <= Nbits; len++)
      {
          for (int n = 1; n <= len; n++)
          {
              // prefix 0 to all combinations of length len-1 
              // with n ones
              for (unsigned long st : DP[len - 1][n])
                  DP[len][n].push_back(st);
      
              // prefix 1 to all combinations of length len-1 
              // with n-1 ones
              for (unsigned long st : DP[len - 1][n - 1])
                  DP[len][n].push_back( (1<<(len-1)) +  st);
          }
      }

      return DP[Nbits][Nset];
    }

  }// namespace ed
}// namespace cmz
