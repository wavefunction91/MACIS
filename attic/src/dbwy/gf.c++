#include "dbwy/gf.h++"

namespace cmz
{
  namespace ed
  {

    void write_GF( const std::vector<std::vector<std::vector<std::complex<double> > > > &GF, const VecCompD &ws, const VecInt &GF_orbs, const VecInt &todelete, const bool is_part )
    {

       size_t nfreqs = ws.size();

       if(GF_orbs.size() > 1)
       {
    
         std::string fname = is_part ?
                             "LanGFMatrix_ADD.dat"
                           : "LanGFMatrix_SUB.dat";
         std::ofstream ofile( fname );
         ofile.precision(dbl::max_digits10);
         for(int iii = 0; iii < nfreqs; iii++){ 
           ofile << scientific << real(ws[iii]) << " " << imag(ws[iii]) << " ";
           for(int jjj = 0; jjj < GF[iii].size(); jjj++){
             for(int lll = 0; lll < GF[iii][jjj].size(); lll++)ofile << scientific << real(GF[iii][jjj][lll]) << " " << imag(GF[iii][jjj][lll]) << " ";
           }
           ofile << std::endl;
         }
         ofile.close();

         std::string fname2 = is_part ?
                             "GFMatrix_OrbitalIndices_ADD.dat"
                           : "GFMatrix_OrbitalIndices_SUB.dat";
         std::ofstream ofile2( fname2 );
         for(int iii = 0; iii < GF_orbs.size(); iii++)
         {
           if(std::find(todelete.begin(), todelete.end(), iii) != todelete.end())
             continue;
           ofile2 << GF_orbs[iii] << std::endl;
         }
         ofile2.close();
       }
       else
       {
         std::string fname = is_part ?
                             "LanGF_ADD_"
                           : "LanGF_SUB_";
         fname += std::to_string(GF_orbs[0]+1) + "_" + std::to_string(GF_orbs[0]+1) + ".dat";
         std::ofstream ofile( fname );
         ofile.precision(dbl::max_digits10);
         for(int iii = 0; iii < nfreqs; iii++) ofile << scientific << real(ws[iii]) << " " << imag(ws[iii]) << " " << real(GF[iii][0][0]) << " " << imag(GF[iii][0][0]) << std::endl;
         ofile.close();
       }
    }

  }// namespace ed
}// namespace cmz
