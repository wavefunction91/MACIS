/**
 * @file freq_grids.h++
 * @brief Simple implementations of various frequency grids.
 *
 * @author Carlos Mejuto Zaera
 * @date 01/07/2022
 */
#pragma once
#include "cmz_ed/utils.h++"

namespace cmz
{
  namespace ed
  {

    /**
     * @brief Computes linear frequency grid. The direction of
     *        the line may be anywhere in the complex plain, as
     *        its edges are given as complex frequencies.
     *
     * @param[in] const CompD& wmin: First edge of complex frequency line.
     * @param[in] const CompD& wmax: Second edge of complex frequency line.
     * @param[in] const size_t& nws: Number of frequency points in the grid.
     *
     * @returns VeCompD: Linear, complex frequency grid.
     *
     * @author Carlos Mejuto Zaera
     * @date 01/07/2022
     */ 
    inline VecCompD LinearFreqGrid( 
           const CompD &wmin, 
           const CompD &wmax,
           const size_t &nws,
           const double& im_shift = 0. )
    {
      VecCompD ws( nws, CompD(0.,0.) );

      for( int i = 0; i < nws; i++ )
        ws[i] = wmin + (wmax - wmin) / double(nws-1.) * double(i)
                + CompD(0., im_shift);

      return ws;
    }

    /**
     * @brief Computes Matsubara frequency grid. Only 'positive' imaginary
     *        frequencies are considered
     *
     * @param[in] const double& beta: Inverse temperature.
     * @param[in] const size_t& nws: Number of frequency points in the grid.
     *
     * @returns VeCompD: Matsubara, imaginary frequency grid.
     *
     * @author Carlos Mejuto Zaera
     * @date 01/07/2022
     */ 
    inline VecCompD MatsubaraFreqGrid( 
           const double &beta, 
           const size_t &nws )
    {
      VecCompD ws( nws, CompD(0.,0.) );

      for( int i = 0; i < nws; i++ )
        ws[i] = CompD(0., (2. * double(i) + 1.) * M_PI / beta);

      return ws;
    }

    /**
     * @brief Computes logarithmic frequency grid. This has to be
     *        along the real or imaginary axis, and for obvious
     *        reasons cannot start, pass or end at zero.
     *
     * @param[in] const double& wmin: First edge of the frequency line.
     * @param[in] const double& wmax: Second edge of the frequency line.
     * @param[in] const size_t& nws: Number of frequency points in the grid.
     * @param[in] const bool& real: Real axis?
     *
     * @returns VeCompD: Logarithmic, complex frequency grid. Either along
     *          the real of imaginary frequency axis.
     *
     * @author Carlos Mejuto Zaera
     * @date 01/07/2022
     */ 
    inline VecCompD LogFreqGrid( 
           const double &wmin, 
           const double &wmax,
           const size_t &nws,
           const bool &real,
           const double &im_shift = 0. )
    {

      if( ( wmin < 0. && wmax > 0.) || ( wmin > 0. && wmax < 0. ) )
        throw( std::runtime_error( "Error in LogFreqGrid! Requested grid touches or passes by 0." ) );

      VecCompD ws( nws, CompD(0.,0.) );
      CompD fac = real ? CompD(1., 0.) : CompD(0., 1.);

      for( int i = 0; i < nws; i++ )
      {
        double step = std::log(wmin) + ( std::log(wmax) - std::log(wmin) ) / double(nws-1.) * double(i);
        ws[i] = fac * std::exp( step ) + CompD( 0., im_shift );
      }

      return ws;
    }

    /**
     * @brief Generates frequency grid as requested through input file.
     *
     * @param[in] const Input_t &input: Input file defining frequency grid.
     *
     * @returns Requested frequency grid.
     *
     * @author Carlos Mejuto Zaera
     * @date 01/07/2022
     */ 
    inline VecCompD GetFreqGrid(
           const Input_t &input )
    {
      std::string fscale = getParam<std::string>( input, "freq_scale" );
      VecCompD ws;
      if( fscale == "lin" )
      {
        double wmin = getParam<double>( input, "wmin" );
        double wmax = getParam<double>( input, "wmax" );
        size_t nws  = getParam<int>   ( input, "nws"  );
        bool realf  = getParam<bool>  ( input, "real_freq");
        CompD fac   = realf ? CompD(1., 0.) : CompD(0., 1.);
        double im_shift = realf ? getParam<double>( input, "broad" ) : 0.;
        ws = LinearFreqGrid( fac * wmin, fac * wmax, nws, im_shift );
      }
      else if( fscale == "log" )
      {
        double wmin = getParam<double>( input, "wmin" );
        double wmax = getParam<double>( input, "wmax" );
        size_t nws  = getParam<int>   ( input, "nws"  );
        bool realf  = getParam<bool>  ( input, "real_freq");
        double im_shift = realf ? getParam<double>( input, "broad" ) : 0.;
        ws = LogFreqGrid( wmin, wmax, nws, realf, im_shift );
      }
      else if( fscale == "matsubara" )
      {
        double beta = getParam<double>( input, "beta" );
        size_t nws  = getParam<int>   ( input, "nws"  );
        ws = MatsubaraFreqGrid( beta, nws );
      }
      else
        throw( std::runtime_error("Error in GetFreqGrid! Frequency grid requested has not been implemented! Options are: lin, log, matsubara.") );
      return ws;
    }


  }//namespace ed
}// namespace cmz
