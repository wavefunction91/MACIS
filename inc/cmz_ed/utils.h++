/** 
 * @file utils.h++
 * @brief Typedefs, and functions of general use for multiple codes.
 *   
 *  Contains definition of Input_t type, and functions to insert,
 *  modify and consult parameters. Also includes some basic linear
 *  algebra functions, such as vector inner products and matrix elements
 *  of linear operators, and basic input/output routines for matrices
 *  and vectors.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
#ifndef __INCLUDE_CMZED_UTILS__
#define __INCLUDE_CMZED_UTILS__
#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
#include <map>
#include <vector>
#include <assert.h>
#include <iomanip>
#include <limits>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace Eigen;

namespace cmz
{
  namespace ed
  {

    typedef std::complex<double> CompD;
    typedef vector<int> VecInt;
    typedef vector<CompD> VecCompD;
    typedef vector<VecCompD> MatCompD;
    
    typedef vector<double> VecD;
    typedef vector<VecD> MatD;
    
    typedef Triplet<double> T;
    typedef vector<T> VecT;
    typedef MatrixXd eigMatD;
    typedef SparseMatrix<double, RowMajor> SpMatD;
    typedef SparseMatrix<std::complex<double>, RowMajor> SpMatCD;
    
    /**
     * @brief Input container, dictionary string -> string. 
     *        Each value is stored as a string, for booleans
     *        true corresponds to "T", and false to "F".
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    typedef map<string, string> Input_t;
    
    /**
     * @brief Simple inner product routine between two std::vector<double>
     * 
     * @param [in] const std::vecotr<double>& vecR
     * @param [in] const std::vecotr<double>& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline double MyInnProd(const VecD& vecR, const VecD& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      double res = 0.;
      #pragma omp declare reduction \
        (Vsum:double:omp_out=omp_out+omp_in)\
        initializer(omp_priv=0.)
      #pragma omp parallel for reduction (Vsum:res)
      for(size_t i = 0; i < vecR.size();i++) res += vecR[i] * vecL[i];
      return res;
    }
    
    /**
     * @brief Simple inner product routine between two std::vector<std::complex<double> >
     * 
     * @param [in] const std::vecotr<std::complex<double> >& vecR
     * @param [in] const std::vecotr<std::complex<double> >& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline std::complex<double> MyInnProd(const VecCompD& vecR, const VecCompD& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      std::complex<double> res(0.,0.);
      #pragma omp declare reduction \
        (Vsum:std::complex<double>:omp_out=omp_out+omp_in)\
        initializer(omp_priv=std::complex<double>(0.,0.))
      #pragma omp parallel for reduction (Vsum:res)
      for(size_t i = 0; i < vecR.size();i++) res += conj(vecR[i]) * vecL[i];
      return res;
    }
    
    /**
     * @brief Simple inner product routine between std::vector<double> and std::vector<std::complex<double> >
     * 
     * @param [in] const std::vecotr<std::complex<double> >& vecR
     * @param [in] const std::vecotr<double>& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline std::complex<double> MyInnProd(const VecCompD& vecR, const VecD& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      std::complex<double> res(0.,0.);
      #pragma omp declare reduction \
        (Vsum:std::complex<double>:omp_out=omp_out+omp_in)\
        initializer(omp_priv=std::complex<double>(0.,0.))
      #pragma omp parallel for reduction (Vsum:res)
      for(size_t i = 0; i < vecR.size();i++) res += conj(vecR[i]) * std::complex<double>(vecL[i], 0.);
      return res;
    }
    
    /**
     * @brief Simple inner product routine for two Eigen::VectorXcd 
     * 
     * @param [in] const Eigen::VectorXcd& vecR
     * @param [in] const Eigen::VectorXcd& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline std::complex<double> MyInnProd(const VectorXcd& vecR, const VectorXcd& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      return vecR.dot(vecL);
    }
    
    /**
     * @brief Simple inner product routine for two Eigen::VectorXd 
     * 
     * @param [in] const Eigen::VectorXd& vecR
     * @param [in] const Eigen::VectorXd& vecL
     * @return     Inner product <vecR|vecL>
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline double MyInnProd(const VectorXd& vecR, const VectorXd& vecL){
      //SIMPLE INNER PRODUCT ROUTINE
      return vecR.dot(vecL);
    }
    
    /**
     * @brief Simple vector addition routine, for iterable Iterable (template)
     *        with container type Val (template). 
     * 
     * @param [in]     const Val a
     * @param [in]     const Iterable& veca
     * @param [in]     const Vals b
     * @param [in]     const Iterable& vecb
     * @param [in,out] Iterable& res
     *
     * @return res = a * veca + b * vecb
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<typename Val, class Iterable>
    inline void MyAdd(const Val a, const Iterable &veca, const Val b, const Iterable &vecb, Iterable &res){
      //SIMPLE ADDITION ROUTINE
      if( veca.size() != vecb.size() or veca.size() != res.size() )
        throw( "Error in MyAdd!! Sizes of vectors do not match!" );
      std::fill(res.begin(), res.end(), 0.);
      #pragma omp parallel for
      for(size_t i = 0; i < veca.size(); i++)
        res[i] = a * veca[i] + b * vecb[i];
    }
    
    /**
     * @brief Specifies vector addition routine, for Eigen::VectorXd
     *        and double.
     * 
     * @param [in]     const double a
     * @param [in]     const Eigen::VectorXd& veca
     * @param [in]     const double b
     * @param [in]     const Eigen::VectorXd& vecb
     * @param [in,out] Eigen::VectorXd& res
     *
     * @return res = a * veca + b * vecb
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<>
    inline void MyAdd<double, VectorXd>(const double a, const VectorXd &veca, const double b, const VectorXd &vecb, VectorXd &res){
      //SIMPLE ADDITION ROUTINE
      res = a * veca + b * vecb;
    }
    
    /**
     * @brief Specifies vector addition routine, for Eigen::VectorXcd
     *        and std::complex<double>. 
     * 
     * @param [in]     const std::complex<double> a
     * @param [in]     const Eigen::VectorXcd& veca
     * @param [in]     const std::compex<double> b
     * @param [in]     const Eigen::VectorXcd& vecb
     * @param [in,out] Eigen::VectorXcd& res
     *
     * @return res = a * veca + b * vecb
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<>
    inline void MyAdd<std::complex<double>, VectorXcd>(const std::complex<double> a, const VectorXcd &veca, const std::complex<double> b, const VectorXcd &vecb, VectorXcd &res){
      //SIMPLE ADDITION ROUTINE
      res = a * veca + b * vecb;
    }
    
    /**
     * @brief Reads in size of sparse square matrix stored in Matrix Market format (CSR). 
     * 
     * @param [in] std::string mat_file
     *
     * @return size_t with size of the square matrix in mat_file.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    size_t  ReadMatSize(string mat_file);
    /**
     * @brief Reads in sparse matrix stored in Matrix Market format (CSR). 
     * 
     * @param [in]     std::string mat_file
     * @param [in,out] Eigen::SparseMatrix<double, Eigen::RowMajor> & mat, storing CSR matrix.
     *
     * @return bool True if successful read, False otherwise.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    bool ReadMat(string mat_file, SpMatD &mat);
    /**
     * @brief Reads in vector stored in text file. 
     * 
     * @param [in]     std::string vec_file.
     * @param [in,out] Eigen::VectorXcd & vec.
     *
     * @return bool True if successful read, False otherwise.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    bool ReadVec(string vec_file, VectorXcd &vec);
    /**
     * @brief Reads in vector stored in text file. 
     * 
     * @param [in]     std::string vec_file.
     * @param [in,out] std::vector<double> & vec.
     *
     * @return bool True if successful read, False otherwise.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    bool ReadVec(string vec_file, VecD &vec);
    /**
     * @brief Reads in input dictionary from file. 
     * 
     * @param [in]     std::string in_file.
     * @param [in,out] Input_t &input.
     *
     * @return bool True if successful read, False otherwise.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    bool ReadInput(string in_file, Input_t &input);
    
    /**
     * @brief Compute matrix element of an operator represented
     *        as a sparse matrix. 
     * 
     * @param [in] const Eigen::VectorXcd &vecL. 
     * @param [in] const Eigen::SparseMatrix<double,Eigen::RowMajor> &O.
     * @param [in] const Eigen::VectorXcd &vecR. 
     *
     * @return std::complex<double>: Matrix element <vecL|O|vecR>.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    std::complex<double> GetMatrixEl(const VectorXcd &vecL, const SpMatD &O, const VectorXcd &vecR);
    /**
     * @brief Compute matrix element of an operator represented
     *        as a dense matrix. 
     * 
     * @param [in] const Eigen::VectorXd &vecL. 
     * @param [in] const Eigen::Matrix<double> &O.
     * @param [in] const Eigen::VectorXd &vecR. 
     *
     * @return double: Matrix element <vecL|O|vecR>.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    double GetMatrixEl(const VectorXd &vecL, const eigMatD &O, const VectorXd &vecR);
    /**
     * @brief Compute matrix element of an operator represented
     *        as a dense matrix. 
     * 
     * @param [in] const std::vector<double> &vecL. 
     * @param [in] const Eigen::Matrix<double> &O.
     * @param [in] const std::vector<double> &vecR. 
     *
     * @return double: Matrix element <vecL|O|vecR>.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    double GetMatrixEl(const VecD &vecL, const eigMatD &O, const VecD &vecR);
    
    /**
     * @brief Evaluates matrix exponential, using Eigen functions.
     *
     * @param [in] const Eigen::Matrix<double> &A
     *
     * @returns Eigen::Matrix<double>: exp(A)
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    eigMatD GetMatExp(const eigMatD &A);
    
    /**
     * @brief Adds parameter into Input_t dictionary. Corresponding
     *        key cannot exist in dictionary.
     *
     * @param [in,out] Input_t &input
     * @param [in]     const std::string key: Parameter key 
     * @param [in]     const std::string val: String representation
     *                 of parameter value
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline void addParam( Input_t &input, const string key, const string val )
    {
      if(input.find(key) != input.end())
        throw("Error in addParam( Input_t&, const string, const string )! Input already has a parameter with this key = " + key);
      input[key] = val; 
    }
    
    /**
     * @brief Modifies parameter in Input_t dictionary. Corresponding key
     *        must already exist in dictionary.
     *
     * @param [in,out] Input_t &input
     * @param [in]     const std::string key: Parameter key 
     * @param [in]     const std::string val: String representation
     *                 of new parameter value
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    inline void setParam( Input_t &input, const string key, const string val )
    {
      if(input.find(key) == input.end())
        throw("Error in setParam( Input_t&, const string, const string )! Input has no parameter with this key = " + key);
      input[key] = val; 
    }

    /**
     * @brief Template for parameter consultation in input dictionary. 
     *
     * @param [in] Input_t &input
     * @param [in] const std::string key: Parameter key 
     *
     * @returns No return, each type is templated explicitly.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<typename T>
    inline T getParam(const Input_t &input, const string key)
    {
      throw ("Function getParam in utils.h++ has not been templated for this type!");
    }
    
    /**
     * @brief Template for parameter consultation in input dictionary. 
     *        Type double
     *
     * @param [in] Input_t &input
     * @param [in] const std::string key: Parameter key 
     *
     * @returns double: Parameter value.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<>
    inline double getParam<double>(const Input_t &input, const string key)
    {
      if(input.find(key) == input.end())
        throw ("Input did not specify parameter " + key);
      
      return std::stod(input.at(key));
    }
    
    /**
     * @brief Template for parameter consultation in input dictionary. 
     *        Type string
     *
     * @param [in] Input_t &input
     * @param [in] const std::string key: Parameter key 
     *
     * @returns string: Parameter value.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<>
    inline string getParam<string>(const Input_t &input, const string key)
    {
      if(input.find(key) == input.end())
        throw ("Input did not specify parameter " + key);
      
      return input.at(key);
    }
    
    /**
     * @brief Template for parameter consultation in input dictionary. 
     *        Type int
     *
     * @param [in] Input_t &input
     * @param [in] const std::string key: Parameter key 
     *
     * @returns int: Parameter value.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<>
    inline int getParam<int>(const Input_t &input, const string key)
    {
      if(input.find(key) == input.end())
        throw ("Input did not specify parameter " + key);
      
      return std::stoi(input.at(key));
    }
    
    /**
     * @brief Template for parameter consultation in input dictionary. 
     *        Type bool
     *
     * @param [in] Input_t &input
     * @param [in] const std::string key: Parameter key 
     *
     * @returns bool: Parameter value.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<>
    inline bool getParam<bool>(const Input_t &input, const string key)
    {
      if(input.find(key) == input.end())
        throw ("Input did not specify parameter " + key);
      
      if(input.at(key) != "T" && input.at(key) != "F")
        throw ("Boolean input parameter " + key + " ill defined! You have to specify T of F!");
      return input.at(key) == "T";
    }

  }// namespace ed
}// namespace cmz
#endif
