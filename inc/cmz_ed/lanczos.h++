/**
 * @brief Implements simple one-band Lanczos method
 *        to compute the lowest eigenvalue of a given
 *        Hamiltonian.
 *
 * @author Carlos Mejuto Zaera
 * @date 05/04/2021
 */
#ifndef __INCLUDE_CMZED_LANCZOS__
#define __INCLUDE_CMZED_LANCZOS__
#include "cmz_ed/utils.h++"
#include "cmz_ed/eigsolver.h++"
#include<omp.h>

namespace cmz
{
  namespace ed
  {

    /**
     * @brief Wrapper class for Eigen::MatrixXd, to be used in 
     *        the Lanczos code. Just needs to implement a matrix-
     *        vector product dot, and a function rows() to return the
     *        nr. of rows in the matrix.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    class eigMatDOp
    {
      private:
        const eigMatD *mat;
      public:
        /**
         * @brief Constructor, takes Eigen::MatrixXd and keeps
         *        pointer to it.
         *
         * @param [in] const Eigen::MatrixXd &A: Matrix to wrap.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/04/2021
         */
        eigMatDOp(const eigMatD &A){mat = &A;}
        /**
         * @brief Simple matrix-vector product. 
         *
         * @param [in] const Eigen::VectorXd &vec: Input vector.
         *
         * @returns Eigen::VectorXd: A * vec.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/04/2021
         */
        VectorXd dot(const VectorXd &vec) const {return (*mat) * vec;}
        /**
         * @brief Returns nr. of rows in the wrapped matrix. 
         *
         * @returns int: Nr. of rows.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/04/2021
         */
        int64_t rows() const {return mat->rows();}
    };
    
    /**
     * @brief Wrapper class for Eigen::SparseMatrix<double, Eigen::RowMajor>, to be used in 
     *        the Lanczos code. Just needs to implement a matrix-
     *        vector product dot, and a function rows() to return the
     *        nr. of rows in the matrix.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    class SpMatDOp
    {
      private:
        const SpMatD *mat;
      public:
        /**
         * @brief Constructor, takes Eigen::SparseMatrix<double, Eigen::RowMajor> and keeps
         *        pointer to it.
         *
         * @param [in] const Eigen::SparseMatrix<double, Eigen::RowMajor> &A: Matrix to wrap.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/04/2021
         */
        SpMatDOp(const SpMatD &A){mat = &A;}
        /**
         * @brief Simple matrix-vector product. 
         *
         * @param [in] const Eigen::VectorXd &vec: Input vector.
         *
         * @returns Eigen::VectorXd: A * vec.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/04/2021
         */
        VectorXd dot(const VectorXd &vec) const {return (*mat) * vec;}
        /**
         * @brief Returns nr. of rows in the wrapped matrix. 
         *
         * @returns int: Nr. of rows.
         *
         * @author Carlos Mejuto Zaera
         * @date 05/04/2021
         */
        int64_t rows() const {return mat->rows();}
    };
    
    /**
     * @brief Simple single band Lanczos implementation.
     *
     * @param [in] const Eigen::VectorXd &start_vec: Starting vector.
     * @param [in] const MatOp &H: Wrapped Matrix. Has to be Hermitian!
     * @param [in] int64_t nLanIts: Maximal number of iterations.
     * @param [out] std::vector<double> &alphas: Diagonal of the tri-diagonal H. 
     * @param [out] std::vector<double> &betas: Off-diagonal of the tri-diagonal H.
     * @param [in] double tol: Lanczos tolerance.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<class MatOp>
    void MyLanczos(const VectorXd &start_vec, const MatOp &H, int64_t nLanIts, VecD &alphas, VecD &betas, double tol)
    {
      //LANCZOS ROUTINE USING TEMPLATED MATRIX
      //CLASS. ONLY NEEDS TO PROVIDE A MATRIX
      //VECTOR PRODUCT.
      int64_t n = start_vec.rows();
      VectorXd qold  = VectorXd::Zero(n);
      VectorXd qtemp = VectorXd::Zero(n);
      VectorXd qnew  = VectorXd::Zero(n);
      alphas.clear();
      betas.clear();
      
      alphas.resize(nLanIts, 0.);
      betas.resize(nLanIts+1, 0.);
    
      double normpsi = sqrt(MyInnProd(start_vec, start_vec));
      int64_t nlan = -1, itermax = nLanIts;
    
      nlan++;
      qold = start_vec / normpsi;
    
      qnew = H.dot(qold);
      alphas[0] = MyInnProd(qold, qnew);
      qnew -= alphas[0] * qold;
      betas[0] = normpsi;
      betas[1] = sqrt(MyInnProd(qnew, qnew));
      if(abs(betas[1]) <= tol) itermax = 1;
      for(size_t iter = 1; iter < itermax; iter++){
        nlan++;
        qtemp = qold;
        qold = qnew / betas[iter];
        qnew = -betas[iter] * qtemp;
    
        qtemp = H.dot(qold);
       
        qnew += qtemp;
        alphas[iter] = MyInnProd(qold, qnew);
        qnew -= alphas[iter] * qold;
        betas[iter + 1] = sqrt(MyInnProd(qnew, qnew));
    
        if(abs(betas[iter + 1]) <= tol){
          itermax = iter;
          cout << "EXIT BECAUSE BETA IS TOO SMALL!! AT ITERATION " << iter << ", betas[iter + 1] = " << betas[iter + 1] << endl;
          break;
        }
      }
    }
    
    /**
     * @brief Single band Lanczos implementation, with backprojection to
     *        evaluate eigenvector in the original basis. A previous Lanczos
     *        to determine the ground state in the Krylov basis has to be performed
     *        first.
     *
     * @param [in] const Eigen::VectorXd &start_vec: Starting vector.
     * @param [in] const MatOp &H: Wrapped Matrix. Has to be Hermitian!
     * @param [in] int64_t nLanIts: Maximal number of iterations.
     * @param [in] Eigen::VectorXd &vec_P: Eigenvector in the Krylov basis. 
     * @param [out] Eigen::VectorXd &vec_BP: Eigenvector in the original basis. 
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<class MatOp>
    void MyLanczos_BackProj(const VectorXd &start_vec, const MatOp &H, int64_t nLanIts, VectorXd &vec_P, VectorXd &vec_BP)
    {
      //REBUILD THE EIGENVECTOR FROM A PREVIOUS LANZOS
      //CALCULATION.
      int64_t n = start_vec.rows();
      VectorXd qold  = VectorXd::Zero(n);
      VectorXd qtemp = VectorXd::Zero(n);
      VectorXd qnew  = VectorXd::Zero(n);
      
      VecD alphas(nLanIts, 0.);
      VecD betas(nLanIts+1, 0.);
    
      double normpsi = sqrt(MyInnProd(start_vec, start_vec));
      int64_t nlan = -1, itermax = nLanIts;
    
      vec_BP = VectorXd::Zero(n);
      
      nlan++;
      qold = start_vec / normpsi;
      vec_BP = vec_P(0) * qold;
    
      qnew = H.dot(qold);
      alphas[0] = MyInnProd(qold, qnew);
      qnew -= alphas[0] * qold;
      betas[0] = normpsi;
      betas[1] = sqrt(MyInnProd(qnew, qnew));
      for(size_t iter = 1; iter < itermax; iter++){
        nlan++;
        qtemp = qold;
        qold = qnew / betas[iter];
        vec_BP += vec_P(iter) * qold;
        qnew = -betas[iter] * qtemp;
    
        qtemp = H.dot(qold);
       
        qnew += qtemp;
        alphas[iter] = MyInnProd(qold, qnew);
        qnew -= alphas[iter] * qold;
        betas[iter + 1] = sqrt(MyInnProd(qnew, qnew));
      }
      alphas.clear();
      betas.clear();
    }
    
    /**
     * @brief Determine ground state of Hamiltonian using Lanczos, returns
     *        the ground state vector in the Krylov basis. 
     *
     * @param [in] const Eigen::VectorXd &start_vec: Starting vector.
     * @param [in] const MatOp &H: Wrapped Hamiltonian.
     * @param [out] double &E0: Ground state energy. 
     * @param [out] Eigen::VectorXd &psi0_Lan: Eigenvector in the Krylov basis. 
     * @param [in] Input_t &input: Input dictionary with Lanczos parameters. 
     *
     * @returns int: Required nr. of Lanczos iterations.
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<class MatOp>
    int64_t GetGSEn_Lanczos(const VectorXd &start_vec, const MatOp &H, double &E0, VectorXd &psi0_Lan, const Input_t &input)
    {
      //COMPUTE LOWEST EIGENVALUE OF MATRIX H
      //USING LANCZOS. RETURNS EIGENVECTOR IN
      //THE BASIS OF KRYLOV VECTORS 
      auto w = std::setw(15);
      double Lantol, E0tol;
      bool print;
      try{Lantol = getParam<double>(input, "Lantol");}catch(...){Lantol = 1.E-8;}
      try{E0tol = getParam<double>(input, "E0tol");}catch(...){E0tol = 1.E-8;}
      try{print = getParam<bool>(input, "print");}catch(...){print = false;}
    
      int64_t n = start_vec.rows();
      VectorXd qold  = VectorXd::Zero(n);
      VectorXd qtemp = VectorXd::Zero(n);
      VectorXd qnew  = VectorXd::Zero(n);
      VectorXd eigvals;
      eigMatD eigvecs;
      VecD alphas, betas;
      double currE0, prevE0;
    
      double normpsi = sqrt(MyInnProd(start_vec, start_vec));
      int64_t nlan = -1, itermax = 1000;
    
      nlan++;
      qold = start_vec / normpsi;
    
      qnew = H.dot(qold);
      alphas.push_back(MyInnProd(qold, qnew));
      qnew -= alphas[0] * qold;
      betas.push_back(normpsi);
      betas.push_back(sqrt(MyInnProd(qnew, qnew)));
      prevE0 = alphas[0];
    
      if(print)
      {
        cout << w << "Lanczos diagonalization:" << endl;
        std::ostringstream header;
        header << w << "Iter." << w << "Alpha" << w << "Beta" << w << "E0" << w << "dE";
        cout << header.str() << endl;
        cout << w << std::string(header.str().length(), '-') << endl;
        cout << w <<  1 << w << scientific << std::setprecision(5) << alphas[0] << w << betas[1] << w << prevE0 << w << "***" << endl;
      }
    
      if(abs(betas[1]) <= Lantol){
        itermax = 1;
        currE0 = alphas[0];
      }
      for(size_t iter = 1; iter < itermax; iter++){
        nlan++;
        qtemp = qold;
        qold = qnew / betas[iter];
        qnew = -betas[iter] * qtemp;
    
        qtemp = H.dot(qold);
       
        qnew += qtemp;
        alphas.push_back(MyInnProd(qold, qnew));
        qnew -= alphas[iter] * qold;
        betas.push_back(sqrt(MyInnProd(qnew, qnew)));
        //GET EIGENVALUES OF CURRENT TRI-DIAG MATRIX
        Hste_v(alphas, betas, eigvals, eigvecs);
        currE0 = eigvals(0);
        if(print)
          cout << w << iter+1 << w << scientific << std::setprecision(5) << alphas[iter] << w << betas[iter+1] << w << currE0 << w << abs(currE0 - prevE0) << endl;
    
        if(abs(betas[iter + 1]) <= Lantol){
          itermax = iter+1;
          if(print)
            cout << "EXIT LANCZOS BECAUSE BETA IS TOO SMALL!! AT ITERATION " << iter << ", betas[iter + 1] = " << betas[iter + 1] << endl;
          break;
        }
        if(abs(currE0 - prevE0) <= E0tol){
          itermax = iter+1;
          if(print)
            cout << "EXIT LANCZOS BECAUSE ENERGY ACCURACY WAS OBTAINED!! AT ITERATION " << iter << ", dE = " << abs(currE0 - prevE0) << endl;
          break;
        }
        prevE0 = currE0; 
      }
     
      if(abs(prevE0 - currE0) > E0tol && nlan == itermax - 1 && abs(betas[itermax]) > Lantol )
      {
        std::ostringstream oss;
        oss << "Unable to achieve the desired accuracy of " << std::scientific << E0tol << " in Lanczos after " << itermax << " iterations!!";
        throw (oss.str());
      }
    
      E0 = currE0;
      if(itermax == 1)
      {
        psi0_Lan = VectorXd::Ones(1);
      }
      else
      {
        psi0_Lan = eigvecs.col(0);
      }
      return itermax;
    }
    
    /**
     * @brief Determine ground state of Hamiltonian using Lanczos, returns
     *        the ground state vector in the original basis. 
     *
     * @param [in] const Eigen::VectorXd &start_vec: Starting vector.
     * @param [in] const MatOp &H: Wrapped Hamiltonian.
     * @param [out] double &E0: Ground state energy. 
     * @param [out] Eigen::VectorXd &psi0: Ground state eigenvector. 
     * @param [in] Input_t &input: Input dictionary with Lanczos parameters. 
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<class MatOp>
    void GetGSEnVec_Lanczos(const VectorXd &start_vec, const MatOp &H, double &E0, VectorXd &psi0, const Input_t &input)
    {
      //COMPUTE LOWEST EIGENVALUE AND EIGENVECTOR
      //OF MATRIX H USING LANCZOS. 
      double Lantol, E0tol;
      bool print;
      try{Lantol = getParam<double>(input, "Lantol");}catch(...){Lantol = 1.E-8;}
      try{E0tol = getParam<double>(input, "E0tol");}catch(...){E0tol = 1.E-8;}
      try{print = getParam<bool>(input, "print");}catch(...){print = false;}
    
      int64_t n = start_vec.rows();
      VectorXd qold  = VectorXd::Zero(n);
      VectorXd qtemp = VectorXd::Zero(n);
      VectorXd qnew  = VectorXd::Zero(n);
      VectorXd eigvals;
      eigMatD eigvecs;
      std::vector<VectorXd> kry_vecs;
      VecD alphas, betas;
      double currE0, prevE0;
    
      double normpsi = sqrt(MyInnProd(start_vec, start_vec));
      int64_t nlan = -1, itermax = 1000;
    
      nlan++;
      qold = start_vec / normpsi;
      kry_vecs.push_back(qold);
    
      qnew = H.dot(qold);
      alphas.push_back(MyInnProd(qold, qnew));
      qnew -= alphas[0] * qold;
      betas.push_back(normpsi);
      betas.push_back(sqrt(MyInnProd(qnew, qnew)));
      prevE0 = alphas[0];
    
      if(abs(betas[1]) <= Lantol){
        itermax = 1;
        currE0 = alphas[0];
      }
      for(size_t iter = 1; iter < itermax; iter++){
        nlan++;
        qtemp = qold;
        qold = qnew / betas[iter];
        kry_vecs.push_back(qold);
        qnew = -betas[iter] * qtemp;
    
        qtemp = H.dot(qold);
       
        qnew += qtemp;
        alphas.push_back(MyInnProd(qold, qnew));
        qnew -= alphas[iter] * qold;
        betas.push_back(sqrt(MyInnProd(qnew, qnew)));
        //GET EIGENVALUES OF CURRENT TRI-DIAG MATRIX
        Hste_v(alphas, betas, eigvals, eigvecs);
        currE0 = eigvals(0);
    
        if(abs(betas[iter + 1]) <= Lantol){
          itermax = iter;
          if(print)
            cout << "EXIT LANCZOS BECAUSE BETA IS TOO SMALL!! AT ITERATION " << iter << ", betas[iter + 1] = " << betas[iter + 1] << endl;
          break;
        }
        if(abs(currE0 - prevE0) <= E0tol){
          itermax = iter;
          if(print)
            cout << "EXIT LANCZOS BECAUSE ENERGY ACCURACY WAS OBTAINED!! AT ITERATION " << iter << ", dE = " << abs(currE0 - prevE0) << endl;
          break;
        }
        prevE0 = currE0; 
      }
     
      if(abs(prevE0 - currE0) > E0tol && nlan == itermax - 1 )
      {
        std::ostringstream oss;
        oss << "Unable to achieve the desired accuracy of " << std::scientific << E0tol << " in Lanczos after " << itermax << " iterations!!";
        throw (oss.str());
      }
    
      E0 = currE0;
      psi0 = VectorXd::Zero(n);
      if(kry_vecs.size() == 1)
      {
        psi0 = kry_vecs[0];
      }
      else
      {
        for(int64_t i = 0; i < kry_vecs.size(); i++)
          psi0 += eigvecs.col(0)(i) * kry_vecs[i];
      }
    }
    
    /**
     * @brief Computes ground state of Hamiltonian and corresponding 
     *        eigenvector using Lanczos.
     *
     * @param [in] const MatOp &H: Wrapped Hamiltonian.
     * @param [out] double &E0: Ground state energy. 
     * @param [out] Eigen::VectorXd &psi0: Ground state eigenvector. 
     * @param [in] Input_t &input: Input dictionary with Lanczos parameters. 
     * @param [in] bool superCI: Determines starting guess for Lanczos. If true
     *             we start from [1,0,0,0,...] vector, otherwise from [1,1,1,...].
     *
     * @author Carlos Mejuto Zaera
     * @date 05/04/2021
     */
    template<class MatOp>
    void GetGS(const MatOp &H, double &E0, VectorXd &psi0, const Input_t &input, bool isHF = false)
    {
      //Computes the lowest eigenvalue and corresponding
      //eigenvector from the dense matrix H by using Lanczos.
     
      int64_t n = H.rows();
      //Initial vector. We choose (1,0,0,0,...)t
      //for HF, Otherwhise  (1,1,1,1,...)t
      VectorXd start_psi = isHF ? VectorXd::Zero(n) : VectorXd::Ones(n);
      start_psi(0) = 1.;
      //Determine lowest eigenvalue for the given
      //tolerance.
      VectorXd psi0_Lan;
      int64_t nLanIts = GetGSEn_Lanczos(start_psi, H, E0, psi0_Lan, input);
      //Reconstruct the eigenvector
      MyLanczos_BackProj(start_psi, H, nLanIts, psi0_Lan, psi0);
      start_psi = psi0;
      GetGSEnVec_Lanczos(start_psi, H, E0, psi0, input);
    }

  }// namespace ed
}// namespace cmz
#endif
