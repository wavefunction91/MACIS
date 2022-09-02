#include "cmz_ed/eigsolver.h++"

/***Written by Carlos Mejuto Zaera***/

namespace cmz
{
  namespace ed
  {

    void Hste_v(const VecD &alphas, const VecD &betas, VectorXd &eigvals, eigMatD &eigvecs){
       /*
        * COMPUTES THE EIGENVALUES AND EIGENVECTORS OF A TRIDIAGONAL, SYMMETRIC MATRIX A USING LAPACK.
        */
       eigvals.resize(alphas.size());
       eigvecs.resize(alphas.size(), alphas.size());
       //INITIALIZE VARIABLES
       char JOBZ = 'I';        // COMPUTE EIGENVALUES AND EIGENVECTORS OF THE TRIDIAGONAL MATRIX
       int  N = alphas.size(), LDZ = N, INFO; // SIZES
       double *D, *E; // DIAGONAL AND SUB-DIAGONAL ELEMENTS
       double *WORK, *Z;   // WORKSPACE AND EIGENVECTORS
       //INITIALIZE MATRIX 
       D = new double[N];
       for(int64_t i = 0; i < N; i++)
         D[i] = alphas[i];
       E = new double[N-1];
       for(int64_t i = 1; i < N; i++)
         E[i-1] = betas[i];
       //ALLOCATE MEMORY
       WORK  = new double[2*N -2];
       Z     = new double[N * LDZ];
    
       //ACTUAL EIGENVALUE CALCULATION
       dsteqr_(&JOBZ, &N, D, E, Z, &LDZ, WORK, &INFO);
       if(INFO != 0){
         if(INFO < 0)
           throw ("In dsteqr_, the " + std::to_string(-1 * INFO) + "-th argument had an illegal value");
         if(INFO > 0)
           throw ("dsteqr_ the algorithm has failed to find all the eigenvalues in a total of " + std::to_string(30*N) + " iterations");
       }
       delete[] WORK;
       delete[] E;
       //SAVE EIGENVECTORS
       for(int  i = 0; i < N; i++){
         for(int j = 0; j < N; j++) eigvecs(i,j) = Z[i+ j * N];
       }
       delete[] Z;
       //SAVE EIGENVALUES
       for(int i = 0; i < N; i++) eigvals(i) = D[i];
    
       delete[] D;
    }
    
    void Hsyev(const eigMatD &H, VectorXd &eigvals, eigMatD &eigvecs){
       /*
        * COMPUTES THE EIGENVALUES AND EIGENVECTORS OF A SYMMETRIC MATRIX A USING LAPACK.
        */
       eigvals.resize(H.rows());
       eigvecs.resize(H.rows(), H.rows());
       //INITIALIZE VARIABLES
       char JOBZ = 'V', UPLO = 'U';        // COMPUTE EIGENVALUES AND EIGENVECTORS, H IS STORED IN THE UPPER TRIANGLE
       int N = H.rows(), LWORK = -1, LDA = N, INFO; // SIZES
       double *A, *WORK; // MATRIX AND WORKSPACE
       double *W;   // EIGENVALUES AND WORKSPACE
       //INITIALIZE MATRIX 
       A = new double[N * N];
       for(int i = 0; i < N; i++){
         for(int j = 0; j < N; j++)
           A[i + j * N] = H(i,j);
       }
       //ALLOCATE MEMORY
       WORK  = new double[2*N + 1];
       W     = new double[N];
     
       //MEMORY QUERY
       dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);
       if(INFO != 0)
         throw ("ERROR IN dsyev_ MEMORY QUERY!! ERROR CODE: " + std::to_string(INFO));
       LWORK = WORK[0];
       delete[] WORK;
       WORK = new double[LWORK];
       //ACTUAL EIGENVALUE CALCULATION
       dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);
       if(INFO != 0){
         if(INFO < 0)
          throw("ERROR IN dsyev_! The " + std::to_string(-1 * INFO) + "-th argument had an illegal value");
         throw("ERROR IN dsyev_! Algorithm failed to converge! " + std::to_string(INFO) + " off-diagonal elements of an intermediate tridiagonal form did not converge to zero");
       }
       delete[] WORK;
       //SAVE EIGENVECTORS
       for(int  i = 0; i < N; i++){
         for(int j = 0; j < N; j++) 
           eigvecs(i,N - 1 - j) = A[i+ j * N];
       }
       delete[] A;
       //SAVE EIGENVALUES
       for(int i = 0; i < N; i++) 
         eigvals(N - 1 - i) = W[i];
    
       delete[] W;
    }

  }// namespace ed
}// namespace cmz
