#include "cmz_ed/bandlan.h++"

inline bool is_file (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

namespace cmz
{
  namespace ed
  {
    bool QRdecomp(std::vector<std::vector<double> > &Q, std::vector<std::vector<double> > &R)
    {
      //CALL LAPACK'S QR DECOMPOSITION ROUTINES.
      //INPUT: Q: INPUT MATRIX TO PERFORM A QR DECOMPOSITION FOR. MAY BE RECTANGULAR,
      //          THE NUMBER OF COLUMNS WILL BE ALWAYS LESS THAN THE NUMBER OF ROWS.
      //OUTPUT: Q: Q MATRIX FROM THE DECOMPOSITION, OVERWRITES INPUT
      //        R: R MATRIX FROM THE DECOMPOSITION. UPPER DIAGONAL, SQUARE.
      //        return : TRUE FOR SUCCESS, FALSE OTHERWISE
    
      R.clear();
      //PREPARE VARIABLES TO CALL LAPACK
      int M = Q.size(), N = Q[0].size();
      assert(M >= N);
      int LDA = M, INFO = 0, LWORK;
      double *A, *TAU, *WORK;
    
      //INITIALIZE A
      A = new double[M * N];
      for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++) A[i + j * M] = Q[i][j];
        Q[i].clear();
      }
      //INITIALIZE TAU, AND PERFORM WORKSPACE QUERY 
      TAU = new double[N];
      WORK =  new double[N];
      LWORK = -1;
    
      dgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dgeqrf_ MEMORY QUERY!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      LWORK = int(WORK[0]);
      delete[] WORK;
      //NOW, PERFORM ACTUAL QR DECOMPOSITION
      WORK = new double[LWORK];
      dgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dgeqrf_ QR DECOMPOSITION!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      //SAVE THE R MATRIX
      R.resize(N);
      for(int i = 0; i < N; i++){
        R[i].resize(N);
        std::fill(R[i].begin(), R[i].end(), 0.);
        for(int j = i; j < N; j++) R[i][j] = A[i + j * M];
      }
    
      //NOW, COMPUTE THE ACTUAL Q MATRIX
      int K = N;
      //FIRST, PERFORM WORKSPACE QUERY
      LWORK = -1;
      dorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dorgqr_ MEMORY QUERY!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      LWORK = int(WORK[0]);
      delete[] WORK;
      WORK = new double[LWORK];
      //NOW, COMPUTE ACTUAL Q
      dorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dorgqr_ COMPUTATION OF Q!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      delete[] TAU;
      delete[] WORK;
      //SAVE THE Q MATRIX
      for(int i = 0; i < M; i++){
        Q[i].resize(N);
        for(int j = 0; j < N; j++) Q[i][j] = A[i + j * M];
      }
      delete[] A;
    
      return true;
    
    }
  
    bool QRdecomp_tr(std::vector<std::vector<double> > &Q, std::vector<std::vector<double> > &R)
    {
      //CALL LAPACK'S QR DECOMPOSITION ROUTINES.
      //INPUT: Q: INPUT MATRIX TO PERFORM A QR DECOMPOSITION FOR. MAY BE RECTANGULAR,
      //          THE NUMBER OF COLUMNS WILL BE ALWAYS LESS THAN THE NUMBER OF ROWS.
      //OUTPUT: Q: Q MATRIX FROM THE DECOMPOSITION, OVERWRITES INPUT
      //        R: R MATRIX FROM THE DECOMPOSITION. UPPER DIAGONAL, SQUARE.
      //        return : TRUE FOR SUCCESS, FALSE OTHERWISE
    
      R.clear();
      //PREPARE VARIABLES TO CALL LAPACK
      int M = Q[0].size(), N = Q.size();
      assert(M >= N);
      int LDA = M, INFO = 0, LWORK;
      double *A, *TAU, *WORK;
    
      //INITIALIZE A
      A = new double[M * N];
      for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++) A[i + j * M] = Q[j][i];
      }
      
      //INITIALIZE TAU, AND PERFORM WORKSPACE QUERY 
      TAU = new double[N];
      WORK =  new double[N];
      LWORK = -1;
    
      dgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dgeqrf_ MEMORY QUERY!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      LWORK = int(WORK[0]);
      delete[] WORK;
      //NOW, PERFORM ACTUAL QR DECOMPOSITION
      WORK = new double[LWORK];
      dgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dgeqrf_ QR DECOMPOSITION!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      //SAVE THE R MATRIX
      R.resize(N);
      for(int i = 0; i < N; i++){
        R[i].resize(N);
        std::fill(R[i].begin(), R[i].end(), 0.);
        for(int j = i; j < N; j++) R[i][j] = A[i + j * M];
      }
    
      //NOW, COMPUTE THE ACTUAL Q MATRIX
      int K = N;
      //FIRST, PERFORM WORKSPACE QUERY
      LWORK = -1;
      dorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dorgqr_ MEMORY QUERY!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      LWORK = int(WORK[0]);
      delete[] WORK;
      WORK = new double[LWORK];
      //NOW, COMPUTE ACTUAL Q
      dorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dorgqr_ COMPUTATION OF Q!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      delete[] TAU;
      delete[] WORK;
      //SAVE THE Q MATRIX
      for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++) Q[j][i] = A[i + j * M];
      }
      delete[] A;
    
      return true;
    
    }
  
    bool GetEigsys(std::vector<std::vector<double> > & mat, std::vector<double> &eigvals, std::vector<std::vector<double> > & eigvecs)
    {
      //COMPUTES THE EIGENVALUES AND EIGENVECTORS OF THE SYMMETRIC MATRIX mat BY CALLING LAPACK.
      //WE ASSUME THE UPPER TRIANGULAR PART OF A IS STORED.
      //FIRST, IT BRINGS THE MATRIX INTO TRIANGULAR FORM, THEN COMPUTES THE EIGENVALUES AND 
      //EIGENVECTORS. THESE ARE STORED IN eigvals AND eigvecs RESPECTIVELY. THE MATRIX mat IS
      //ERASED DURING COMPUTATION
      eigvals.clear(); eigvecs.clear();
      //PREPARE VARIABLES FOR LAPACK
      char UPLO = 'U', COMPZ = 'V';
      int N = mat.size(), LDA = mat.size(), LWORK = -1, INFO;
      double *A, *D, *E, *TAU, *WORK;
  
      //INITIALIZE A
      A = new double[N * N];
      for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++) A[i + j * N] = mat[i][j];
        mat[i].clear();
      }
      mat.clear();
      //ALLOCATE REST OF THE MEMORY
      D = new double[N];
      E = new double[N-1];
      TAU = new double[N-1];
      WORK = new double[N];
  
      //TRANSFORM THE MATRIX TO TRIDIAGONAL FORM
      //FIRST, PERFORM MEMORY QUERY
      dsytrd_(&UPLO, &N, A, &LDA, D, E, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dsytrd_ MEMORY QUERY!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      LWORK = WORK[0];
      delete[] WORK;
      WORK = new double[LWORK];
      //NOW, TRANSFORM MATRIX TO TRIDIAGONAL FORM
      dsytrd_(&UPLO, &N, A, &LDA, D, E, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dsytrd_ COMPUTING THE TRIDIAGONAL MATRIX!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
  
      //COMPUTE THE TRANSFORMATION MATRIX, NECESSARY TO COMPUTE EIGENVECTORS
      //FIRST, PERFORM MEMORY QUERY
      LWORK = -1;
      dorgtr_(&UPLO, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dorgtr_ MEMORY QUERY!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      LWORK = WORK[0];
      delete[] WORK;
      WORK = new double[LWORK];
      //NOW, COMPUTE THE TRANSFORMATION MATRIX. IT WILL BE STORED IN A
      dorgtr_(&UPLO, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dorgtr_ COMPUTING TRANSFORMATION MATRIX!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      delete[] TAU;
  
      //FINALLY, COMPUTE THE EIGENVALUES AND EIGENVECTORS!
      delete[] WORK;
      WORK = new double[2 * N - 2];
      dsteqr_(&COMPZ, &N, D, E, A, &LDA, WORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dsteqr_ COMPUTING EIGENVECTORS AND EIGENVALUES!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      delete[] WORK;
      delete[] E;
  
      //NOW, STORE THE EIGENVALUES AND EIGENVECTORS
      eigvals.resize(N);
      for(int i = 0; i < N; i++) eigvals[i] = D[i]; 
      delete[] D;
      eigvecs.resize(N);
      for(int i = 0; i < N; i++){
        eigvecs[i].resize(N);
        for(int j = 0; j < N; j++) eigvecs[i][j] = A[j + i * N];
      }
      delete[] A;
  
      return true;
  
    }
  
    bool GetEigsysBand(std::vector<std::vector<double> > & mat, int nSupDiag, std::vector<double> &eigvals, std::vector<std::vector<double> > & eigvecs)
    {
      //COMPUTES THE EIGENVALUES AND EIGENVECTORS OF THE SYMMETRIC BAND MATRIX mat BY CALLING LAPACK.
      //WE ASSUME THE UPPER TRIANGULAR PART OF A IS STORED.
      //FIRST, IT BRINGS THE MATRIX INTO TRIANGULAR FORM, THEN COMPUTES THE EIGENVALUES AND 
      //EIGENVECTORS. THESE ARE STORED IN eigvals AND eigvecs RESPECTIVELY. THE MATRIX mat IS
      //ERASED DURING COMPUTATION
      eigvals.clear(); eigvecs.clear();
      //PREPARE VARIABLES FOR LAPACK
      char UPLO = 'U', VECT = 'V', COMPZ = 'V';
      int N = mat.size(), LDQ = mat.size(), LDAB = nSupDiag + 1, LWORK = -1, INFO;
      double *AB, *D, *E, *Q, *WORK;
  
      //INITIALIZE A
      AB = new double[(nSupDiag + 1) * N];
      for(int j = 0; j < N; j++){
        for(int i = std::max(0,j-nSupDiag); i <= j; i++) AB[nSupDiag + i - j + j * (nSupDiag+1)] = mat[i][j];
      }
      mat.clear();
      //ALLOCATE REST OF THE MEMORY
      Q = new double[N * N];
      D = new double[N];
      E = new double[N-1];
      WORK = new double[N];
  
      //TRANSFORM THE MATRIX TO TRIDIAGONAL FORM
      //NOW, TRANSFORM MATRIX TO TRIDIAGONAL FORM
      dsbtrd_(&VECT, &UPLO, &N, &nSupDiag, AB, &LDAB, D, E, Q, &LDQ, WORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dsbtrd_ COMPUTING THE TRIDIAGONAL MATRIX!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      delete[] AB;
  
      //FINALLY, COMPUTE THE EIGENVALUES AND EIGENVECTORS!
      delete[] WORK;
      WORK = new double[2 * N - 2];
      dsteqr_(&COMPZ, &N, D, E, Q, &LDQ, WORK, &INFO);
      if(INFO != 0){
        std::cout << "ERROR IN dsteqr_ COMPUTING EIGENVECTORS AND EIGENVALUES!! ERROR CODE: " << INFO << std::endl;
        return false;
      }
      delete[] WORK;
      delete[] E;
  
      //NOW, STORE THE EIGENVALUES AND EIGENVECTORS
      eigvals.resize(N);
      for(int i = 0; i < N; i++) eigvals[i] = D[i]; 
      delete[] D;
      eigvecs.resize(N);
      for(int i = 0; i < N; i++){
        eigvecs[i].resize(N);
        for(int j = 0; j < N; j++) eigvecs[i][j] = Q[j + i * N];
      }
      delete[] Q;
  
      return true;
  
    }
  
  }// namespace ed 
}// namespace cmz
