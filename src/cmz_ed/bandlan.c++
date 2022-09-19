#include "cmz_ed/bandlan.h++"
#include <random>

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
  
    void BandResolvent(
      const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> > &H, 
      std::vector<std::vector<double> > &vecs, 
      const std::vector<std::complex<double> > &ws, 
      std::vector<std::vector<std::vector<std::complex<double> > > > &res,
      int nLanIts,
      double E0,
      bool ispart, 
      bool print ) 
    {
      //COMPUTES THE RESOLVENT (ws - H)^-1 IN MATRIX FORM FOR THE "BASIS" GIVEN BY THE
      //vecs VECTORS AND THE FREQUENCY GRID IN ws. USES THE BAND LANCZOS ALGORITHM. IT
      //GETS STORED IN res.
      res.clear();
      std::cout << "RESOLVENT ROUTINE: ";
      res.resize(ws.size(), std::vector<std::vector<std::complex<double> > >(vecs.size(), std::vector<std::complex<double> >(vecs.size(), std::complex<double>(0.,0.)) ));
      int n = vecs.size();
  
  
      //FIRST, COMPUTE QR DECOMPOSITION OF THE "BASIS" VECTORS vecs, NECESSARY FOR
      //LANCZOS
      std::vector<std::vector<double> > R;
      std::cout << "QR DECOMPOSITION ...";
      bool worked = QRdecomp_tr(vecs, R);
      if(not worked)
      {
        std::cout << "QR DECOMPOSITION FAILED!!" << std::endl;
        return;
      }
      std::cout << "DONE! ";
  
      if(print){
        std::ofstream ofile("QRresVecs.dat", std::ios::out);
        ofile.precision(dbl::max_digits10);
        ofile << "RESULT OF QR DECOMPOSITION: " << std::endl;
        ofile << " New Vectors: " << std::endl;
        for(int i = 0; i < vecs[0].size(); i++){
          for(int j = 0; j < vecs.size(); j++) ofile << std::scientific << vecs[j][i] << "    ";
          ofile << std::endl;
        }
        ofile.close();
        ofile.clear();
        ofile.open("QRresRmat.dat", std::ios::out);
        ofile << " R Matrix: " << std::endl;
        for(int i = 0; i < R.size(); i++){
          for(int j = 0; j < R[i].size(); j++) ofile << std::scientific << R[i][j] << "  ";
          ofile << std::endl;
        }
        ofile.close();
      }
  
      //NEXT, COMPUTE THE BAND LANCZOS
      std::vector<std::vector<double> > bandH;
      std::cout << "BAND LANCZOS ...";
      MyBandLan<double>(H, vecs, bandH, nLanIts, 1.E-6, print);
      std::cout << "DONE! ";
      if(print){
        std::ofstream ofile("BLH.dat", std::ios::out);
        ofile.precision(dbl::max_digits10);
        ofile << "RESULT OF BAND LANCZOS: " << std::endl;
        ofile << " bandH Matrix: " << std::endl;
        for(int i = 0; i < bandH.size(); i++){
          for(int j = 0; j < bandH[i].size(); j++) ofile << std::scientific << bandH[i][j] << "  ";
          ofile <<  std::endl;
        }
        ofile.close();
      }
  
      if(n == 1)
      {
        //ONLY ONE BAND. DIAGONAL GREEN'S FUNCTION ELEMENT.
        //COMPUTE THROUGH CONTINUED FRACTION.
        std::cout << "COMPUTING GF AS CONTINUED FRACTION...";
        std::vector<double> alphas(bandH.size(), 0.), betas(bandH.size(), 0.);
        for(int i = 0; i < bandH.size(); i++)
          alphas[i] = ispart ? E0 - bandH[i][i] : bandH[i][i] - E0;
        for(int i = 0; i < bandH.size()-1; i++)
          betas[i+1] = ispart ? -bandH[i][i+1] : bandH[i][i+1];
        betas[0] = R[0][0];
        #pragma omp parallel for
        for(int indx_w = 0; indx_w < ws.size(); indx_w++){
          res[indx_w][0][0] = betas.back() * betas.back() / (ws[indx_w] + alphas.back());
          for(int i = betas.size() - 2; i >= 0; i--) res[indx_w][0][0] = betas[i] * betas[i] / (ws[indx_w] + alphas[i] - res[indx_w][0][0]);
        }
      }
      else
      {
        //NEXT, COMPUTE THE EIGENVALUES AND EIGENVECTORS OF THE BAND DIAGONAL KRYLOV
        //HAMILTONIAN
        std::vector<double> eigvals;
        std::vector<std::vector<double> > eigvecs;
        std::cout << "COMPUTING EIGENVALES ...";
        if( ispart )
          for( int rr = 0; rr < bandH.size(); rr++ )
          {
            bandH[rr][rr] = E0 - bandH[rr][rr];
            for( int cc = rr+1; cc < bandH.size(); cc++ )
            {
              bandH[rr][cc] = -bandH[rr][cc];
              bandH[cc][rr] = -bandH[cc][rr];
            }
          }
        else
          for( int rr = 0; rr < bandH.size(); rr++ )
            bandH[rr][rr] = bandH[rr][rr] - E0;

        GetEigsysBand(bandH, std::min(size_t(n), bandH.size() - 1), eigvals, eigvecs);
        if(print){
          std::ofstream ofile("BLEigs.dat", std::ios::out);
          ofile.precision(dbl::max_digits10);
          ofile << "RESULT OF EIGENVALUE CALCULATION: " << std::endl;
          ofile << " Eigvals: [";
          for(int i = 0; i < eigvals.size(); i++) ofile << std::scientific << eigvals[i] << ", ";
          ofile << std::endl;
          ofile << "Eigvecs: " << std::endl;
          for(int i = 0; i < eigvecs.size(); i++){
            for(int j = 0; j < eigvecs[i].size(); j++) ofile << std::scientific << eigvecs[i][j] << "  ";
            ofile << std::endl;
          }
          ofile.close();
        }
        std::cout << "DONE! ";
        //FINALLY, COMPUTE S-MATRIX AND RESOLVENT
        std::vector<std::vector<double> > S(nLanIts, std::vector<double>(n, 0.));
        std::cout << " COMPUTING S MATRIX ...";
        for(int i_lan = 0; i_lan < nLanIts; i_lan++){
          for(int j_n = 0; j_n < n; j_n++){
            for(int l = 0; l < n; l++) S[i_lan][j_n] += eigvecs[i_lan][l] * R[l][j_n];
          }
        }
        std::cout << "DONE! COMPUTING RESOLVENT ...";
        #pragma omp parallel for
        for(int iw = 0; iw < ws.size(); iw++){
          for(int k = 0; k < n; k++){
            for(int l = 0; l < n; l++){
              for(int i_lan = 0; i_lan < nLanIts; i_lan++){
                res[iw][k][l] += S[i_lan][k] * 1. / (ws[iw] + eigvals[i_lan]) * S[i_lan][l];
              }
            }
          }
        }
      }
      std::cout << "DONE!" << std::endl; 
    }
  
    void MyBandLanGS(
      const sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, int32_t> > &H, 
      std::vector<std::vector<double> > &qs, 
      std::vector<double> &evals,
      std::vector<std::vector<double> > &evecs,
      int & nLanIts, 
      double tol,
      double thres, 
      bool print)
    {
      //BAND LANCZOS ROUTINE. TAKES AS INPUT THE HAMILTONIAN H, INITIAL VECTORS qs AND 
      //RETURNS THE BAND HAMILTONIAN bandH. IT PERFORMS nLanIts ITERATIONS, STOPPING IF
      //THE NORM OF ANY NEW KRYLOV VECTOR IS BELOW thres. IF LANCZOS IS STOPPED PREMATURELY
      //, nLanIts IS OVERWRITTEN WITH THE ACTUAL NUMBER OF ITERATIONS! THE qs VECTOR
      //IS ERASED AT THE END OF THE CALCULATION
      std::vector<std::vector<double> > bandH;
      bandH.clear();
      bandH.resize(nLanIts, std::vector<double>(nLanIts, 0.));
  
      int nbands = qs.size();
      evals.clear(); evals.resize( nbands, M_PI );
      evecs.clear(); evecs.resize( nbands );
      auto spmv_info = sparsexx::spblas::generate_spmv_comm_info( H );
      //MAKE SPACE FOR 2 * nbands VECTORS
      qs.resize(2 * nbands, std::vector<double>(qs[0].size(), 0.));
      std::vector<double> temp(qs[0].size(), 0.);
      if(print){
        for(int i = 0; i < nbands; i++){
          std::ofstream ofile("lanvec_" + std::to_string(i+1) + ".dat", std::ios::out);
          ofile.precision(dbl::max_digits10);
          for(size_t el = 0; el < qs[i].size(); el++) ofile << std::scientific << qs[i][el] << std::endl;
          ofile.close();
        }
      }
      //DICTIONARY TO KEEP THE REAL INDICES OF THE KRYLOV VECTORS
      //THIS IS NECESSARY IN ORDER TO ONLY STORE 2* nbands OF THEM
      //AT ANY POINT IN TIME, PLUS ONE SCRATCH VECTOR TO BE DEFINED
      //INSIDE THE FOR LOOP
      std::vector<int> true_indx(nLanIts+1);
      for(int i = 0; i < nbands; i++) true_indx[i+1] = i;
      int next_indx = nbands;
  
      for(int it = 1; it <= nLanIts; it++){
        int band_indx_i = true_indx[it]; //TO WHAT ELEMENT OF THE VECTOR SET DO WE APPLY THIS
        sparsexx::spblas::pgespmv( 1., H, qs[band_indx_i].data(), 
                                   0., temp.data(), spmv_info );
        if(print){
          std::ofstream ofile("Htimes_lanvec_" + std::to_string(it) + ".dat", std::ios::out);
          ofile.precision(dbl::max_digits10);
          for(size_t el = 0; el < temp.size(); el++) ofile << std::scientific << temp[el] << std::endl;
          ofile.close();
        }
        for(int jt = std::max(1, it - nbands); jt <= std::min(it - 1, nLanIts); jt++){
          int band_indx_j = true_indx[jt];
          #pragma omp parallel for
          for(size_t coeff = 0; coeff < temp.size(); coeff++) temp[coeff] -= bandH[it-1][jt-1] * qs[band_indx_j][coeff];
        }
        for(int jt = it; jt <= std::min(it + nbands - 1, nLanIts); jt++){
          int band_indx_j = true_indx[jt];
          bandH[it-1][jt-1] = MyInnProd(temp, qs[band_indx_j]);
          bandH[jt-1][it-1] = bandH[it-1][jt-1];
          #pragma omp parallel for
          for(size_t coeff = 0; coeff < temp.size(); coeff++) temp[coeff] -= bandH[it-1][jt-1] * qs[band_indx_j][coeff];
        }
	// Get eigenvalues
	std::vector<double> curr_evals;
	std::vector<std::vector<double> > curr_evecs;
	size_t curr_nbands = std::min( it, nbands );
	std::vector<std::vector<double> > currH( it, std::vector<double>(it, 0.) );
	for( int ir = 0; ir < it; ir++ )
	  for( int ic = 0; ic < it; ic++ )
	    currH[ir][ic] = bandH[ir][ic];
	std::cout << "  * Within BandLanczos GS " << std::endl;
	std::cout << "    * Current Hamiltonian: " << std::endl;
	for( int ir = 0; ir < it; ir++ )
	{
	  std::cout << "     ";
	  for( int ic = 0; ic < it; ic++ )
	    std::cout << currH[ir][ic] << ", ";
	  std::cout << std::endl;
	}
	std::cout << std::endl;
	GetEigsysBand( currH, curr_nbands, curr_evals, curr_evecs);
	std::cout << "  * Generating the Eigenvalues:" << std::endl;
	std::cout << "    * ";
	for( int iii = 0; iii < curr_nbands; iii++ )
	  std::cout << curr_evals[iii] << ", ";
	std::cout << std::endl << std::endl;
	bool done = true;
	for( int iii = 0; iii < curr_nbands; iii++ )
	{
	  double err = std::abs( curr_evals[iii] - evals[iii] );
	  if( err > tol ) done = false;
	  evals[iii] = curr_evals[iii];
	  evecs[iii] = curr_evecs[iii];
	}
	if( done && it >= nbands ) break;
        if(it + nbands <= nLanIts){
          bandH[it-1][it + nbands-1] = std::sqrt(std::real(MyInnProd(temp, temp)));
          bandH[it + nbands-1][it-1] = bandH[it-1][it + nbands-1];
          true_indx[it+nbands] = next_indx;
          if(std::abs(bandH[it-1][it + nbands-1]) < thres){
  //          std::cout << "BAND LANCZOS STOPPED PREMATURELY DUE TO SMALL NORM! NAMELY " << bandH[it-1][it + nbands-1] << ", STOPPED AT ITERATION: " << it << std::endl; 
  //          nLanIts = it;
  //          for(int i = 0; i < nLanIts; i++) bandH[i].resize(nLanIts);
  //          bandH.resize(nLanIts);
  //          break;
            #pragma omp parallel for
            for(size_t coeff = 0; coeff < temp.size(); coeff++) qs[true_indx[it+nbands]][coeff] = 0.;
            std::cout << "FOUND A ZERO VECTOR AT POSITION " << next_indx << std::endl;
          }
          else{
            #pragma omp parallel for
            for(size_t coeff = 0; coeff < temp.size(); coeff++) qs[true_indx[it+nbands]][coeff] = temp[coeff] / bandH[it-1][it + nbands-1];
            if(print){
              std::ofstream ofile("lanvec_" + std::to_string(it+nbands) + ".dat", std::ios::out);
              ofile.precision(dbl::max_digits10);
              for(size_t el = 0; el < qs[true_indx[it+nbands]].size(); el++) ofile << std::scientific << qs[true_indx[it+nbands]][el] << std::endl;
              ofile.close();
            }
          }
          next_indx = (next_indx + 1 >= 2 * nbands) ? 0 : next_indx + 1;
        }
      }
      qs.clear();
    }
  
  }// namespace ed 
}// namespace cmz
