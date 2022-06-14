#include "cmz_ed/utils.h++"

namespace cmz
{
  namespace ed
  {

    size_t ReadMatSize(string mat_file){
      /*
       * READS THE DIMENSIONS OF THE SPARSE MATRIX
       */
      ifstream ifile(mat_file.c_str(), ios::in);
      if(!ifile){
        cout << "FILE DOES NOT EXIST!" << endl;
        return 0;
      }
      // READ THE LINE CONTAINING
      // THE DIMENSIONS AND NNZ OF THE MATRIX.
      string line;
      int rows, cols, nnz;
      getline(ifile, line);
      istringstream iss(line);
      iss >> rows >> cols >> nnz;
    
      assert(rows == cols);
      return rows;
      
    }
    
    bool ReadMat(string mat_file, SpMatD &mat){
      /*
       * READS MATRIX FROM FILE. NOT MORE, NOT LESS.   
       */
      
      ifstream ifile(mat_file.c_str(), ios::in);
      if(!ifile){
        cout << "FILE DOES NOT EXIST!" << endl;
        return false;
      }
      // FIRST READ THE LINE CONTAINING
      // THE DIMENSIONS AND NNZ OF THE MATRIX.
      string line;
      int rows, cols, nnz;
      getline(ifile, line);
      istringstream iss(line);
      iss >> rows >> cols >> nnz;
      
      if(mat.rows() != rows or mat.cols() != cols){
    
        cerr << "ERROR!! MATRIX DIMENSIONS MUST AGREE!!" << endl;
        return false;
      }
      // READ THE TRIPLETS TO FILL THE MATRIX.
      VecT tripletlist;
      while(getline(ifile, line)){
        istringstream iss(line);
        int r, c;
        double val;
        iss >> r >> c >> val;
        tripletlist.push_back(T(r , c , val)); 
        if(r != c) tripletlist.push_back(T(c , r , val)); 
      } 
      mat.setFromTriplets(tripletlist.begin(), tripletlist.end());
      mat.makeCompressed();
      return true;
    }
    
    bool ReadVec(string vec_file, VectorXcd &vec){
      /*
       * READS VECTOR FROM FILE
       */
      ifstream ifile(vec_file.c_str(), ios::in);
      if(!ifile){
        cout << "FILE DOES NOT EXIST!" << endl;
        return false;
      }
      
      int size, cont = 0;
      ifile >> size;
      vec = VectorXcd::Zero(size);
      double val;
      while(ifile >> val){
        vec(cont) = std::complex<double>(val, 0.);
        cont++;
      }  
      return true;
    }
    
    bool ReadVec(string vec_file, VecD &vec){
      /*
       * READS VECTOR FROM FILE
       */
      ifstream ifile(vec_file.c_str(), ios::in);
      if(!ifile){
        cout << "FILE DOES NOT EXIST!" << endl;
        return false;
      }
      
      int size, cont = 0;
      ifile >> size;
      vec.resize(size, 0.);
      double val;
      while(ifile >> val){
        vec[cont] = val;
        cont++;
      }  
      return true;
    }
    
    bool ReadInput(string in_file, Input_t &input){
      /*
       * READ INPUT DICTIONARY
       */
      ifstream ifile(in_file.c_str(), ios::in);
      if(!ifile)
        throw ("Error reading input: Cannot open " + in_file);
      string line, key, val;
      while(getline(ifile, line)){
        if(line[0] == '#' || line[0] == '!' || line[0] == '%') continue;
        istringstream iss(line);
        iss >> key;
        iss >> std::ws;
        std::getline(iss, val);
        if(input.find(key) == input.end()){
          input[key] = val;
        }
        else{
          throw ("Key " + key + " was declared more than once in input file!!");
        }
      }
      return true;
    }
    
    std::complex<double> GetMatrixEl(const VectorXcd &vecL, const SpMatD &O, const VectorXcd &vecR){
      /*
       * COMPUTE THE MATRIX ELEMENT OF OPERATOR O WITH STATES vecL AND vecR.
       */
      VectorXcd temp = O * vecR;
      return MyInnProd(vecL, temp); 
    }
    
    double GetMatrixEl(const VectorXd &vecL, const eigMatD &O, const VectorXd &vecR){
      /*
       * COMPUTE THE MATRIX ELEMENT OF OPERATOR O WITH STATES vecL AND vecR.
       */
      VectorXd temp = O * vecR;
      return MyInnProd(vecL, temp); 
    }
    
    double GetMatrixEl(const VecD &vecL, const eigMatD &O, const VecD &vecR){
      /*
       * COMPUTE THE MATRIX ELEMENT OF OPERATOR O WITH STATES vecL AND vecR.
       */
      VectorXd EvecL = VectorXd::Zero(vecL.size());
      VectorXd EvecR = VectorXd::Zero(vecR.size());
      for(int i = 0; i < vecL.size(); i++)
      {
        EvecL(i) = vecL[i];
        EvecR(i) = vecR[i];
      }
      return GetMatrixEl(EvecL, O, EvecR); 
    }
    
    
    eigMatD GetMatExp(const eigMatD &A){
      /*
       * GET MATRIX EXPONENTIAL FROM EIGEN
       */
      return A.exp();
    }
  
  }//namespace ed
}// namespace cmz
