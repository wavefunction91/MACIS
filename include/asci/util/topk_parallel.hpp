#pragma once
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "mpi.hpp"
#include <iostream>
#include <limits>

namespace asci {

template <typename T>
T min_value() {
  return std::numeric_limits<T>::min();
}

template <typename T, int Chunk, class Compare>
void nth_element_mpi_op( void* _invec, void* _inoutvec, int* len, MPI_Datatype* d_type ) {
  
  static_assert( Chunk > 0, "Chunk Must Be Positive");

  // Cast Data
  T* invec    = static_cast<T*>(_invec);
  T* inoutvec = static_cast<T*>(_inoutvec);

  // Check Length
  if(*len != 1) throw std::runtime_error("nth_element_mpi_op only works for len = 1");

  // Combine data into local array
  std::vector<T> combined_data(2*Chunk);
  for(int i = 0; i < Chunk; ++i) {
    combined_data[i]         = invec[i];
    combined_data[i + Chunk] = inoutvec[i];
  }

  // Perform reduction
  std::nth_element(combined_data.begin(), combined_data.begin() + Chunk, 
    combined_data.end(), Compare());

  // Copy data back to output buffer
  for(int i = 0; i < Chunk; ++i) {
    inoutvec[i] = combined_data[i];
  }
  
}


// XXX: Assumes that we have unique elements across all ranks
template <int Chunk, typename T, class Compare>
void topk_allreduce( const T* begin, const T* end, int K, T* out, 
  Compare comp, MPI_Comm comm ) {

  const size_t n_local = std::distance(begin, end);

  // Create MPI Reduction Op
  MPI_Op nth_element_reduction_op;
  MPI_Op_create( (MPI_User_function*) nth_element_mpi_op<T,Chunk,Compare>,
    true, &nth_element_reduction_op );

  // Create contiguous MPI type
  auto contig_dtype = make_contiguous_mpi_datatype<T>(Chunk);

  // Copy of data
  std::vector<T> copy_vector(begin, end);

  for(int i_chunk = 0; i_chunk < K/Chunk; ++i_chunk) {

    auto chunk_out = out + i_chunk * Chunk;

    if( copy_vector.size() >= Chunk ) {

      // If local buffer size is larger than Chunk, do local top-K
      if( copy_vector.size() > Chunk ) {
        std::nth_element( copy_vector.begin(), copy_vector.begin() + Chunk,
          copy_vector.end(), comp );
      }

      // Copy local top-K data into buffer
      std::copy_n( copy_vector.begin(), Chunk, chunk_out );

    } else {

      // If there are less than Chunk elements in the local buffer,
      // copy all elements and pad with zeros
      const size_t n_local = copy_vector.size();
      for(int i = 0;       i < n_local; ++i) chunk_out[i] = copy_vector[i];
      for(int i = n_local; i < Chunk;   ++i) chunk_out[i] = min_value<T>();

    }


    // Reduce
    MPI_Allreduce( MPI_IN_PLACE, chunk_out, 1, contig_dtype,
      nth_element_reduction_op, comm );


    // Find loweest element
    // XXX: max_element will return the last element that satisfies comp
    auto min_el = *std::max_element(chunk_out, chunk_out + Chunk, comp);

    // Remove elements from copy buffer
    auto it = std::partition(copy_vector.begin(), copy_vector.end(),
      [=](const auto& v){ return comp(min_el, v); } );

    copy_vector.erase(it, copy_vector.end());

  }

  // Chunk does not evenly divide K
  if( K % Chunk ) {

    std::vector<T> temp_out(Chunk);
    topk_allreduce<Chunk, T, Compare>( 
      copy_vector.data(), copy_vector.data() + copy_vector.size(),
      Chunk, temp_out.data(), comp, comm
    );

    std::nth_element(temp_out.begin(), temp_out.begin() + K % Chunk,
      temp_out.end(), comp);
    std::copy_n( temp_out.begin(), K % Chunk, out + (K/Chunk)*Chunk );

  }

  // Free up MPI types
  MPI_Op_free( &nth_element_reduction_op );

}
}
