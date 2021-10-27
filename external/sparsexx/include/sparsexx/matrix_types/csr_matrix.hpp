#pragma once

#include "type_fwd.hpp"

#ifdef SPARSEXX_ENABLE_CEREAL
  #include <cereal/types/vector.hpp>
#endif

namespace sparsexx {

/**
 *  @brief A class to manipulate sparse matrices stored in CSR format
 *
 *  @tparam T       Field over which the elements of the sparse matrix are defined
 *  @tparam index_t Integer type for the sparse indices
 *  @tparam Alloc   Allocator type for internal storage
 */
template < typename T, typename index_t, typename Alloc >
class csr_matrix {

public:

  using value_type     = T; ///< Field over which the matrix elements are defined
  using index_type     = index_t; ///< Sparse index type
  using size_type      = int64_t; ///< Size type
  using allocator_type = Alloc;   ///< Allocator type

protected:

  using alloc_traits = typename std::allocator_traits<Alloc>;

  template <typename U>
  using rebind_alloc = typename alloc_traits::template rebind_alloc<U>;

  template <typename U>
  using internal_storage = typename std::vector< U, rebind_alloc<U> >;

  size_type m_;         ///< Number of rows in the sparse matrix
  size_type n_;         ///< Number of cols in the sparse matrix
  size_type nnz_;       ///< Number of non-zeros in the sparse matrix
  size_type indexing_;  ///< Indexing base (0 or 1)

  internal_storage< T >       nzval_;  ///< Storage of the non-zero values
  internal_storage< index_t > colind_; ///< Storage of the column indices
  internal_storage< index_t > rowptr_; 
    ///< Storage of the starting indices for each row of the sparse matrix

public:

  csr_matrix() = default;

  /**
   *  @brief Construct a CSR matrix.
   *
   *  @param[in] m    Number of rows in the sparse matrix
   *  @param[in] n    Number of columns in the sparse matrix
   *  @param[in] nnz  Number of non-zeros in the sparse matrix
   *  @param[in] indexing Indexing base (default 1)
   */
  csr_matrix( size_type m, size_type n, size_type nnz,
    size_type indexing = 1) :
    m_(m), n_(n), nnz_(nnz), indexing_(indexing),
    nzval_(nnz), colind_(nnz), rowptr_(m+1)  { }

  csr_matrix( const csr_matrix& other )          = default;
  csr_matrix( csr_matrix&& other      ) noexcept = default;

  csr_matrix& operator=( const csr_matrix& )          = default;
  csr_matrix& operator=( csr_matrix&&      ) noexcept = default;

  // Convert between sparse formats
  csr_matrix( const coo_matrix<T, index_t, Alloc>& other );
  csr_matrix( const csc_matrix<T, index_t, Alloc>& other );
  //csr_matrix& operator=( const coo_matrix<T, index_t, Alloc>& other );





  /**
   *  @brief Get the number of rows in the sparse matrix
   *
   *  @returns Number of rows in the sparse matrix
   */
  size_type m()   const { return m_; };

  /**
   *  @brief Get the number of columns in the sparse matrix
   *
   *  @returns Number of columns in the sparse matrix
   */
  size_type n()   const { return n_; };

  /**
   *  @brief Get the number of non-zeros in the sparse matrix
   *
   *  @returns Number of non-zeros in the sparse matrix
   */
  size_type nnz() const { return nnz_; };

  /**
   *  @brief Get the indexing base for the sparse matrix
   *
   *  @returns The indexing base for the sparse matrix
   */
  size_type indexing() const { return indexing_; }

  /**
   *  @brief Access the non-zero values of the sparse matrix in 
   *  CSR format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in CSR format
   */
  auto& nzval()  { return nzval_; };

  /**
   *  @brief Access the column indices of the sparse matrix in 
   *  CSR format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  column indices of the sparse matrix in CSR format
   */
  auto& colind() { return colind_; };

  /**
   *  @brief Access the row pointer indirection array of the sparse matrix in 
   *  CSR format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  row pointer indirection array of the sparse matrix in CSR format
   */
  auto& rowptr() { return rowptr_; };

  /**
   *  @brief Access the non-zero values of the sparse matrix in 
   *  CSR format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in CSR format
   */
  const auto& nzval () const { return nzval_; };

  /**
   *  @brief Access the column indices of the sparse matrix in 
   *  CSR format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  column indices of the sparse matrix in CSR format
   */

  const auto& colind() const { return colind_; };
  /**
   *  @brief Access the row pointer indirection array of the sparse matrix in 
   *  CSR format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  row pointer indirection array of the sparse matrix in CSR format
   */
  const auto& rowptr() const { return rowptr_; };




#ifdef SPARSEXX_ENABLE_CEREAL
  template <class Archive>  
  void serialize( Archive& ar ) {
    ar( m_, n_, nnz_, indexing_, rowptr_, colind_, nzval_ );
  }
#endif




}; // class csr_matrix

} // namespace sparsexx

#include "conversions.hpp"
