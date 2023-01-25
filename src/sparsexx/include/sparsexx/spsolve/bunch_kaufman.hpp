#pragma once

#include <memory>
#include <stdexcept>
#include <iostream>

namespace sparsexx::spsolve {

namespace detail {
  template <typename SpMatType>
  struct bunch_kaufman_pimpl {

    using value_type = typename SpMatType::value_type;

    virtual void factorize( const SpMatType& ) = 0;
    virtual void solve( int64_t NRHS, const value_type* B, int64_t LDB, 
      value_type* X, int64_t LDX ) = 0;
    virtual void solve( int64_t NRHS, value_type* B, int64_t LDB ) = 0;

    virtual std::tuple<int64_t, int64_t, int64_t> get_inertia() = 0;

    virtual ~bunch_kaufman_pimpl() noexcept = default;

  };

  struct bunch_kaufman_init_exception : public std::exception {
    const char* what() const throw() {
      return "Bunch-Kaufman Instance Has Not Been Initialized";
    }
  };
}

template <typename SpMatType>
class bunch_kaufman {

public:

  using value_type = typename SpMatType::value_type;

protected:

  using pimpl_type = std::unique_ptr< detail::bunch_kaufman_pimpl<SpMatType> >;
  pimpl_type pimpl_;

public:

  virtual ~bunch_kaufman() noexcept = default;

  bunch_kaufman( pimpl_type&& pimpl ) :
    pimpl_( std::move( pimpl ) ) { }
  
  bunch_kaufman( ) : bunch_kaufman(nullptr){ }

  bunch_kaufman( const bunch_kaufman& ) = delete;
  bunch_kaufman( bunch_kaufman&&      ) noexcept = default;

  bunch_kaufman& operator=( const bunch_kaufman& ) = delete;
  bunch_kaufman& operator=( bunch_kaufman&&      ) noexcept = default;

  void factorize(const SpMatType& A) {
    if( pimpl_ ) pimpl_->factorize(A);
    else throw detail::bunch_kaufman_init_exception();
  }

  void solve( int64_t K, const value_type* B, int64_t LDB, value_type* X, int64_t LDX ) {
    if( pimpl_ ) pimpl_->solve( K, B, LDB, X, LDX );
    else throw detail::bunch_kaufman_init_exception();
  };

  void solve( int64_t K, value_type* B, int64_t LDB ) {
    if( pimpl_ ) pimpl_->solve( K, B, LDB );
    else throw detail::bunch_kaufman_init_exception();
  };

  std::tuple< int64_t, int64_t, int64_t > get_inertia() {
    if( pimpl_ ) return pimpl_->get_inertia();
    else throw detail::bunch_kaufman_init_exception();
  }
}; // class bunch_kaufman

}
