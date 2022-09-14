#pragma once
#include <vector>
#include "bfgs_hessian.hpp"
#include "line_search.hpp"

namespace bfgs {

template <typename Functor>
detail::arg_type_t<Functor> bfgs( 
  Functor& op,
  const detail::arg_type_t<Functor>& x0,
  BFGSHessian<Functor>& B
) {

  using arg_type  = detail::arg_type_t<Functor>;
  using ret_type  = detail::ret_type_t<Functor>;

  // Initialize BFGS
  arg_type x   = x0;
  ret_type fx  = op.eval(x);
  arg_type gfx = op.grad(x);

  // Initial Hessian
  arg_type p = B.apply(gfx); Functor::scal(-1.0, p);
  ret_type step = 1.; // Initialize step for gradient descent

  size_t max_iter = 3;
  bool converged = false;
  for(size_t iter = 0; iter < max_iter; ++iter) {

    // Do line search
    arg_type x_new, gfx_new = gfx;
    ret_type f_sav = fx;
    try {
    backtracking_line_search(op, x, p, step, x_new, fx, gfx_new);
    //nocedal_wright_line_search(op, x, p, step, x_new, fx, gfx_new);
    } catch(...) {
      std::cout << "Line Search Failed!" << std::endl;

      //double dk = 0.001;
      //double max_k = 1.0;
      //size_t nk = max_k / dk;

      //for( auto ik = 0; ik < nk; ++ik ) {
      //  arg_type xp = x; Functor::axpy(ik * dk, p, xp);
      //  auto fk = op.eval(xp);
      //  std::cout << ik * dk << ", " << fk << ", " << fk - f_sav << std::endl;
      //}

      throw std::runtime_error("die die die");
    }
    std::cout << "  STEP = " << step << std::endl;
    
    // Compute update steps
    arg_type s = Functor::subtract(x_new, x);
    arg_type y = Functor::subtract(gfx_new, gfx);

    // Save quantities for possible next step or return
    x   = x_new;
    gfx = gfx_new;
    step = 1.0;
    //std::cout << std::setprecision(15) << std::scientific;
    std::cout << iter << ", " <<  fx <<  ", " << fx-f_sav << ", " 
              << Functor::norm(gfx)  << std::endl;

    // Check for convergence
    if( Functor::norm(gfx) < 1e-6 /** Functor::norm(x)*/ ) {
        converged = true;
        break;
    }

    // Update and apply Hessian
    std::cout << "X NORM = " << Functor::norm(x_new) << std::endl;
    std::cout << "S NORM = " << Functor::norm(s) << std::endl;
    std::cout << "Y NORM = " << Functor::norm(y) << std::endl;
    B.update(s,y);
    p = B.apply(gfx); Functor::scal(-1.0, p);
    std::cout << "  P NORM = " << Functor::norm(p) << std::endl;

  }

  if(!converged) throw std::runtime_error("BFGS Did Not Converge");
  return x;
  
}

template <typename Functor>
detail::arg_type_t<Functor> bfgs( 
  Functor& op,
  const detail::arg_type_t<Functor>& x0
) {
  auto B = make_updated_scaled_hessian<Functor>();
  return bfgs(op, x0, *B);
}

}
