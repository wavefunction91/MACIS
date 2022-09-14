#pragma once
#include "bfgs_traits.hpp"
#include <iostream>
#include <spdlog/spdlog.h>

namespace bfgs {

template <typename Functor>
void backtracking_line_search(
    Functor& op,
    const detail::arg_type_t<Functor>& x0,
    const detail::arg_type_t<Functor>& p,
          detail::ret_type_t<Functor>& step,
          detail::arg_type_t<Functor>& x,
          detail::ret_type_t<Functor>& fx,
          detail::arg_type_t<Functor>& gfx
) {

    using arg_type = detail::arg_type_t<Functor>;
    using ret_type = detail::ret_type_t<Functor>;

    const auto fx0  = op.eval(x0);
    const auto gfx0 = op.grad(x0);
    const auto dg0 = Functor::dot(p, gfx0);
    if(dg0 > 0)
      throw std::logic_error("the moving direction increases the objective function value");
    if(step <= 0.0)
      throw std::runtime_error("Step must be positive");

#if 1

    constexpr auto c   = 0.5;
    constexpr auto tau = 1./1.618033988749894;

    const auto test_t = -dg0 * c;  

    //std::cout << "IN LINE SEARCH" << std::endl;
    //std::cout << "F(X0) = " << fx0 << " t = " << test_t << std::endl;
    auto logger = spdlog::get("line_search");
    logger->debug("");
    logger->debug("Starting Backtracking Line Search");
    logger->debug("c = {:.4f}, tau = {:.4f}, F(X0) = {:15.12f}, t = {:15.12e}", 
      c,tau,fx, test_t);
    
    const std::string fmt_str =
      "iter = {:4}, F(X) = {:15.12f}, dF = {:20.12e}, S = {:10.6e}";
    
    // Initialization
    step = 1.0;
    x = x0; Functor::axpy(step, p, x);
    fx = op.eval(x);
    //std::cout << "  " << 0 << " F = " << fx  << " s = " << step << std::endl;
    logger->debug(fmt_str, 0, fx, fx - fx0, step);

    size_t max_iter = 100;
    for(size_t iter = 0; iter < max_iter; ++iter) {
      if( (fx0 - fx) - step * test_t > -1e-6 ) break;
      step *= tau;
      x = x0; Functor::axpy(step, p, x);
      fx = op.eval(x);
      logger->debug(fmt_str, iter+1, fx, (fx-fx0), step);
      //std::cout << "  " << iter+1 << " F = " << fx  << " s = " << step << std::endl;
    }

    gfx = op.grad(x);
    logger->debug("Line Search Converged with S = {:10.6e}",step);
    logger->debug("");

#else
    constexpr auto c1 = 1e-4;
    constexpr auto c2 = 0.9;

    const auto armijo_test_val = c1 * dgi;
    const auto wolfe_test_curv = c2 * dgi;

    const size_t max_iter = 100;
    bool converged = false;
    for( size_t iter = 0; iter < max_iter; ++iter ) {

        // Compute new x
        x = x0; Functor::axpy(step, p, x);

        // Evaluate new f(x)
        fx = op.eval(x);

        // Compute Armijo test value
        auto armijo_compare = fx0 + step * armijo_test_val;

        auto width = 1.;
        if( fx > armijo_compare ) { 
            width = 0.5;
        } else { 
            //break; // Armijo condition
            gfx = op.grad(x);
            auto _dg = Functor::dot(gfx, p);
            if( _dg < wolfe_test_curv ) {
                width = 2.1;
            } else {
                //break; // Basic Wolfe condition
                if( _dg > -wolfe_test_curv ) {
                    width = 0.5;
                } else { converged = true; break; } // Strong Wolfe condition
            }
        } 

        // Update test
        step = step * width;

    } // Line search iterations 

    if(!converged) throw std::runtime_error("Backtracking Line Search Failed");
#endif
}








template <typename Functor>
void nocedal_wright_line_search(
    Functor& op,
    const detail::arg_type_t<Functor>& x0,
    const detail::arg_type_t<Functor>& p,
          detail::ret_type_t<Functor>& step,
          detail::arg_type_t<Functor>& x,
          detail::ret_type_t<Functor>& fx,
          detail::arg_type_t<Functor>& gfx
) {

    using arg_type = detail::arg_type_t<Functor>;
    using ret_type = detail::ret_type_t<Functor>;

    const auto fx0 = fx;
    const auto dgi = Functor::dot(p, gfx);
    if(dgi > 0)
      throw std::logic_error("the moving direction increases the objective function value");
    if(step <= 0.0)
      throw std::runtime_error("Step must be positive");


    constexpr auto c1 = 1e-4;
    constexpr auto c2 = 0.9;
    constexpr auto expansion = 2.0;

    const auto armijo_test_val = c1 * dgi;
    const auto wolfe_test_curv = -c2 * dgi;

    ret_type step_hi, step_lo = 0,
             fx_hi,   fx_lo   = fx0,
             dg_hi,   dg_lo   = dgi;

    int iter = 0;
    const size_t max_iter = 100;
    bool converged = false;
    // Bracketing Phase
    for(;;) {
        // Compute new x
        x = x0; Functor::axpy(step, p, x);

        // Evaluate new f(x) and gradiend
        fx  = op.eval(x);
        gfx = op.grad(x);

        if(iter++ >= max_iter) break;
        auto dg = Functor::dot(gfx, p);

        if(fx - fx0 > step * armijo_test_val || (0 < step_lo and fx >= fx_lo)) {
          step_hi = step;
          fx_hi   = fx;
          dg_hi   = dg;
          break;
        }

        if(std::abs(dg) <= wolfe_test_curv) {converged = true; break;} 

        step_hi = step_lo;
        fx_hi   = fx_lo;
        dg_hi   = dg_lo;

        step_lo = step;
        fx_lo   = fx;
        dg_lo   = dg;

        if(dg >= 0) break;
        step *= expansion;
    }
    if(converged) return;

    // Zoom Phase
    iter = 0;
    for(;;) {
        step = (fx_hi - fx_lo) * step_lo - (step_hi * step_hi - step_lo * step_lo) * dg_lo / 2;
        step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo;

        if(step <= std::min(step_lo, step_hi) || step >= std::max(step_lo, step_hi))
            step = step_lo / 2 + step_hi / 2;

        // Compute new x
        x = x0; Functor::axpy(step, p, x);

        // Evaluate new f(x) and gradiend
        fx  = op.eval(x);
        gfx = op.grad(x);

        if(iter++ >= max_iter) break;
        auto dg = Functor::dot(gfx, p);

        if(fx - fx0 > step * armijo_test_val or fx >= fx_lo) {
          if(step == step_hi) throw std::runtime_error("Line Search Failed");
          step_hi = step;
          fx_hi = fx;
          dg_hi = dg;
        } else {
          if(std::abs(dg) <= wolfe_test_curv) {converged = true; break;} 
          if(dg * (step_hi - step_lo) >= 0) {
            step_hi = step_lo;
            fx_hi = fx_lo;
            dg_hi = dg_lo;
          }
          if(step == step_lo) throw std::runtime_error("Line Search Failed");
          step_lo = step;
          fx_lo = fx;
          dg_lo = dg;
        }
    }

    if(!converged) throw std::runtime_error("Line Search Failed");
}



template <typename Functor>
void discrete_line_search(
    Functor& op,
    const detail::arg_type_t<Functor>& x0,
    const detail::arg_type_t<Functor>& p,
          detail::ret_type_t<Functor>& step,
          detail::arg_type_t<Functor>& x,
          detail::ret_type_t<Functor>& fx,
          detail::arg_type_t<Functor>& gfx
) {

    using arg_type = detail::arg_type_t<Functor>;
    using ret_type = detail::ret_type_t<Functor>;

    const auto fx0 = fx;
    const auto dgi = Functor::dot(p, gfx);
    if(dgi > 0)
      throw std::logic_error("the moving direction increases the objective function value");
    if(step <= 0.0)
      throw std::runtime_error("Step must be positive");


    constexpr auto c1 = 1e-4;
    constexpr auto c2 = 0.9;

    const auto armijo_test_val = c1 * dgi;
    const auto wolfe_test_curv = -c2 * dgi;
    std::cout << "DGI = " << dgi << std::endl;

    size_t ngrid = 50;
    const ret_type ds = step / (ngrid-1);
    
    auto min_fx = fx0;
    auto min_dg = dgi;
    size_t min_step_idx = 0.0;
    for( auto i = 0; i < ngrid; ++i ) {
      // Compute new x
      auto t_step = i * ds;
      x = x0; Functor::axpy(t_step, p, x);

      // Evaluate new f(x) and gradiend
      fx  = op.eval(x);
      gfx = op.grad(x);
      auto dg = Functor::dot(gfx, p);

      std::cout << "  ";
      std::cout << t_step << ", " << fx << ", " << dg << std::endl;

      if(fx < min_fx) {
        min_step_idx = i;
      }

      if(i == (ngrid-1) and min_step_idx == i) {
        ngrid *= 2;
      }
    }

    step = min_step_idx * ds;
    x = x0; Functor::axpy(step, p, x);

    // Evaluate new f(x) and gradiend
    fx  = op.eval(x);
    gfx = op.grad(x);


    std::cout << "MIN STEP = " << step << " FX = " << fx << std::endl;
}

}
