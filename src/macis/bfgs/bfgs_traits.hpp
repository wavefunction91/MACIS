#pragma once
#include <type_traits>

namespace bfgs::detail {

template <typename Functor>
struct bfgs_traits {
    using arg_type = typename Functor::argument_type;
    using ret_type = typename Functor::return_type;
};

template <typename Functor>
using arg_type_t = typename bfgs_traits<Functor>::arg_type;
template <typename Functor>
using ret_type_t = typename bfgs_traits<Functor>::ret_type;

}
