#pragma once
#include <vector>

namespace asci {

template <typename T>
double to_gib(const std::vector<T>& x) {
  return double(x.capacity() * sizeof(T)) / 1024. / 1024. / 1024.;
}

}
