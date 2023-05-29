#include <sstream>
#include <string>

std::string GEMMName(int m, int n, int k) {
  std::stringstream ss;
  ss << "gemm_m" << m << "_n" << n << "_k" << k;
  return ss.str();
}

std::string ReluName(int m, int n) {
  std::stringstream ss;
  ss << "relu_m" << m << "_n" << n;
  return ss.str();
}