#pragma once
#include <sstream>
#include <string>

inline std::string GEMMName(int m, int n, int k) {
  std::stringstream ss;
  ss << "gemm_m" << m << "_n" << n << "_k" << k;
  return ss.str();
}

inline std::string ReluName(int m, int n) {
  std::stringstream ss;
  ss << "relu_m" << m << "_n" << n;
  return ss.str();
}

struct ConstPassGuard {
  ConstPassGuard() { visitor = 0;}
  ~ConstPassGuard() { visitor = 1;}
  void visit() { visitor = 1;}
  bool visited() { return visitor; }
  int visitor = 0;
};