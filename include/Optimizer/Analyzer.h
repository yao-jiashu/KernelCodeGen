#pragma once

#include "IR/IR.h"

#include <vector>

namespace KernelCodeGen {

struct CompareLoop {
  int operator()(const mlir::AffineForOp& x, const mlir::AffineForOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

struct Analyzer {
  Analyzer() = default;
  static std::vector<mlir::AffineForOp> collectOutermostLoop(mlir::ModuleOp& module); 
};

}