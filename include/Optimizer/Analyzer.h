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

struct CompareFunc {
  int operator()(const mlir::func::FuncOp& x, const mlir::func::FuncOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

struct CompareFuncCall {
  int operator()(const mlir::func::CallOp& x, const mlir::func::CallOp& y) const {
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

  /// @brief 
  /// @param parallelLevel 
  /// @param totalNumber 
  /// @return 
  static std::vector<int64_t> getParallelNumber(mlir::AffineParallelOp parallelLevel, int64_t& totalNumber) {
    auto dim = parallelLevel.getNumDims();
    totalNumber = 1;
    std::vector<int64_t> result;
    for (int i = 0; i < dim; i++) {
      auto map = parallelLevel.getUpperBoundMap(i);
      auto exprs = map.getResults();
      assert(exprs.size() == 1);
      auto constExpr = exprs[0].dyn_cast<mlir::AffineConstantExpr>();
      assert(constExpr);
      totalNumber *= constExpr.getValue();
      result.push_back(constExpr.getValue());
    }
    return result;
  }
  static std::vector<mlir::func::FuncOp> collectFunctions(mlir::ModuleOp& module, const std::string& targetFuncName = {""});
  static std::vector<mlir::AffineForOp> collectFuncLoops(mlir::func::FuncOp funcOp);
  static std::vector<mlir::func::CallOp> collectFuncCalls(mlir::ModuleOp& module);
  static mlir::func::FuncOp getTargetFunction(mlir::ModuleOp& module, const std::string& targetFuncName);
  static int getUsersNumber(mlir::Value::user_range users);

  template<typename OpType, typename ParentOpType>
  static OpType getLastOp(ParentOpType father) {
    auto& ops = father.getBody()->getOperations();
    OpType res;
    for (auto& op : ops) {
      if (mlir::dyn_cast<OpType>(op)) {
        res = mlir::dyn_cast<OpType>(op);
      }
    }
    return res;
  }
};

}