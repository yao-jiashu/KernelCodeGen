#pragma once
#include "IR/MLIRExtension.h"
#include "ComputeDAG.h"
#include "Expression.h"

namespace KernelCodegen {
using namespace mlir;

class Ele_Collecter {
public:
  using DType = ComputeDAG::DType;
  using Placeholder = ComputeDAG::Placeholder;
  // using Function = mlir::func::FuncOp;
  using Loop = AffineForOp;

  Ele_Collecter() = default;
  Ele_Collecter(ComputeDAG* graph_) : graph(graph_) {}

  // collect specific functions
  std::vector<mlir::func::FuncOp> collectFunctions(std::string&& functionName);
  // collect all functions
  std::vector<mlir::func::FuncOp> collectFunctions();

  // std::vector<LoopInfo> collectLoops(Function& func);
  // std::vector<Value> collectInputsAndOutputs();
  // DType getDataType(std::string dtype);
private:
  ComputeDAG* graph {nullptr};
};

}