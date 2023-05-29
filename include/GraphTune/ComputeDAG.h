#pragma once
#include "MLIR.h"
#include "GraphTune/ComputeDAG/ComputeDAGDialect.h"
#include "GraphTune/ComputeDAG/ComputeDAGOps.h"


namespace KernelCodegen {

class KernelCodegenMachine;


enum class MemorySpace {
  global = 1,
  shared = 2,
  local = 3,
  constant = 4,
};


class ComputeDAG {
public:
  using Placholder = mlir::memref::AllocOp;
  using GEMM = mlir::compute_dag::GEMMOp;
  using Relu = mlir::compute_dag::ReluOp;
  using Operand = mlir::Operation*;
  using DType = mlir::Type;

  friend class KernelCodegenMachine;

  ComputeDAG(std::string graphName, Context & ctx)
   : builder(&ctx) {
      module = mlir::ModuleOp::create(
        builder.getUnknownLoc(),
        mlir::Optional<mlir::StringRef>(graphName));
      builder.setInsertionPointToEnd(module.getBody());
      registerElementMapping();
  }
  void dumpAndVerify() {
      module->dump();
      if (mlir::failed(mlir::verify(module))) {
          module->emitError("graph verification error");
          assert(false);
      }
  }
  void registerElementMapping();
  void operatorImpl();

  DType getDataType(std::string dtype);
  // operations
  Placholder placeholder(std::vector<int64_t> l, std::string dtype,  MemorySpace ms = MemorySpace::global);
  GEMM gemm(Operand lhs, Operand rhs);
  Relu relu(Operand input);

private:
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
};


}
