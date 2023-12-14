#pragma once

#include "IR/IR.h"
#include "enum.h"
#include "log.h"

namespace KernelCodeGen {

// Responsible for the construction of the graph,
//  and store graph module. 
struct ComputeDAG {
  ComputeDAG(mlir::OpBuilder& builder_) : builder(builder_) {};
  ComputeDAG() = default;
  template <typename OperatorType, typename... Args>
  mlir::Value create(Args &&...args) {
    // auto block = builder.getInsertionBlock();
    // auto iter = builder.getInsertionPoint();
    mlir::Value result;
    {
      // mlir::OpBuilder::InsertionGuard guard(builder);
      //Need to gurantee that OperatorType::build only create a nested AffineForOp or AllocOp.
      result = OperatorType::build(this, std::forward<Args>(args)...);
    }
    // builder.setInsertionPoint(block, ++(++iter));
    return result;
  }

  void dump(const std::string& info = "") {
    if (KCGLog::level == Log::Release) return;
    llvm::errs() << "-----------------------------------------------------------\n";
    llvm::errs() << "           " << info << "\n";
    llvm::errs() << "-----------------------------------------------------------\n";
    module->dump();
    if (mlir::failed(mlir::verify(module))) {
      module->emitError("graph verification error");
      assert(false);
    }
  }

  // ComputeDAG& operator=(const ComputeDAG& other) {
  //   if (module != other.module) {
  //     module = other.module;
  //   }
  //   return *this;
  // } 

  // reference to KernelCodeGenerator::builder.
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
};

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);

/*
// Call like this
KernelCodeGenerator.ComputeDAG.create<>(args...);
*/

// Interface to define operators.
template <typename T>
struct Operator {
  template <typename... Args>
  static mlir::Value build(ComputeDAG* graph, Args &&...args) {
    return T::build(graph, std::forward<Args>(args)...);
  }
};


struct PlaceHolder : Operator<PlaceHolder> {
  static mlir::Value build(ComputeDAG* graph,
    const std::vector<int64_t>& shapes, 
    const std::string& dtype);
};

struct Matmul : Operator<Matmul> {
  static mlir::Value build(ComputeDAG* graph,
    mlir::Value A, mlir::Value B, 
    const std::string& dtype = {""});
};

struct Relu : Operator<Relu> {
  static mlir::Value build(ComputeDAG* graph,
    mlir::Value input,
    const std::string& dtype = {""});
};


}
