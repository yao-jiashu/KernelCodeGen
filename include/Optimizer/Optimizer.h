#pragma once

#include "Optimizer/Analyzer.h"
#include "Optimizer/Rewriter.h"

#include "IR/IR.h"

#include <unordered_map>

namespace KernelCodeGen {

struct Optimizer {
  virtual bool applicable(mlir::ModuleOp& module) = 0;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) = 0;
  bool operator==(const Optimizer& other) {
    return name == other.name;
  }
  std::string name;
};

struct MatmulOprimizer : Optimizer {

  MatmulOprimizer() {
    this->name = std::move(std::string("Matmul"));
  }

  bool isMatmulPattern(mlir::AffineForOp forOp);

  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;

  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder);

  void clear() {
    matmuls.clear();
    matmulLoops.clear();
    RWBuffers.clear();
  }

  // using the outermost loop represent a matmul.
  std::set<mlir::AffineForOp, CompareLoop> matmuls;


  // Map: from outermost loop to all loops in the matmul(loopM->[loopM, loopN, loopK]).
  std::map<mlir::AffineForOp, std::vector<mlir::AffineForOp>, CompareLoop> matmulLoops;

  // Memory: A, B, C
  struct MemoryBuffer {
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
  };

  // loopM->[A, B, C]
  std::map<mlir::AffineForOp, MemoryBuffer, CompareLoop> RWBuffers;
  static std::map<std::string, int> matmulConfig;
};


}