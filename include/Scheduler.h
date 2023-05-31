#pragma once
#include "IR/MLIRExtension.h"
#include "ComputeDAG.h"

namespace KernelCodegen {
using namespace mlir;

enum class LoopAttribute {
  spatial = 1,
  reduction = 2,
};

enum class GPUArch {
  blockIdxX = 1,
  blockIdxY = 2,
  vthreadX = 3,
  vthreadY = 4,
  threadIdxX = 5,
  threadIdxY = 6,
};

struct LoopInfo {
  AffineForOp forOp;
  int scope;
  LoopAttribute attibute;
};


class Scheduler {
public:
  Scheduler() = default;
  Scheduler(ComputeDAG* graph_) : graph(graph_) {}

  // Samplers
  std::vector<LoopInfo> collectLoops();
  std::vector<Value> collectMemRef(MemorySpace ms = MemorySpace::global);
  
  // Primitives
  std::vector<AffineForOp> split(AffineForOp forOp, 
    int num_output, const std::vector<int>& factors);
  void reorder(std::vector<AffineForOp> loopsOrder);
  void bind(AffineForOp forOp, GPUArch level);

private:
  void loweringAffineLoadStore();
  ComputeDAG* graph {nullptr};
};

}