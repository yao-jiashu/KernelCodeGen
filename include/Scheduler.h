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

inline std::string getGPUArchStr(GPUArch arch) {
  switch(arch) {
    case GPUArch::blockIdxX : return "blockIdx.x";
    case GPUArch::blockIdxY : return "blockIdx.y";
    case GPUArch::vthreadX : return "vthread.x";
    case GPUArch::vthreadY : return "vthread.y";
    case GPUArch::threadIdxX : return "threadIdx.x";
    case GPUArch::threadIdxY : return "threadIdx.y";
  }
  return "";
}

struct LoopInfo {
  AffineForOp forOp;
  int scope;
  LoopAttribute attibute;
};


class Scheduler {
public:
  Scheduler() = default;
  Scheduler(ComputeDAG* graph_) : graph(graph_) {}

  std::vector<LoopInfo> collectLoops();
  std::vector<Value> collectInputsAndOutputs();
  
  // Primitives
  std::vector<AffineForOp> split(AffineForOp forOp, 
    int num_output, const std::vector<int>& factors);
  // move the inner loop to the outer is always true;
  void reorder(std::vector<AffineForOp> loopsOrder);
  void bind(AffineForOp forOp, GPUArch level);
  // The src of cache_write can be read and write
  Value cache_write(Value src, MemorySpace ms, AffineForOp where);
  // The src of cache_read only can be read
  Value cache_read(Value src, MemorySpace ms, AffineForOp where);

private:
  void loweringAffineLoadStore();
  ComputeDAG* graph {nullptr};
};

}