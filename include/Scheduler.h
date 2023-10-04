#pragma once
#include "IR/MLIRExtension.h"
#include "ComputeDAG.h"
#include "Expression.h"

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

enum class ThreadScope {
  devive = 1,
  cluster = 2,
  block = 3,
  warp = 4,
  thread = 5,
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

struct Tensor {

  using Expr = std::shared_ptr<Expression>;

  Tensor(const Value& inputOp, 
    std::vector<Expr>&& start_, 
    std::vector<Expr>&& end_, 
    const std::vector<int>& pack_width_) : 
      memory(inputOp), start(start_), end(end_), pack_width(pack_width_) {
    
    auto memType = memory.getType();

    auto type = memType.dyn_cast<MemRefType>();
    if(type.isa<mlir::MemRefType>()) {
      auto shape_ = type.dyn_cast<mlir::MemRefType>();
      rank = shape_.getShape().size();
      dtype = shape_.getElementType();
      ms = shape_.getMemorySpace();
    } else {
      std::cout << "Unsupport input of Tensor.\n";
    }
  }

  Value memory;
  int rank;
  std::vector<int64_t> shape;
  std::vector<Expr> start;
  std::vector<Expr> end;
  std::vector<int> pack_width;
  mlir::Attribute ms;
  ComputeDAG::DType dtype;
};

class Scheduler {
public:
  using DType = ComputeDAG::DType;
  using Placeholder = ComputeDAG::Placeholder;
  using Function = mlir::func::FuncOp;
  using Loop = AffineForOp;

  Scheduler() = default;
  Scheduler(ComputeDAG* graph_) : graph(graph_) {}

  std::vector<Function> collectFunctions(std::string&& functionName);

  std::vector<LoopInfo> collectLoops(Function& func);
  std::vector<Value> collectInputsAndOutputs();
  DType getDataType(std::string dtype);
  
  // Primitives
  std::vector<Loop> split(Loop forOp, 
    int num_output, const std::vector<int>& factors);
  // move the inner loop to the outer is always true;
  void reorder(std::vector<Loop> loopsOrder);
  void bind(Loop forOp, GPUArch level);

  // The write buffer are ususlly private to each writer and only be writen once by its owner.
    // To store temperary variable.
    // So write buffer usually placed to registers.
  // The read buffer are ususlly public to multiple readers and read multiple times by some readers.
    // To achieve data reuse.
    // So read buffer ususlly placed at shared memory in GPU architecture, so need additional param
      // `transpose` to decide whether this buffer should be transposed to resolve bank conflicts.

  // The src of cache_write can be read and write
  Value cache_write(Value src, MemorySpace ms, Loop declare_at, Loop compute_at);
  // The src of cache_read only can be read
  Value cache_read(Value src, MemorySpace ms, Loop declare_at, Loop compute_at, bool transpose);
  Value alloc_buffer(Function& func, MemorySpace ms, std::vector<int64_t> l, std::string dtype);

  Placeholder vectorize(Placeholder& src, uint32_t vectorWidth);

  void memcpy(Tensor& dst, Tensor& src, Loop& compute_at);


private:
  
  int load_length_per_thread {4};
  void loweringAffineLoadStore();
  ComputeDAG* graph {nullptr};
};

}