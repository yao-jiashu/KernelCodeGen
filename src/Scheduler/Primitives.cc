#include "Scheduler.h"
#include "utils.h"

// used to change the IR structure

using namespace mlir;
using namespace KernelCodegen;
namespace {


static std::vector<AffineForOp> loops;

struct SplitTargetAffineForOp : 
  public PassWrapper<SplitTargetAffineForOp, OperationPass<func::FuncOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitTargetAffineForOp)
   SplitTargetAffineForOp(AffineForOp targetForOp_, int num_output_, std::vector<int> factors_) 
    : targetForOp(targetForOp_), num_output(num_output_), factors(factors_) {}
   void runOnOperation() override;

   AffineForOp targetForOp;
   int num_output;
   std::vector<int> factors;
};

void SplitTargetAffineForOp::runOnOperation() {
  
  func::FuncOp func = getOperation();

  func.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {
    if (forOp == targetForOp) {
      auto lowerbound = forOp.getLowerBoundMap();
      auto upperbound = forOp.getUpperBoundMap();
      int step = forOp.getStep();
      assert(lowerbound.isConstant() == true);
      assert(upperbound.isConstant() == true);
      int64_t lb = lowerbound.getSingleConstantResult();
      int64_t ub = upperbound.getSingleConstantResult();
      assert(step == 1 && lb == 0);

      // create op before the forOp
      OpBuilder builder(forOp.getOperation());

  
      std::vector<int> ups;
      ups.reserve(num_output);
      int64_t mul = 1;
      for (auto factor : factors) {
        ups.push_back(factor / mul);
        mul = factor;
      }
      ups.push_back(ub / mul);
      factors = ups;
      std::reverse(std::begin(ups), std::end(ups));
    
      // build loops
      assert(num_output <= 8);
      SmallVector<int64_t, 8> lowerBounds(num_output, /*Value=*/0);
      SmallVector<int64_t, 8> steps(num_output, /*Value=*/1);
      // SmallVector<int64_t, 8> upperBounds(ups);
      SmallVector<int64_t, 8> upperBounds(ups.begin(), ups.end());
      arith::AddIOp ivReplacement;
      buildAffineLoopNest(
         builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // ivReplacement = builder.create<arith::AddIOp>(builder.getUnknownLoc(), 
          //  ivOuter, ivInner);
          arith::MulIOp mul;
          for (int i = num_output - 1, j = 0; i >= 0; i--, j++) {
            if (j != 0) {
              ivReplacement = builder.create<arith::AddIOp>(builder.getUnknownLoc(), 
                                mul.getResult(), ivs[j]);
              // means should move with parent affine for Op
              // ivReplacement->setAttr("affine.compute_for", builder.getStringAttr("address"));
            }
            if (i > 0) {
              auto factor = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), factors[i - 1]);
              // factor->setAttr("affine.compute_for", builder.getStringAttr("address"));
              if (j != 0) {
                mul = builder.create<arith::MulIOp>(builder.getUnknownLoc(), 
                        ivReplacement.getResult(), factor.getResult());
                // mul->setAttr("affine.compute_for", builder.getStringAttr("address"));
              } else {
                mul = builder.create<arith::MulIOp>(builder.getUnknownLoc(), 
                        ivs[j], factor.getResult());
                // mul->setAttr("affine.compute_for", builder.getStringAttr("address"));
              }
            }
          }
        }
      );
      AffineForOp innermostForOp;
      auto attr = forOp->getAttr("schedule.loop_attr");
      auto prevNode = forOp->getPrevNode();
      AffineForOp outermostForOp = dyn_cast<AffineForOp>(prevNode);
      outermostForOp.walk<WalkOrder::PreOrder>([&](AffineForOp newLoop) {
        loops.push_back(newLoop);
        innermostForOp = newLoop;
        newLoop->setAttr("schedule.loop_attr", attr);
      });

      // erase the yield op
      innermostForOp.getBody()->back().erase();
      
      innermostForOp.getBody()->getOperations().splice(
        innermostForOp.getBody()->end(),
        forOp.getBody()->getOperations());
      auto oldIv = forOp.getInductionVar();
      oldIv.replaceAllUsesWith(ivReplacement.getResult());
      forOp.erase();
    }
    
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> 
SplitTargetAffineForOpPass(AffineForOp forOp, int num_output,  std::vector<int> factors) {
  return std::make_unique<SplitTargetAffineForOp>(forOp, num_output, factors);
}

struct ReorderAffineForOp : 
  public PassWrapper<ReorderAffineForOp, OperationPass<func::FuncOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReorderAffineForOp)
   ReorderAffineForOp(std::vector<AffineForOp>& loopsOrder) 
    : targetOrder(loopsOrder) {}
   void runOnOperation() override;

   std::vector<AffineForOp> targetOrder;
};

void ReorderAffineForOp::runOnOperation() {
  
  func::FuncOp func = getOperation();

  while (targetOrder.size() > 1 ) {
    AffineForOp outermostForOp;
    bool found = false;
    func.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {
      if (!found) {
        for (int i = 0; i < targetOrder.size(); i++) {
          if (forOp == targetOrder[i]) {
            found = true;
            outermostForOp = forOp;
            return;
          }
        }
      }

    });
    assert(found);
    if (outermostForOp != targetOrder[0]) {
      // extract expect outermost for op' body to parent op
      AffineForOp expectOutermost = targetOrder[0];
      OpBuilder builder(expectOutermost.getContext());

      ///TODO: Consider if there are other ops in the same block as the reduction affine for
      // auto attr = expectOutermost->getAttr("schedule.loop_attr");
      // if (attr == builder.getStringAttr("reduction")) {

      //   // auto prevOp = expectOutermost->getPrevNode();
      //   // if (!(prevOp->hasAttr(builder.getStringAttr("affine.compute_for")) && 
      //   //     prevOp->getAttr(builder.getStringAttr("affine.compute_for")) ==
      //   //       builder.getStringAttr("address"))) {
      //   //   auto cloned = outermostForOp.clone();
      //   //   outermostForOp->getBlock()->getOperations().insert(
      //   //     Block::iterator(outermostForOp),
      //   //     cloned.getOperation());
      //   //   auto ops = cloned.getBody()->getOperations().erase()    
      //   // }

      //   auto cloned = outermostForOp.clone();
      //   if (cloned == outermostForOp) {
      //     llvm::errs() << "They are same!\n";
      //   }
      //   outermostForOp->getBlock()->getOperations().insert(
      //     std::next(Block::iterator(outermostForOp)),
      //     cloned.getOperation());
      // }

      // erase the yield op
      expectOutermost.getBody()->back().erase();

      // this block contain the expectOutermost Op
      expectOutermost->getBlock()->getOperations().splice(
        Block::iterator(expectOutermost),
        expectOutermost.getBody()->getOperations());

      // move expect outermost for op to new position
      expectOutermost->moveBefore(outermostForOp);

      auto commonParentOp = outermostForOp->getParentOp();
      // move the entile body of outermostForOp into expectOutermost
      expectOutermost.getBody()->getOperations().splice(
        expectOutermost.getBody()->end(),
        outermostForOp->getBlock()->getOperations(),
        Block::iterator(outermostForOp));

      OpBuilder::InsertionGuard nestedGuard(builder);
      builder.setInsertionPointToEnd(expectOutermost.getBody());
      builder.create<AffineYieldOp>(builder.getUnknownLoc());

    }
    targetOrder.erase(targetOrder.begin());
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> 
ReorderAffineForOpPass(std::vector<AffineForOp>& loopsOrder) {
  return std::make_unique<ReorderAffineForOp>(loopsOrder);
}

struct BindArchTag2AffineForOp : 
  public PassWrapper<BindArchTag2AffineForOp, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindArchTag2AffineForOp)
    BindArchTag2AffineForOp(AffineForOp forOp, GPUArch level) 
      : targetOp(forOp), targetArch(level) {}
    void runOnOperation() override;

    AffineForOp targetOp;
    GPUArch targetArch;
};

void BindArchTag2AffineForOp::runOnOperation() {
  
  func::FuncOp func = getOperation();
  func.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {
    if (forOp == targetOp) {
      auto attrName = getGPUArchStr(targetArch);
      // OpBuilder builder(forOp.getContext());
      OpBuilder builder(forOp.getRegion());

      forOp->setAttr(std::string("gpu.parallel_arch"),
        builder.getStringAttr(attrName));

      // auto iter = forOp.getInductionVar();

      // mlir::OpBuilder::InsertionGuard nestedGuard(builder);

      // switch (targetArch) {
      //   case GPUArch::blockIdxX: {
      //     auto blockIdxX = builder.create<gpu::BlockIdOp>(builder.getUnknownLoc(), builder.getIndexType(), gpu::Dimension::x);
      //     iter.replaceAllUsesWith(blockIdxX);
      //     break;
      //   }
      //   case GPUArch::blockIdxY: {
      //     auto blockIdxY = builder.create<gpu::BlockIdOp>(builder.getUnknownLoc(), builder.getIndexType(), gpu::Dimension::y);
      //     iter.replaceAllUsesWith(blockIdxY);
      //     break;
      //   }
      //   case GPUArch::threadIdxX: {
      //     auto threadIdxX = builder.create<gpu::ThreadIdOp>(builder.getUnknownLoc(), builder.getIndexType(), gpu::Dimension::x);
      //     iter.replaceAllUsesWith(threadIdxX);
      //     break;
      //   }
      //   case GPUArch::threadIdxY: {
      //     auto threadIdxY = builder.create<gpu::ThreadIdOp>(builder.getUnknownLoc(), builder.getIndexType(), gpu::Dimension::y);
      //     iter.replaceAllUsesWith(threadIdxY);
      //     break;
      //   }
      //   default:
      //     llvm::errs() << "Unsupport index type\n";
      // }
    }
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> 
BindArchTag2AffineForOpPass(AffineForOp forOp, GPUArch level) {
  return std::make_unique<BindArchTag2AffineForOp>(forOp, level);
}

static bool found = false;
static std::vector<Value> localConstantsOperands;
static std::vector<AffineForOp> outerAffineForOps;
static std::vector<AffineForOp> innerAffineFOrOps;
static Value cacheWriteResult;

struct CacheWrite : 
  public PassWrapper<CacheWrite, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheWrite)
    CacheWrite(Value src_, MemorySpace ms_, AffineForOp declare_at_, AffineForOp compute_at_) 
      : src(src_), ms(ms_), declare_at(declare_at_), compute_at(compute_at_) {
        // When we want to cache the write, we need to palce it in the position where it will be used.
        // In general, only a writer will write the position, two writers will bring conflicts
        // So, every writer can cache its write to local scope belong to itself.
        // Unfortunately, cache_read will be more complex.
        assert(declare_at_ == compute_at_);
      }
    void runOnOperation() override;

    Value src;
    MemorySpace ms;
    // Affect the position where the memref.alloc placed 
    AffineForOp declare_at;
    // Afect the size, which loop will use this cache from begin to end
    AffineForOp compute_at;
    std::vector<int> memorySize;
};

void collectMutableOperands(Value operand) {
  for (auto operand_ : localConstantsOperands) {
    if (operand_ == operand) {
      return;
    }
  }
  // auto owner = operand->getOwner();
  Operation* owner = operand.getDefiningOp();
  /// Pay attentiont to it.
  if (operand.dyn_cast<BlockArgument>()) {
    auto blockArgument = operand.dyn_cast<BlockArgument>();
    owner = blockArgument.getOwner()->getParentOp();
  }
  if (dyn_cast<arith::ConstantOp>(owner)) {
    return;
  }
  auto ownerForOp = dyn_cast<AffineForOp>(owner);
  if (ownerForOp) {
    for (auto forOp : outerAffineForOps) {
      if (ownerForOp == forOp) {
        llvm::errs() << "Pay attention, not be supposed!\n";
        return;
      }
    }
    for (auto forOp : innerAffineFOrOps) {
      if (forOp == ownerForOp) {
        return;
      }
    }
    innerAffineFOrOps.push_back(ownerForOp);
    return;
  }
  auto operands = owner->getOperands();
  for (auto operandNest : operands) {
    collectMutableOperands(operandNest);
  }
}

void CacheWrite::runOnOperation() {

  func::FuncOp func = getOperation();
  
  // Step 1: infer size
  func.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {

    if (!found) {
      if (forOp == compute_at) found = true;
      localConstantsOperands.push_back(forOp.getInductionVar());
      outerAffineForOps.push_back(forOp);
    }
  });
  auto users = src.getUsers();
  if (users.empty()) return;
  std::vector<Value> indices;
  bool init = false;
  for (auto user : users) {
    auto loadOp = dyn_cast<memref::LoadOp>(user);
    if (loadOp != nullptr) {
      auto operands = loadOp.getIndices();
      if (!init) {
        init = true;
        int num = operands.size();
        for (int i = 0; i < num; i++) {
          indices.push_back(operands[i]);
        }
      } else {
        int num = operands.size();
        for (int i = 0; i < num; i++) {
          assert(operands[i] == indices[i]);
        }
      }
      continue;
    }

    auto storeOp = dyn_cast<memref::StoreOp>(user);
    if (storeOp != nullptr) {
      auto operands = storeOp.getIndices();
      if (!init) {
        init = true;
        int num = operands.size();
        for (int i = 0; i < num; i++) {
          indices.push_back(operands[i]);
        }
      } else {
        int num = operands.size();
        for (int i = 0; i < num; i++) {
          assert(operands[i] == indices[i]);
        }
      }
      continue;
    } else {
      assert(false);
    }
  }
  for (auto operand : indices) {
    memorySize.push_back(0);
    collectMutableOperands(operand);
  }

  // Step 2: declare

  // mlir::MemRefType tensorShape = mlir::MemRefType::get(
  //   memorySize, dtype, {}, static_cast<int>(ms));
  // OpBuilder builder(declare_at);
  // auto cache_read = builder.create<ComputeDAG::Placeholder>(builder.getUnknownLoc(), tensorShape);

  // Step 3: load


  // Step 4: replace
  // cache_read.replaceAllUsesWith(src);

}

std::unique_ptr<OperationPass<func::FuncOp>> 
CacheWritePass(Value src, MemorySpace ms, AffineForOp declare_at, AffineForOp compute_at) {
  return std::make_unique<CacheWrite>(src, ms, declare_at, compute_at);
}

}

namespace KernelCodegen {

std::vector<Scheduler::Loop> Scheduler::split(Scheduler::Loop forOp, 
  int num_output, const std::vector<int>& factors) {
  loweringAffineLoadStore();
  loops.clear();
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(SplitTargetAffineForOpPass(forOp, num_output, factors));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Split loops failed.";
  }
  return loops;
}


void Scheduler::reorder(std::vector<Scheduler::Loop> loopsOrder) {
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(ReorderAffineForOpPass(loopsOrder));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Reorder loop failed.";
  }
  return;
}

Value Scheduler::bind(Scheduler::Loop forOp, GPUArch level) {
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(BindArchTag2AffineForOpPass(forOp, level));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Bind Arch Tag to loop failed.";
  }
  return forOp.getInductionVar();
}

Value Scheduler::cache_write(Value src, MemorySpace ms, Scheduler::Loop declare_at, Scheduler::Loop compute_at) {
  found = false;
  localConstantsOperands.clear();
  outerAffineForOps.clear();
  innerAffineFOrOps.clear();
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(CacheWritePass(src, ms, declare_at, compute_at));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Cache write failed.";
  }
  return cacheWriteResult;

}

Value Scheduler::alloc_buffer(
  Scheduler::Function& func, MemorySpace ms, std::vector<int64_t> l, std::string dtype) {
  llvm::ArrayRef<int64_t> shape (l);
  auto dtype_ = getDataType(dtype);
  mlir::MemRefType tensorShape = mlir::MemRefType::get(
    shape, dtype_, {}, static_cast<int>(ms));
  
  mlir::OpBuilder::InsertionGuard nestedGuard(graph->builder);
  graph->builder.setInsertionPointToStart(&func.front());
  
  return graph->builder.create<Scheduler::Placeholder>(graph->builder.getUnknownLoc(), tensorShape)->getResult(0);
}



using llvm::SmallMapVector;


// /// Get the number of bits require to store a value of the given shaped type.
// /// Compute the value recursively since tensors are allowed to have vectors as
// /// elements.
// int64_t getSizeInBits(ShapedType type)  {
//   assert(type.hasStaticShape() &&
//          "cannot get the bit size of an aggregate with a dynamic shape");

//   auto elementType = type.getElementType();
//   if (elementType.isIntOrFloat())
//     return elementType.getIntOrFloatBitWidth() * type.getNumElements();

//   if (auto complexType = elementType.dyn_cast<ComplexType>()) {
//     elementType = complexType.getElementType();
//     return elementType.getIntOrFloatBitWidth() * type.getNumElements() * 2;
//   }

//   // Tensors can have vectors and other tensors as elements, other shaped types
//   // cannot.
//   assert(type.isa<TensorType>() && "unsupported element type");
//   assert((elementType.isa<VectorType, TensorType>()) &&
//          "unsupported tensor element type");
//   return type.getNumElements() * elementType.cast<ShapedType>().getSizeInBits();
// }

// /// Given an input type, provides a vector type for it of the provided width.
// static VectorType getVectorizedType(Type inputType, unsigned width) {
//   assert(width > 1 && "unexpected vector width");
//   assert(!inputType.isa<IndexType>() && "index type can't be vectorized");
//   Type baseEltType = inputType;
//   SmallVector<int64_t, 4> vecShape;
//   if (auto vecEltType = inputType.dyn_cast<VectorType>()) {
//     baseEltType = vecEltType.getElementType();
//     vecShape.reserve(vecShape.size() + vecEltType.getRank());
//     vecShape.assign(vecEltType.getShape().begin(), vecEltType.getShape().end());
//   }
//   vecShape.push_back(width);
//   return VectorType::get(vecShape, baseEltType);
// }

// /// Casts a given input memref, uses memref_shape_cast op to cast it to a memref
// /// with an elemental type that is `vector width` times (for eg., f32 becomes
// /// vector<8xf32>, vector<8xf32> becomes vector<8x8xf32> if `vectorWidth` were
// /// to be 8).
// Value createVectorMemRef(Value scalMemRef, unsigned vectorWidth) {
//   auto scalMemRefType = scalMemRef.getType().cast<MemRefType>();
//   auto shape = scalMemRefType.getShape();

//   OpBuilder b(scalMemRef.getContext());
//   if (auto *defOp = scalMemRef.getDefiningOp())
//     b.setInsertionPointAfter(defOp);
//   else
//     b.setInsertionPointToStart(scalMemRef.cast<BlockArgument>().getOwner());

//   auto vecMemRefEltType =
//       getVectorizedType(scalMemRefType.getElementType(), vectorWidth);

//   SmallVector<int64_t, 4> vecMemRefShape(shape.begin(), shape.end());
//   if (vecMemRefShape.back() != -1)
//     vecMemRefShape.back() /= vectorWidth;

//   auto vecMemRefType = MemRefType::get(vecMemRefShape, vecMemRefEltType);

//   // FIXME: we are using a shape cast here, but we do not know whether the base
//   // memref is aligned to the right boundary. The load/stores on cast memref (of
//   // vector elt type) would be mapped to aligned load/stores by default and
//   // lead to a protection fault.
//   // We are going to fix this at least where we have access to the defining
//   // alloc op.
//   if (auto allocOp = dyn_cast_or_null<memref::AllocOp>(scalMemRef.getDefiningOp()))
//     allocOp.alignmentAttr(
//         b.getI64IntegerAttr(getSizeInBits(vecMemRefEltType) / 8));

//   return b.create<enhance::VectorizeOp>(b.getUnknownLoc(), vecMemRefType,
//                                         scalMemRef);
// }

Scheduler::Placeholder Scheduler::vectorize(Scheduler::Placeholder& src, uint32_t vectorWidth) {

}

void Scheduler::memcpy_async(Tensor& dst, Tensor& src, Loop& compute_at, std::vector<Value>& thread_hierarchy, ThreadScope scope) {

  assert(dst.ms.isa<IntegerAttr>() &&
        "Using `getMemorySpaceInteger` with non-Integer attribute");

 auto dst_mem = static_cast<MemorySpace>(dst.ms.cast<IntegerAttr>().getInt());
 auto src_mem = static_cast<MemorySpace>(src.ms.cast<IntegerAttr>().getInt());

 MemcpyDirection direction;
 if (src_mem == MemorySpace::global && dst_mem == MemorySpace::local) {
  direction = MemcpyDirection::global2local;
 } else if (src_mem == MemorySpace::local && dst_mem == MemorySpace::shared) {
  direction = MemcpyDirection::local2shared;
 } else if (src_mem == MemorySpace::local && dst_mem == MemorySpace::global) {
  direction = MemcpyDirection::local2global;
 } else if (src_mem == MemorySpace::shared && dst_mem == MemorySpace::local) {
  direction = MemcpyDirection::shared2local;
 }

 if (direction == MemcpyDirection::global2local && scope == ThreadScope::block) {

  // get total load times in scope;
  auto total_times = src.load_times();

  auto threadIdxY = thread_hierarchy[2];
  auto threadIdxX = thread_hierarchy[3];

  // For BlockArgument, first get the block then get the AffineForOp.
  auto forOp = dyn_cast<Loop>(threadIdxY.getParentBlock()->getParentOp());

  auto blockDimY = forOp.getConstantUpperBound();
  auto blockDimX = forOp.getConstantUpperBound();

  auto total_threads = blockDimY * blockDimX;
  auto loopTimes = total_times / total_threads;

  OpBuilder builder(&compute_at->getRegion(0));

  src.offset = createOpsFromExpressions(src.start, builder);

  /*
  Mapping:
  threadIdx  = threadIdx.y * blockDim.x + threadIdx.x; // thread index to 1 dim
  threads_per_row = size[0] / pack_width;
  y = (row_base + offset[0] + x / threads_per_row, offset[1] + x % threads_per_row;
  */


  auto blockDimXOp = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), blockDimX);

  auto threadBlockBase = builder.create<arith::MulIOp>(builder.getUnknownLoc(), threadIdxY, blockDimXOp.getResult());

  auto threadIdx = builder.create<arith::AddIOp>(builder.getUnknownLoc(), threadBlockBase.getResult(), threadIdxX);

  auto threads_per_row = src.size[1] / src.pack_width;
  auto rowStride = total_threads / threads_per_row;
  auto totalRow = src.total_size() / src.size[1];

  // auto threads_per_row_Op = builder.create<arith::ConstantOp>(
    // builder.getUnknownLoc(), builder.getI64Type(), builder.getIntegerAttr(builder.getIndexType(), threads_per_row));
  auto threads_per_row_Op = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), threads_per_row);


  auto rowOffset = builder.create<arith::DivUIOp>(builder.getUnknownLoc(), threadIdx.getResult(), threads_per_row_Op.getResult());
  auto colOffset = builder.create<arith::RemUIOp>(builder.getUnknownLoc(), threadIdx.getResult(), threads_per_row_Op.getResult());


  auto memcpyLoopBody = [&](OpBuilder &builder, Location nestedLoc, Value iv, ValueRange iterArgs) {
    OpBuilder::InsertionGuard nestedGuard(builder);

    auto rank = src.size.size();

    SmallVector<OpFoldResult, 4> srcOffsets, srcSizes, srcStrides;
    SmallVector<OpFoldResult, 4> dstOffsets, dstSizes, dstStrides;

    srcOffsets.reserve(rank);
    srcSizes.reserve(rank);
    srcStrides.reserve(rank);
    dstOffsets.reserve(rank);
    dstSizes.reserve(rank);
    dstStrides.reserve(rank);

    for (int i = 0; i < rank; i++) {
      srcStrides.push_back(builder.getIndexAttr(1));
      dstStrides.push_back(builder.getIndexAttr(1));
      if (i < rank - 1) {
        srcSizes.push_back(builder.getIndexAttr(1));
        dstSizes.push_back(builder.getIndexAttr(1));
      }
    }
    srcSizes.push_back(builder.getIndexAttr(src.pack_width));
    dstSizes.push_back(builder.getIndexAttr(src.pack_width));

    auto sum1 = builder.create<arith::AddIOp>(builder.getUnknownLoc(), iv, src.offset[0]);
    auto rowIdx = builder.create<arith::AddIOp>(builder.getUnknownLoc(), sum1.getResult(), rowOffset.getResult());
  
    auto colIdx = builder.create<arith::AddIOp>(builder.getUnknownLoc(), src.offset[1], colOffset.getResult());

    srcOffsets.push_back(rowIdx.getResult());
    srcOffsets.push_back(colIdx.getResult());


    auto load_width = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), src.pack_width);
    auto rowStrideOp = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), rowStride);

    auto div = builder.create<arith::DivUIOp>(builder.getUnknownLoc(), iv, rowStrideOp.getResult());
    auto mul = builder.create<arith::MulIOp>(builder.getUnknownLoc(), div.getResult(), load_width.getResult());
    dstOffsets.push_back(builder.getIndexAttr(0));
    dstOffsets.push_back(mul.getResult());

    auto copySrc = builder.create<memref::SubViewOp>(
        builder.getUnknownLoc(), src.memory, srcOffsets, srcSizes, srcStrides);
    auto copyDest = builder.create<memref::SubViewOp>(
        builder.getUnknownLoc(), dst.memory, dstOffsets, dstSizes, dstStrides);
    
    builder.create<memref::CopyOp>(builder.getUnknownLoc(), copySrc, copyDest);
    builder.create<AffineYieldOp>(builder.getUnknownLoc());
  };

  
  auto memcpyLoop = builder.create<Scheduler::Loop>(builder.getUnknownLoc(), 
                  0, totalRow, rowStride, /*iterArgs=lvm::None*/ ValueRange({}), memcpyLoopBody);    

 } else if (direction == MemcpyDirection::local2shared && scope == ThreadScope::warp) {

 } else {
  llvm::errs() << "Unsupport memcopy\n";
 }
}


}
