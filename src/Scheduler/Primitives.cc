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
              ivReplacement->setAttr("affine.compute_for", builder.getStringAttr("address"));
            }
            if (i > 0) {
              auto factor = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), factors[i - 1]);
              factor->setAttr("affine.compute_for", builder.getStringAttr("address"));
              if (j != 0) {
                mul = builder.create<arith::MulIOp>(builder.getUnknownLoc(), 
                        ivReplacement.getResult(), factor.getResult());
                mul->setAttr("affine.compute_for", builder.getStringAttr("address"));
              } else {
                mul = builder.create<arith::MulIOp>(builder.getUnknownLoc(), 
                        ivs[j], factor.getResult());
                mul->setAttr("affine.compute_for", builder.getStringAttr("address"));
              }
            }
          }
        }
      );
      AffineForOp innermostForOp;
      auto attr = forOp->getAttr("compute_dag.loop_attr");
      auto prevNode = forOp->getPrevNode();
      AffineForOp outermostForOp = dyn_cast<AffineForOp>(prevNode);
      outermostForOp.walk<WalkOrder::PreOrder>([&](AffineForOp newLoop) {
        loops.push_back(newLoop);
        innermostForOp = newLoop;
        newLoop->setAttr("compute_dag.loop_attr", attr);
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
      // auto attr = expectOutermost->getAttr("compute_dag.loop_attr");
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
      OpBuilder builder(forOp.getContext());
      forOp->setAttr(std::string("gpu.parallel_arch"),
        builder.getStringAttr(attrName));
      
      // // Parallel this AffineForOp, but will drop attributes, need you add again
      // SmallVector<LoopReduction> reductions;
      // affineParallelize(forOp, reductions);
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
    CacheWrite(Value src_, MemorySpace ms_, AffineForOp where_) 
      : src(src_), ms(ms_), where(where_) {}
    void runOnOperation() override;

    Value src;
    MemorySpace ms; 
    AffineForOp where;
};

void collectMutableOperands(Value operand) {
  for (auto operand_ : localConstantsOperands) {
    if (operand_ == operand) {
      return;
    }
  }
  // auto owner = operand->getOwner();
  Operation* owner = operand.getDefiningOp();
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
  func.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {

    if (!found) {
      if (forOp == where) found = true;
      localConstantsOperands.push_back(forOp.getInductionVar());
      outerAffineForOps.push_back(forOp);
    }
  });
  auto users = src.getUsers();
  if (users.empty()) return;
  std::vector<Value> indices;
  bool init = false;
  bool existLoad = false;
  bool existStore = false;
  for (auto user : users) {
    auto loadOp = dyn_cast<memref::LoadOp>(user);
    if (loadOp != nullptr) {
      existLoad = true;
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
      existStore = true;
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
    collectMutableOperands(operand);
  }
  assert(existStore);
}

std::unique_ptr<OperationPass<func::FuncOp>> 
CacheWritePass(Value src, MemorySpace ms, AffineForOp where) {
  return std::make_unique<CacheWrite>(src, ms, where);
}

}

namespace KernelCodegen {

std::vector<AffineForOp> Scheduler::split(AffineForOp forOp, 
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


void Scheduler::reorder(std::vector<AffineForOp> loopsOrder) {
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(ReorderAffineForOpPass(loopsOrder));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Reorder loop failed.";
  }
  return;
}

void Scheduler::bind(AffineForOp forOp, GPUArch level) {
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(BindArchTag2AffineForOpPass(forOp, level));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Bind Arch Tag to loop failed.";
  }
  return;
}

Value Scheduler::cache_write(Value src, MemorySpace ms, AffineForOp where) {
  found = false;
  localConstantsOperands.clear();
  outerAffineForOps.clear();
  innerAffineFOrOps.clear();
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(CacheWritePass(src, ms, where));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Cache write failed.";
  }
  return cacheWriteResult;

}

}
