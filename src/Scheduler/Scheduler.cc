#include "Scheduler.h"
#include "utils.h"

// used to sample static information

using namespace mlir;
using namespace KernelCodegen;
namespace {


static std::vector<LoopInfo> loopInfos;

void DFS(Operation* op, int scope) {
  LoopInfo loopInfo;
  loopInfo.scope = scope;
  if (isa<AffineForOp>(*op)) {
    scope += 1;
    auto forOp = dyn_cast<AffineForOp>(*op);
    loopInfo.forOp = forOp;
    OpBuilder builder(forOp.getContext());

    auto attr = forOp->getAttr(builder.getStringAttr("compute_dag.loop_attr"));
    if (attr != nullptr) {
      auto attrStr =attr.dyn_cast<StringAttr>().str();
      if (attrStr == "spatial") loopInfo.attibute = LoopAttribute::spatial;
      else if (attrStr == "reduction") loopInfo.attibute = LoopAttribute::reduction;
      else assert(false);
    }
    else {
      assert(false);
    }
    loopInfos.emplace_back(loopInfo);
  }
  int numRegion = op->getNumRegions();
  if (numRegion != 0) {
    auto regions = op->getRegions();
    for (auto& region : regions) {
      auto& blocks = region.getBlocks();
      for (auto& block : blocks) {
        auto& ops = block.getOperations();
        for (auto& op : ops) {
          DFS(&op, scope);
        }
      }
    }
  }
}

struct CollectAffineForOp : 
  public PassWrapper<CollectAffineForOp, OperationPass<func::FuncOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectAffineForOp)
   CollectAffineForOp(ConstPassGuard* passGuard_) : passGuard(passGuard_) {}
   void runOnOperation() override;
   ConstPassGuard* passGuard;
};

void CollectAffineForOp::runOnOperation() {
  func::FuncOp func = getOperation();
  
  if (passGuard->visited()) return;
  passGuard->visit();
  // depth-first search
  DFS(func, 0);
}

std::unique_ptr<OperationPass<func::FuncOp>> CollectAffineForOpPass(
  ConstPassGuard* passGuard) {
   return std::make_unique<CollectAffineForOp>(passGuard);
}

/// lowering affine load/store to memref load/store
struct LoweringAffineLoadStore
 : public PassWrapper<LoweringAffineLoadStore, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringAffineLoadStore)
   LoweringAffineLoadStore() {}
   void runOnOperation() override;
};
 
/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};


/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

void LoweringAffineLoadStore::runOnOperation() {
    RewritePatternSet patterns(&getContext());
  // clang-format off
    patterns.add<
      AffineLoadLowering,
      AffineStoreLowering
      >(patterns.getContext());
  // clang-format on
    ConversionTarget target(getContext());
    target.addLegalOp<AffineYieldOp, AffineForOp>();
    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLoweringAffinePass() {
  return std::make_unique<LoweringAffineLoadStore>();
}


static std::vector<Value> insAndOuts;

struct CollectInsAndOuts : 
  public PassWrapper<CollectInsAndOuts, OperationPass<func::FuncOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectInsAndOuts)
   CollectInsAndOuts(ConstPassGuard* passGuard_) : passGuard(passGuard_) {}
   void runOnOperation() override;
   ConstPassGuard* passGuard;
};

void CollectInsAndOuts::runOnOperation() {
  if (passGuard->visited()) return;
  passGuard->visit();
  func::FuncOp func = getOperation();

  auto args = func.getArguments();
  for (auto& arg : args) {
    auto type = arg.getType();
    if(type.isa<MemRefType>()) {
      insAndOuts.push_back(arg);
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CollectInsAndOutsPass(
  ConstPassGuard* passGuard) {
   return std::make_unique<CollectInsAndOuts>(passGuard);
}

}

namespace KernelCodegen {

/// @brief TODO: optimize this function
/// @return 
std::vector<LoopInfo> Scheduler::collectLoops() {
  loopInfos.clear();
  ConstPassGuard passGuard;
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(CollectAffineForOpPass(&passGuard));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Collects loops information failed.\n";
  }
  return loopInfos;
}

void Scheduler::loweringAffineLoadStore() {
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(createLoweringAffinePass());
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Lowering Affine Load/Store Op failed.\n";
  }
}

std::vector<Value> Scheduler::collectInputsAndOutputs() {
  insAndOuts.clear();
  ConstPassGuard passGuard;
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(CollectInsAndOutsPass(&passGuard));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Collects inputs and outpus information failed.\n";
  }
  return insAndOuts;
}

}