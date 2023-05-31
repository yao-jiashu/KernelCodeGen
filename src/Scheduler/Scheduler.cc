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

}

namespace KernelCodegen {

std::vector<LoopInfo> Scheduler::collectLoops() {
  loopInfos.clear();
  ConstPassGuard passGuard;
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(CollectAffineForOpPass(&passGuard));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Collects loops information failed.";
  }
  return loopInfos;
}

}

///TODO: lowering affine load/store to memref load/store
// /// Convert all parallel affine.for op into 1-D affine.parallel op.
// struct AffineParallelize : public AffineParallelizeBase<AffineParallelize> {
//    AffineParallelize(GEMMParamConfig& gemmPC_) : gemmPC(gemmPC_) {}
//    void runOnOperation() override;
//    bool isGEMMLoopParallel(
//    AffineForOp forOp,
//    SmallVectorImpl<LoopReduction> *parallelReductions = nullptr);
//    GEMMParamConfig gemmPC;
// };
// /// Returns true if `forOp' is a parallel loop. If `parallelReductions` is
// /// provided, populates it with descriptors of the parallelizable reductions and
// /// treats them as not preventing parallelization.
// bool AffineParallelize::isGEMMLoopParallel(
//    AffineForOp forOp,
//    SmallVectorImpl<LoopReduction> *parallelReductions) {
   
//    unsigned numIterArgs = forOp.getNumIterOperands();

//    // Loop is not parallel if it has SSA loop-carried dependences and reduction
//    // detection is not requested.
//    if (numIterArgs > 0 && !parallelReductions)
//       return false;

//    // Find supported reductions of requested.
//    if (parallelReductions) {
//       getSupportedReductions(forOp, *parallelReductions);
//       // Return later to allow for identifying all parallel reductions even if the
//       // loop is not parallel.
//       if (parallelReductions->size() != numIterArgs)
//          return false;
//    }

//    // // Check memory dependences.
//    // return isLoopMemoryParallel(forOp);

//    auto getIterTimes = [](AffineForOp& forOp)->int64_t {
//       auto up = forOp.getConstantUpperBound();
//       auto low = forOp.getConstantLowerBound();
//       auto step = forOp.getStep();
//       return (up-low)/step;
//    };
//    auto isGridDim = [&] (AffineForOp forOp)->bool {
//       auto times = getIterTimes(forOp);
//       auto gridDim = gemmPC.getGridDim();

//       // gridDim.y
//       auto parentOp = forOp->getParentOp();
//       if (isa<func::FuncOp>(parentOp)) {
//          if (times == gridDim[0])
//             return true;
//       }
//       // gridDim.x      
//       if (isa<func::FuncOp>(parentOp->getParentOp())) {
//          if (times == gridDim[1])
//             return true;
//       }
//       return false;
//    };
//    auto istBlockDim = [&](AffineForOp forOp)->bool {
//       auto times = getIterTimes(forOp);
//       auto blockDim = gemmPC.getBlockDim();

//       // blockDim.y
//       if (times == blockDim[0]) {
//          auto sonOp = forOp.begin();
//          if (isa<AffineForOp>(sonOp)) {
//             auto sonForOp = dyn_cast<AffineForOp>(sonOp);
//             if (getIterTimes(sonForOp) == blockDim[1])
//                return true;
//          }
//       }
//       // blockDim.x
//       if (times == blockDim[1]) {
//          auto parentOp = forOp->getParentOp();
//          if (isa<AffineForOp>(parentOp)) {
//             auto parentForOp = dyn_cast<AffineForOp>(parentOp);
//             if (getIterTimes(parentForOp) == blockDim[0])
//                return true;
//          }
//       }
//       return false;      
//    };
//    if (isGridDim(forOp) or istBlockDim(forOp)) return true;
//    else return false;
// }
// /// Descriptor of a potentially parallelizable loop.
// struct ParallelizationCandidate {
//   ParallelizationCandidate(AffineForOp l, SmallVector<LoopReduction> &&r)
//       : loop(l), reductions(std::move(r)) {}

//   /// The potentially parallelizable loop.
//   AffineForOp loop;
//   /// Desciprtors of reductions that can be parallelized in the loop.
//   SmallVector<LoopReduction> reductions;
// };

// /// Convert an "affine.apply" operation into a sequence of arithmetic
// /// operations using the StandardOps dialect.
// class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
// public:
//   using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(AffineApplyOp op,
//                                 PatternRewriter &rewriter) const override {
//     auto maybeExpandedMap =
//         expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
//                         llvm::to_vector<8>(op.getOperands()));
//     if (!maybeExpandedMap)
//       return failure();
//     rewriter.replaceOp(op, *maybeExpandedMap);
//     return success();
//   }
// };

// /// Apply the affine map from an 'affine.load' operation to its operands, and
// /// feed the results to a newly created 'memref.load' operation (which replaces
// /// the original 'affine.load').
// class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
// public:
//   using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(AffineLoadOp op,
//                                 PatternRewriter &rewriter) const override {
//     // Expand affine map from 'affineLoadOp'.
//     SmallVector<Value, 8> indices(op.getMapOperands());
//     auto resultOperands =
//         expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
//     if (!resultOperands)
//       return failure();

//     // Build vector.load memref[expandedMap.results].
//     rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
//                                                 *resultOperands);
//     return success();
//   }
// };


// /// Apply the affine map from an 'affine.store' operation to its operands, and
// /// feed the results to a newly created 'memref.store' operation (which replaces
// /// the original 'affine.store').
// class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
// public:
//   using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(AffineStoreOp op,
//                                 PatternRewriter &rewriter) const override {
//     // Expand affine map from 'affineStoreOp'.
//     SmallVector<Value, 8> indices(op.getMapOperands());
//     auto maybeExpandedMap =
//         expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
//     if (!maybeExpandedMap)
//       return failure();

//     // Build memref.store valueToStore, memref[expandedMap.results].
//     rewriter.replaceOpWithNewOp<memref::StoreOp>(
//         op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
//     return success();
//   }
// };


// } // namespace

// void AffineParallelize::runOnOperation() {
// //   func::FuncOp f = getOperation();

// //   // The walker proceeds in pre-order to process the outer loops first
// //   // and control the number of outer parallel loops.
// //   std::vector<ParallelizationCandidate> parallelizableLoops;
// //   f.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
// //     SmallVector<LoopReduction> reductions;
// //     if (isGEMMLoopParallel(loop, parallelReductions ? &reductions : nullptr))
// //       parallelizableLoops.emplace_back(loop, std::move(reductions));
// //   });
  
// //   llvm::errs() << "Affine Parallelize ops = " << parallelizableLoops.size() << "\n";
// //   for (const ParallelizationCandidate &candidate : parallelizableLoops) {
// //     unsigned numParentParallelOps = 0;
// //     AffineForOp loop = candidate.loop;
// //     for (Operation *op = loop->getParentOp();
// //          op != nullptr && !op->hasTrait<OpTrait::AffineScope>();
// //          op = op->getParentOp()) {
// //       if (isa<AffineParallelOp>(op))
// //         ++numParentParallelOps;
// //     }

// //     if (numParentParallelOps < maxNested) {
// //       if (failed(affineParallelize(loop, candidate.reductions))) {
// //         LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] failed to parallelize\n"
// //                                 << loop);
// //          llvm::errs() << "[" DEBUG_TYPE "] failed to parallelize : "  << loop << "\n";
// //       }
      
// //     } else {
// //       LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] too many nested loops\n"
// //                               << loop);

// //       llvm::errs() << "[" DEBUG_TYPE "] too many nested loops\n"
// //                               << loop;
// //     }
// //   }

// /// new
//     RewritePatternSet patterns(&getContext());
//   // clang-format off
//     patterns.add<
//       AffineApplyLowering,
//     //   AffineDmaStartLowering,
//     //   AffineDmaWaitLowering,
//       AffineLoadLowering,
//     //   AffineMinLowering,
//     //   AffineMaxLowering,
//     //   AffineParallelLowering,
//     //   AffinePrefetchLowering,
//       AffineStoreLowering
//     //   AffineForLowering,
//     //   AffineIfLowering,
//     //   AffineYieldOpLowering
//       >(patterns.getContext());
//   // clang-format on
//     populateAffineToVectorConversionPatterns(patterns);
//     ConversionTarget target(getContext());
//     target.addLegalOp<AffineYieldOp, AffineForOp>();
//     target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
//                            scf::SCFDialect, vector::VectorDialect>();
//     if (failed(applyPartialConversion(getOperation(), target,
//                                       std::move(patterns))))
//       signalPassFailure();
// }

// std::unique_ptr<OperationPass<func::FuncOp>>
// createLoweringAffinePass(GEMMParamConfig& gemmPC) {
//   return std::make_unique<AffineParallelize>(gemmPC);
// }