#include "Scheduler.h"

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

      // create op before the forOp
      OpBuilder builder(forOp.getOperation());

      auto attr = forOp->getAttr("compute_dag.loop_attr");

      // arith::AddIOp ivReplacement;
      AffineApplyOp ivReplacement;


      auto outerLoopBody = [&](OpBuilder &builder, Location nestedLoc, Value ivOuter,
                          ValueRange iterArgs) {
        auto innerLoopBody = [&](OpBuilder &builder, Location nestedLoc, Value ivInner,
                          ValueRange iterArgs) {
            auto dim0 = builder.getAffineDimExpr(0);
            auto dim1 = builder.getAffineDimExpr(1);   
            auto addMap = AffineMap::get(/*dimCount*/2, 0,/*AffineExpr*/{dim0 + dim1});             
            // ivReplacement = builder.create<arith::AddIOp>(builder.getUnknownLoc(), 
            //  ivOuter, ivInner);
                
            ivReplacement = builder.create<AffineApplyOp>(builder.getUnknownLoc(),
                        addMap, ValueRange({ivOuter, ivInner}));
            builder.create<AffineYieldOp>(builder.getUnknownLoc());
        };
        auto innerForOp = builder.create<AffineForOp>(builder.getUnknownLoc(), 
          0, factor * step, step, /*iterArgs=lvm::None*/ ValueRange({}), innerLoopBody);
        innerForOp->setAttr("compute_dag.loop_attr", attr);
        builder.create<AffineYieldOp>(builder.getUnknownLoc());
      };
      auto outerForOp = builder.create<AffineForOp>(builder.getUnknownLoc(), 
            lb, ub, step * factor, /*iterArgs=lvm::None*/ ValueRange({}), outerLoopBody);
      ->setAttr("compute_dag.loop_attr", attr);

      auto& innerOp = outerForOp.getBody()->front();
      auto innerForOp = dyn_cast<AffineForOp>(innerOp);

      loops.push_back(outerForOp);
      loops.push_back(innerForOp);

      // erase the yield op
      innerForOp.getBody()->back().erase();
      
      innerForOp.getBody()->getOperations().splice(
        innerForOp.getBody()->end(),
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

}

namespace KernelCodegen {

std::vector<AffineForOp> Scheduler::split(AffineForOp forOp, 
  int num_output, const std::vector<int>& factors) {
  llvm::errs() << "factor = " << factor << "\n";
  loops.clear();
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(SplitTargetAffineForOpPass(forOp, num_output, factors));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Split loop failed.";
  }
  return loops;
}

}
