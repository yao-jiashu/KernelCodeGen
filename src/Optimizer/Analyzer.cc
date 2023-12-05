#include "Optimizer/Analyzer.h"

struct ConstPassGuard {
  ConstPassGuard() { visitor = 0;}
  ~ConstPassGuard() { visitor = 1;}
  void visit() { visitor = 1;}
  bool visited() { return visitor; }
  int visitor = 0;
};

struct CollectOutermostLoop : 
  public mlir::PassWrapper<CollectOutermostLoop, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectOutermostLoop)
   CollectOutermostLoop(ConstPassGuard* passGuard_, std::vector<mlir::AffineForOp>& res_) : 
        passGuard(passGuard_), outermostForOps(res_) {}
   void runOnOperation() override;
   ConstPassGuard* passGuard;
   std::vector<mlir::AffineForOp>& outermostForOps;
};

void CollectOutermostLoop::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  
  if (passGuard->visited()) return;
  passGuard->visit();

  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
    if (forOp->getParentOp() == module) {
      outermostForOps.push_back(forOp);
    }
  });
}


std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> collectOutermostLoopPass(
  ConstPassGuard* passGuard, std::vector<mlir::AffineForOp>& res) {
   return std::make_unique<CollectOutermostLoop>(passGuard, res);
}

namespace KernelCodeGen {

std::vector<mlir::AffineForOp> Analyzer::collectOutermostLoop(mlir::ModuleOp& module) {
  ConstPassGuard constPassGuard;
  mlir::PassManager pm(module.getContext());
  std::vector<mlir::AffineForOp> res;
  pm.addPass(collectOutermostLoopPass(&constPassGuard, res));
  if (failed(pm.run(module))) {
    llvm::errs() << "Collects outermost loop failed.\n";
  }
  return res;
}

}