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

int Analyzer::getUsersNumber(mlir::Value::user_range users) {
  int count = 0;
  for (auto user : users) {
    count += 1;
  }
  return count;
}


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

std::vector<mlir::func::FuncOp> Analyzer::collectFunctions(mlir::ModuleOp& module, const std::string& targetFuncName) {
  std::vector<mlir::func::FuncOp> result;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    // auto funcAttrName = funcOp.getFunctionTypeAttrName(); ///<function_type
    auto funcAttrName = funcOp.getSymName();
    auto funcName = funcAttrName.str();
    if (targetFuncName == "") result.push_back(funcOp);
    else if (funcName.find(targetFuncName) != std::string::npos) {
      result.push_back(funcOp);
    }
  });
  return std::move(result);
}

mlir::func::FuncOp Analyzer::getTargetFunction(mlir::ModuleOp& module, const std::string& targetFuncName) {
  mlir::func::FuncOp res;
  bool found = false;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    auto funcAttrName = funcOp.getSymName();
    auto funcName = funcAttrName.str();
    if (funcName == targetFuncName) {
      res = funcOp;
      found = true;
    }
  });
  if (!found) {
    llvm::errs() << "Failed get the function which name is " << targetFuncName << "\n";
  }
  return res;
}

std::vector<mlir::AffineForOp> Analyzer::collectFuncLoops(mlir::func::FuncOp funcOp) {
  std::vector<mlir::AffineForOp> res;
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
    res.push_back(forOp);
  });
  return std::move(res);
}

std::vector<mlir::func::CallOp> Analyzer::collectFuncCalls(mlir::ModuleOp& module) {
  std::vector<mlir::func::CallOp> res;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::CallOp callOp) {
    res.push_back(callOp);
  });
  return std::move(res);
}

}