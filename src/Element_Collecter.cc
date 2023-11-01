#include "Element_Collecter.h"
#include "utils.h"

// used to sample static information

using namespace mlir;
using namespace KernelCodegen;

namespace
{
	//----------------------------------------------- collect functions---------------------------------//
	static std::vector<mlir::func::FuncOp> funcs;

	struct CollectFuncOp : public PassWrapper<CollectFuncOp, OperationPass<ModuleOp>>
	{
		MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectFuncOp)
		CollectFuncOp(std::string &name_, ConstPassGuard *passGuard_) : name(name_), passGuard(passGuard_) {}
		CollectFuncOp(ConstPassGuard *passGuard_) : passGuard(passGuard_) { name.clear(); }
		void runOnOperation() override;
		std::string name;
		ConstPassGuard *passGuard;
	};

	void CollectFuncOp::runOnOperation()
	{
		ModuleOp module = getOperation();

		if (passGuard->visited())
			return;
		passGuard->visit();

		module.walk<WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp)
																		 {
    if(!name.empty()) {
			// get the name of the func::FuncOp
			std::string funcName {funcOp.getSymName()};
			if (funcName.find(name) != std::string::npos) {
				funcs.push_back(funcOp);
			}        
    }
    else
      funcs.push_back(funcOp); });
	}

	std::unique_ptr<OperationPass<ModuleOp>> CollectFunctionsPass(
			std::string &name, ConstPassGuard *passGuard)
	{
		return std::make_unique<CollectFuncOp>(name, passGuard);
	}
	std::unique_ptr<OperationPass<ModuleOp>> CollectFunctionsPass(ConstPassGuard *passGuard)
	{
		return std::make_unique<CollectFuncOp>(passGuard);
	}
}

namespace KernelCodegen
{

std::vector<mlir::func::FuncOp> Ele_Collecter::collectFunctions(std::string &&functionName)
{
	ConstPassGuard passGuard;
	PassManager pm(graph->module.getContext());
	funcs.clear();
	pm.addPass(CollectFunctionsPass(functionName, &passGuard));
	if (failed(pm.run(graph->module)))
	{
		llvm::errs() << "Collects functions failed.\n";
	}
	return funcs;
}

std::vector<mlir::func::FuncOp> Ele_Collecter::collectFunctions()
{
	ConstPassGuard passGuard;
	PassManager pm(graph->module.getContext());
	funcs.clear();
	pm.addPass(CollectFunctionsPass(&passGuard));
	if (failed(pm.run(graph->module)))
	{
		llvm::errs() << "Collects functions failed.\n";
	}
	return funcs;
}

}