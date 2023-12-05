#include "KernelCodeGen.h"

namespace KernelCodeGen {
  mlir::ModuleOp& KernelCodeGenerator::optimize(ComputeDAG& graph_) {
    graph = graph_;
    mlir::Operation *cloned = graph.module->clone();
    auto module = mlir::dyn_cast<mlir::ModuleOp>(cloned);

    saveBestModule(module);

    // if (module == bestModule) std::cout << "滑天下之大稽\n";
    // else std::cout << "有点人样\n";

    for (auto& opt : opts) {
      backupModule(module);
      if (*opt == MatmulOprimizer()) {
        for (auto& matmulConfig : matmulConfigs) {
          MatmulOprimizer::matmulConfig = matmulConfig;
          resetModule(module);
          if (opt->applicable(module)) {
            opt->applyOptimzer(module, builder);
            auto curLatency = evaluate(module);
            if (curLatency < minLatency) {
              minLatency = curLatency;
              saveBestModule(module);
            }
          }
        }
      } else if (opt->applicable(module)) {
        opt->applyOptimzer(module, builder);
        auto curLatency = evaluate(module);
        if (curLatency < minLatency) {
          minLatency = curLatency;
          saveBestModule(module);
        }
      }
    }
    return bestModule;
  }
}