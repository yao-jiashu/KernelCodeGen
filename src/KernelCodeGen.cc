#include "KernelCodeGen.h"
#include "log.h"

namespace KernelCodeGen {

Log KCGLog::level = Log::Debug;

mlir::ModuleOp& KernelCodeGenerator::optimize(ComputeDAG& graph_) {
  graph = graph_;
  mlir::Operation *cloned = graph.module->clone();
  auto module = mlir::dyn_cast<mlir::ModuleOp>(cloned);

  saveBestModule(module);

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