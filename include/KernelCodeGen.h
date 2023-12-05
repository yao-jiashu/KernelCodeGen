#pragma once

#include "Frontend/Operators.h"
#include "Optimizer/Optimizer.h"

// #include "ComputeDAG.h"
// #include "GraphTune.h"
// #include "AutoConfig.h"
// #include "Optimizer.h"
// #include "Element_Collecter.h"
// #include "Element_Parser.h"
// #include "CodeGen.h"

#include <string>
#include <sstream>
#include <initializer_list>
#include <climits>
#include <cfloat>

namespace KernelCodeGen {

class KernelCodeGenerator {
public:
  KernelCodeGenerator(const std::string& platform_ = {"CUDA"}) : 
    builder(&context),
    graph(builder),
    platform(std::move(platform_)) {
    initMLIRContext();
    opts.push_back(std::move(std::make_unique<MatmulOprimizer>()));
    matmulConfigs = {
      { {"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 8}, {"GROUP_SIZE_M", 8}, 
        {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
    };
  }
  KernelCodeGenerator() = delete;

  void initMLIRContext() {
    // context.getOrLoadDialect<mlir::compute_dag::ComputeDAGDialect>();
    // context.getOrLoadDialect<mlir::schedule::ScheduleDialect>();
    context.getOrLoadDialect<mlir::AffineDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    mlir::registerAllPasses();
  }

  ComputeDAG& createGraph(const std::string& graphName) {
    graph.module = mlir::ModuleOp::create(
      builder.getUnknownLoc(),
      mlir::Optional<mlir::StringRef>(std::move(graphName)));
    graph.builder.setInsertionPointToEnd(graph.module.getBody());
    return graph;
  }

  ComputeDAG& getGraph() {
    return graph;
  }

  void dump(mlir::ModuleOp& module, const std::string& info = "") {
    llvm::outs() << "----------------------------------------------------------\n";
    llvm::outs() << "           " << info << "\n";
    llvm::outs() << "----------------------------------------------------------\n";
    module->dump();
    if (mlir::failed(mlir::verify(module))) {
      module->emitError("module verification error");
      assert(false);
    }
  }

  void resetModule(mlir::ModuleOp& module) {
    mlir::Operation *cloned = backupModule_->clone();
    module = mlir::dyn_cast<mlir::ModuleOp>(cloned);   
  }

  void backupModule(mlir::ModuleOp& module) {
    mlir::Operation *cloned = module->clone();
    backupModule_ = mlir::dyn_cast<mlir::ModuleOp>(cloned);
  }

  void saveBestModule(mlir::ModuleOp& module) {
    mlir::Operation *cloned = module->clone();
    bestModule = mlir::dyn_cast<mlir::ModuleOp>(cloned);    
  }

  mlir::ModuleOp& optimize(ComputeDAG& graph_);

  float evaluate(mlir::ModuleOp& module) {
    return 0.0f;
  }

private:
  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::ModuleOp backupModule_;
  mlir::ModuleOp bestModule;
  ComputeDAG graph;
  std::string platform;
  float minLatency = FLT_MAX;
  std::vector<std::unique_ptr<Optimizer>> opts;
  std::vector<std::map<std::string, int>> matmulConfigs;
};

}