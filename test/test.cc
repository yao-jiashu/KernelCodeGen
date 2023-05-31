#include <iostream>
#include "KernelCodegen.h"
namespace KC = KernelCodegen;
using KCM = KC::KernelCodegenMachine;


int main(int argc, char* argv[]) {

  KC::Context ctx;
  KC::initContext(ctx);
  KC::ComputeDAG graph("fuse_gemm_relu", ctx);
  int m = 4096, n = 4096, k = 4096;
  // define the inputs first
  auto A = graph.placeholder({m, k}, "float32", KC::MemorySpace::global);
  auto B = graph.placeholder({k, n}, "float32", KC::MemorySpace::global);
  auto gemmOp = graph.gemm(A, B);
  graph.relu(gemmOp);
  graph.dumpAndVerify();

  llvm::errs() << "-------------------------------------------\n";

  graph.operatorImpl();
  graph.dumpAndVerify();
  llvm::errs() << "-------------------------------------------\n";


  KCM kcm(&graph);

  kcm.setTarget(KC::Target::CUDA);
  
  llvm::errs() << kcm.codeGen();

  kcm.autoTune();
  llvm::errs() << kcm.codeGen();

  kcm.autoSchedule();
  kcm.autoTune();
  graph.dumpAndVerify();
  llvm::errs() << "-------------------------------------------\n";
  llvm::errs() << kcm.codeGen();

  // kcm.graphTune();
  // kcm.autoSchedule();
  // kcm.autoTune();
  // llvm::errs() << kcm.codeGen();
}