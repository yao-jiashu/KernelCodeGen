#include <iostream>
#include "KernelCodegen.h"
namespace KC = KernelCodegen;
using KCM = KC::KernelCodegenMachine;


int main(int argc, char* argv[]) {

  KCM kcm;
  KC::ComputeDAG graph("fuse_gemm_relu", kcm.getContext());
  int m = 4096, n = 4096, k = 4096;
  // define the inputs first
  auto A = graph.placeholder({m, k}, "float32", KC::MemorySpace::global);
  auto B = graph.placeholder({k, n}, "float32", KC::MemorySpace::global);
  auto gemmOp = graph.gemm(A, B);
  graph.relu(gemmOp);
  graph.dumpAndVerify();

  std::cout << "-------------------------------------------\n";

  graph.operatorImpl();
  graph.dumpAndVerify();
  std::cout << "-------------------------------------------\n";

  kcm.setTarget(KC::Target::CUDA);
  
  std::cout << kcm.codegen(graph);

  kcm.autoTune(graph);
  std::cout << kcm.codegen(graph);

  kcm.autoSchedule(graph);
  kcm.autoTune(graph);
  std::cout << kcm.codegen(graph);

  kcm.graphTune(graph);
  kcm.autoSchedule(graph);
  kcm.autoTune(graph);
  std::cout << kcm.codegen(graph);
}