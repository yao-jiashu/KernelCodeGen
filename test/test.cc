#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
using namespace KernelCodeGen;


int main(int argc, char* argv[]) {

  KernelCodeGenerator codegen("CUDA");

  auto graph = codegen.createGraph("fuse_matmul_relu");

  int m = 4096, n = 2048, k = 1024;

  auto A = graph.create<PlaceHolder>(std::vector<int64_t>{m, k}, std::string{"float32"});
  auto B = graph.create<PlaceHolder>(std::vector<int64_t>{k, n}, std::string{"float32"});
  auto C = graph.create<Matmul>(A, B);
  // auto D = graph.create<Relu>(C);

  graph.dump();
  auto module = codegen.optimize(graph);

  codegen.dump(module);

}