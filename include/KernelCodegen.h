#pragma once
#include "ComputeDAG.h"
#include "GraphTune.h"
#include "AutoConfig.h"
#include "Scheduler.h"
#include "Element_Collecter.h"
#include "Element_Parser.h"
#include "CodeGen.h"

#include <string>
#include <sstream>
#include <initializer_list>

namespace KernelCodegen {

enum class Target {
  CUDA = 0,
  ROCm = 1,
};

class KernelCodegenMachine {
public:
  
  KernelCodegenMachine() = default;
  KernelCodegenMachine(ComputeDAG* graph_) : graph(graph_) {}
  void setTarget(Target target_) { target = target_;}
  void setTask(ComputeDAG* graph_) { 
    graph = graph_;
  }
  std::string codeGen();
  // std::string codeGen() {return {};}
  void autoTune() {}
  void autoSchedule(GEMMConfig&& config);
  void graphTune() {}

private:
  ComputeDAG* graph {nullptr};
  Scheduler scheduler;
  Target target {Target::CUDA};
};

}