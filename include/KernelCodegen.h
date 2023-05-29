#include "MLIREnhance.h"

#include <string>
#include <initializer_list>

namespace KernelCodegen {

class ComputeDAG;

enum class Target {
  CUDA = 0,
  ROCm = 1,
};

class KernelCodegenMachine {
public:
  
  KernelCodegenMachine() {initContext(context);}

  void setTarget(Target target_) { target = target_;}
  Context& getContext() { return context; }
  std::string codegen(ComputeDAG & dag) {return {};}
  void autoTune(ComputeDAG & dag) {}
  void autoSchedule(ComputeDAG & dag) {}
  void graphTune(ComputeDAG & dag) {}

private:
  Context context;
  Target target {Target::CUDA};
};

}