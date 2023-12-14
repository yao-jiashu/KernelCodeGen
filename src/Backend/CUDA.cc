#include "IR/IR.h"
#include "Optimizer/Analyzer.h"
#include "Backend/CUDA.h"
#include "enum.h"
#include "log.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

inline std::string toCStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"half_t"};
  if(type.isa<mlir::Float32Type>()) return {"float"};
  if(type.isa<mlir::Float64Type>()) return {"double"};
  return nullptr;
}

int64_t kernelCounter = 0;

int64_t varCounter = 0;

struct CompareValue {
  int operator()(const mlir::Value& x, const mlir::Value& y) const {
    // auto x_hashCode = reinterpret_cast<size_t>(&x);
    // auto y_hashCode = reinterpret_cast<size_t>(&y);
    // if (x_hashCode >= y_hashCode) return 0;
    if (x == y) return 0;
    auto x_hashCode = x.getAsOpaquePointer();
    auto y_hashCode = y.getAsOpaquePointer();
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

struct CompareKernel {
  int operator()(const mlir::AffineParallelOp& x, const mlir::AffineParallelOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

std::stringstream source;

std::map<mlir::Value, std::string, CompareValue> valueNameMap;

std::map<mlir::AffineParallelOp, std::string, CompareKernel> kernelNameMap;

std::string getKernelName() {
  return std::string("kernel") + std::to_string(kernelCounter++);
}

std::string getArgName() {
  return std::string("arg") + std::to_string(varCounter++);
}

bool setValueName(mlir::Value val, std::string name) {
 if (valueNameMap.count(val) != 0) {
  llvm::errs() << "value already exists\n";
  return false;
 }
 valueNameMap[val] = name;
}

std::string getValueName(mlir::Value val) {
  if (valueNameMap.count(val) == 0) {
    llvm::errs() << "value not exists\n";
    return "false";
  }
  return valueNameMap[val];
}

namespace KernelCodeGen {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the ModuleOp traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class CUDAGenerator {
public:
  CUDAGenerator() {
    kernelCounter = 0;
    varCounter = 0;
    valueNameMap.clear();
  }
  void codegen(mlir::ModuleOp node);

private:
// mlir::arith::ConstantIndexOp, mlir::arith::MulFOp, mlir::arith::AddFOp, mlir::memref::AllocOp,
// mlir::AffineApplyOp, mlir::AffineIfOp, mlir::AffineForOp, mlir::AffineLoadOp, mlir::AffineStoreOp,
// mlir::AffineVectorLoadOp, mlir::AffineVectorStoreOp, mlir::gpu::BarrierOp
  void codegen(mlir::arith::ConstantIndexOp);
  void codegen(mlir::arith::ConstantFloatOp);
  void codegen(mlir::arith::MulFOp);
  void codegen(mlir::arith::AddFOp);
  void codegen(mlir::memref::AllocOp);
  void codegen(mlir::AffineApplyOp);
  void codegen(mlir::AffineIfOp);
  void codegen(mlir::AffineForOp);
  void codegen(mlir::AffineLoadOp);
  void codegen(mlir::AffineStoreOp);
  void codegen(mlir::AffineVectorLoadOp);
  void codegen(mlir::AffineVectorStoreOp);
  void codegen(mlir::gpu::BarrierOp);
  void codegen(mlir::AffineParallelOp);
  void codegen(mlir::AffineMap, const llvm::SmallVector<mlir::Value>&);
  std::string codegen(mlir::AffineExpr, const llvm::SmallVector<mlir::Value>&);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      source << "  ";
  }
  int curIndent = -1;
};

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  // indent();

void varDeclear(mlir::Value var) {
  auto memrefType = var.getType().dyn_cast<mlir::MemRefType>();
  auto elementType = memrefType.getElementType();
  auto memorySpace = memrefType.getMemorySpaceAsInt();
  if (memorySpace == static_cast<int>(MemorySpace::shared)) {
    source << "__shared__ ";
  }
  source << toCStr(elementType);

  auto getContinusStar = [&](int num) {
    std::string str = "";
    for (int i = 0; i < num; i++) {
      str += "*";
    }
    return str;
  };

  std::string varName = getValueName(var);

  auto dims = memrefType.getShape();
  if (memorySpace == static_cast<int>(MemorySpace::global)) {
    // llvm::errs() << getContinusStar(dims.size()) << " " << varName;
    source << getContinusStar(1) << " " << varName;
  } else {
    source << " " << varName;
    for (int i = 0; i < dims.size(); i++) {
     source << "[" << dims[i] << "]";
    }
  }
}

/// @brief collect value and its name to valueNameMap
/// @param node 
/// @return return the operands not defined in the `node`'s scope.
std::vector<mlir::Value> collectVars(mlir::AffineParallelOp node) {

  std::vector<std::string> int3str {"x", "y", "z"};
  int id = 0;
  std::map<mlir::Value, int, CompareValue> outsidesVars;

  //parallel index
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineParallelOp parallelOp) {
    auto operands = parallelOp.getIVs();
    std::string prefix {""};
    if (parallelOp == node) {
      prefix += "blockIdx.";
    } else {
      prefix += "threadIdx.";
    }
    for (int i = 0; i < operands.size(); i+= 1) {
      setValueName(operands[i], prefix + int3str[operands.size() - i - 1]);
    }
  });

  // induction var of loops
  int iterVarCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
    auto iterVar = forOp.getInductionVar();
    setValueName(iterVar, "iter" + std::to_string(iterVarCounter++));
  });

  int applyCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineApplyOp applyOp) {
    auto results = applyOp->getResults();
    for (int i = 0; i < results.size(); i += 1) {
      setValueName(results[i], "expr" + std::to_string(applyCounter++));
    }
  });

  int allocCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::memref::AllocOp allocOp) {
    auto result = allocOp.getResult();
    setValueName(result, "array" + std::to_string(allocCounter++));
  });

  int vectorLoadCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorLoadOp vecLoadOp) {
    auto mem = vecLoadOp.getMemref();
    if (valueNameMap.count(mem) == 0) {
      if (outsidesVars.count(mem) == 0) {
        outsidesVars[mem] = id ++;
        setValueName(mem, getArgName());
      }
    }
    
    auto results = vecLoadOp->getResults();
    for (int i = 0; i < results.size(); i += 1) {
      setValueName(results[i], "vec" + std::to_string(vectorLoadCounter++));
    }
  });

  int regLoadCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineLoadOp loadOp) {
    auto mem = loadOp.getMemref();
    if (valueNameMap.count(mem) == 0) {
      if (outsidesVars.count(mem) == 0) {
        outsidesVars[mem] = id ++;
        setValueName(mem, getArgName());
      }
    }
    auto results = loadOp->getResults();
    for (int i = 0; i < results.size(); i += 1) {
      setValueName(results[i], "R" + std::to_string(regLoadCounter++));
    }
  });

  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineStoreOp storeOp) {
    auto mem = storeOp.getMemref();
    if (valueNameMap.count(mem) == 0) {
      if (outsidesVars.count(mem) == 0) {
        outsidesVars[mem] = id ++;
        setValueName(mem, getArgName());
      }
    }
  });

  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorStoreOp storeOp) {
    auto mem = storeOp.getMemref();
    if (valueNameMap.count(mem) == 0) {
      if (outsidesVars.count(mem) == 0) {
        outsidesVars[mem] = id ++;
        setValueName(mem, getArgName());
      }
    }
  });

  int constCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantIndexOp constOp) {
    auto result = constOp.getResult();
    setValueName(result, "const" + std::to_string(constCounter++) + "th");
  });
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantFloatOp constOp) {
    auto result = constOp.getResult();
    setValueName(result, "const" + std::to_string(constCounter++) + "th");
  });

  int tempCounter = 0;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::MulFOp mulOp) {
    auto result = mulOp.getResult();
    setValueName(result, "temp" + std::to_string(tempCounter++));
  });

  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::AddFOp addOp) {
    auto result = addOp.getResult();
    setValueName(result, "temp" + std::to_string(tempCounter++));
  });

  std::vector<mlir::Value> result;
  for (auto var : outsidesVars) {
    result.push_back(var.first);
  }
  auto cmp = [&](mlir::Value a, mlir::Value b) {
    return outsidesVars[a] < outsidesVars[b];
  };
  std::sort(result.begin(), result.end(), cmp);
  return result;
}

void CUDAGenerator::codegen(mlir::memref::AllocOp allocOp) {
  indent();
  varDeclear(allocOp.getResult());
  source << ";\n";
}

void CUDAGenerator::codegen(mlir::gpu::BarrierOp) {
  indent();
  source << "__syncthreads();\n";

}

std::string CUDAGenerator::codegen(mlir::AffineExpr expr, const llvm::SmallVector<mlir::Value>& operands) {
  if (auto dimExpr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return getValueName(operands[dimExpr.getPosition()]);
  }
  if (auto constExpr = expr.dyn_cast<mlir::AffineConstantExpr>()) {
    auto val = constExpr.getValue();
    if (val >= 10240) {
      return std::to_string(val) + "";
    }
    return std::to_string(val);
  }
  auto binaryExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>();
  assert(binaryExpr);
  auto lhs = codegen(binaryExpr.getLHS(), operands);
  auto rhs = codegen(binaryExpr.getRHS(), operands);
  switch (binaryExpr.getKind()) {
    case mlir::AffineExprKind::Add: return "(" + lhs + " + " + rhs + ")";
    // case mlir::AffineExprKind::CeilDiv: return (lhs + rhs - 1) / rhs;
    case mlir::AffineExprKind::CeilDiv: return "((" + lhs + " + " + rhs + " - 1)" + " / " + rhs + ")";
    case mlir::AffineExprKind::FloorDiv: return "(" + lhs + " / " + rhs + ")";
    case mlir::AffineExprKind::Mod: return "(" + lhs + " % " + rhs + ")";
    case mlir::AffineExprKind::Mul: return "(" + lhs + " * " + rhs + ")";
    default: assert(false);
  }
}
void CUDAGenerator::codegen(mlir::AffineMap map, const llvm::SmallVector<mlir::Value>& operands) {}

void CUDAGenerator::codegen(mlir::AffineApplyOp applyOp) {
  auto map = applyOp.getAffineMap();
  auto operands = applyOp.getMapOperands();
  auto exprs = map.getResults();
  assert(exprs.size() == 1);
  auto result = applyOp.getResult();

  indent();
  source << "int " << getValueName(applyOp.getResult()) << " = " 
               << this->codegen(exprs[0], llvm::SmallVector<mlir::Value>(operands))
               << ";\n";
}

void CUDAGenerator::codegen(mlir::arith::ConstantIndexOp constOp) {
  indent();
  source << "constexpr int " << getValueName(constOp.getResult()) 
               << " = " << constOp.value() << ";\n"; 
}

void CUDAGenerator::codegen(mlir::arith::ConstantFloatOp floatOp) {
  auto eleT = floatOp.getType();
  indent();
  source << "constexpr " << toCStr(eleT) << " "
               << getValueName(floatOp.getResult()) 
               << " = " << static_cast<float>(floatOp.value().convertToFloat()) << ";\n"; 
}

void CUDAGenerator::codegen(mlir::arith::MulFOp mulOp) {
  indent();
  source << "auto " << getValueName(mulOp.getResult()) << " = "
               << getValueName(mulOp.getLhs()) << " * "
               << getValueName(mulOp.getRhs()) << ";\n";
}

void CUDAGenerator::codegen(mlir::arith::AddFOp addOp) {
  indent();
  source << "auto " << getValueName(addOp.getResult()) << " = "
               << getValueName(addOp.getLhs()) << " + "
               << getValueName(addOp.getRhs()) << ";\n";
}

void CUDAGenerator::codegen(mlir::AffineIfOp ifOp) {
  auto iset = ifOp.getIntegerSet();
  int numConstraints = iset.getNumConstraints();
  auto operands = ifOp.getOperands();
  indent();
  source << "if (";
  for (int i = 0; i < numConstraints; i += 1) {
    auto expr = iset.getConstraint(i);
    auto isEq = iset.isEq(i);
    std::string relation = isEq ? "==" : ">=";
    source << this->codegen(expr, operands) << " " << relation << " 0 && ";
  }
  source << " true) {\n";
  {
    INDENT();
    auto& ops = ifOp.getBody()->getOperations();
    for (auto& op : ops) {
      if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(&op)) {
        this->codegen(forOp);
      } else if (auto vecLoad = mlir::dyn_cast<mlir::AffineVectorLoadOp>(&op)) {
        this->codegen(vecLoad);
      } else if (auto vecStore = mlir::dyn_cast<mlir::AffineVectorStoreOp>(&op)) {
        this->codegen(vecStore);
      } else if (auto barrierOp = mlir::dyn_cast<mlir::gpu::BarrierOp>(&op)) {
        this->codegen(barrierOp);
      } else {
        auto yieldOp = mlir::dyn_cast<mlir::AffineYieldOp>(&op);
        assert(yieldOp);
      }
    }
  }
  indent();
  source << "}\n";
}

void CUDAGenerator::codegen(mlir::AffineLoadOp loadOp) {
  indent();
  source << "auto " << getValueName(loadOp.getResult()) << " = " 
               << getValueName(loadOp.getMemref());
  auto map = loadOp.getAffineMap();
  auto operands = loadOp.getMapOperands();
  auto exprs = map.getResults();

  auto type = loadOp.getMemref().getType().dyn_cast<mlir::MemRefType>();
  auto memorySpace = type.getMemorySpaceAsInt();
  if (memorySpace == static_cast<int>(MemorySpace::global)) {
    auto shape = type.getShape();
    std::vector<int> strides;
    auto size = shape.size();
    for (int i = 0; i < shape.size(); i++) {
      if (i == 0) {
        strides.push_back(1);
      } else {
        strides.push_back(strides[i - 1] * shape[size - i]);
      }
    }
    source << "[";
    int index = exprs.size() - 1;
    for (auto expr : exprs) {
      std::string suffix = "";
      auto stride = strides[index--];
      if (stride >= 10240) suffix += "";
      source << this->codegen(expr, operands) << " * " << stride << suffix << " + ";
    }
    source << "0]";
  } else {
    for (auto expr : exprs) {
      source << "[" << this->codegen(expr, operands) << "]";
    }
  }
  source << ";\n";
}

void CUDAGenerator::codegen(mlir::AffineStoreOp storeOp) {
  indent();
  source << getValueName(storeOp.getMemref());
  auto map = storeOp.getAffineMap();
  auto operands = storeOp.getMapOperands();
  auto exprs = map.getResults();

  auto type = storeOp.getMemref().getType().dyn_cast<mlir::MemRefType>();
  auto memorySpace = type.getMemorySpaceAsInt();
  if (memorySpace == static_cast<int>(MemorySpace::global)) {
    auto shape = type.getShape();
    std::vector<int> strides;
    auto size = shape.size();
    for (int i = 0; i < shape.size(); i++) {
      if (i == 0) {
        strides.push_back(1);
      } else {
        strides.push_back(strides[i - 1] * shape[size -i]);
      }
    }
    source << "[";
    int index = exprs.size() - 1;
    for (auto expr : exprs) {
      std::string suffix = "";
      auto stride = strides[index--];
      if (stride >= 10240) suffix += "";
      source << this->codegen(expr, operands) << " * " << stride << suffix << " + ";
    }
    source << "0]";
  } else {
    for (auto expr : exprs) {
      source << "[" << this->codegen(expr, operands) << "]";
    }
  }

  source << " = " << getValueName(storeOp.getValue());
  source << ";\n";
}

std::string getVectorFetchType(mlir::VectorType vt) {
  auto eleT = vt.getElementType();
  int width = -1;
  if (eleT.isF16()) {
    width = 16;
  } else if (eleT.isF32()) {
    width = 32;
  } else if (eleT.isF64()) {
    width = 64;
  }
  if (width == -1) {
    llvm::errs() << "Vector type error\n";
  }
  auto vecLen = vt.getShape()[0];
  auto totalBits = vecLen * width;
  auto totalFloat = totalBits / 32;

  return "float" + std::to_string(totalFloat);
}

void CUDAGenerator::codegen(mlir::AffineVectorLoadOp loadOp) {
  indent();
  source << "auto " << getValueName(loadOp.getResult()) << " = ";

  auto codegenMemref = [&](mlir::AffineVectorLoadOp loadOp) -> std::string {
    auto result = getValueName(loadOp.getMemref());
    auto map = loadOp.getAffineMap();
    auto operands = loadOp.getMapOperands();
    auto exprs = map.getResults();

    auto type = loadOp.getMemref().getType().dyn_cast<mlir::MemRefType>();
    auto memorySpace = type.getMemorySpaceAsInt();
    if (memorySpace == static_cast<int>(MemorySpace::global)) {
      auto shape = type.getShape();
      auto size = shape.size();
      std::vector<int> strides;
      for (int i = 0; i < shape.size(); i++) {
        if (i == 0) {
          strides.push_back(1);
        } else {
          strides.push_back(strides[i - 1] * shape[size - i]);
        }
      }
      result += "[";
      int index = exprs.size() - 1;
      for (auto expr : exprs) {
        std::string suffix = "";
        auto stride = strides[index--];
        if (stride >= 10240) suffix += "";
        result += this->codegen(expr, operands) + " * " + std::to_string(stride)  + suffix + " + ";
      }
      result += "0]";
    } else {
      for (auto expr : exprs) {
        result += "[" + this->codegen(expr, operands) + "]";
      }
    }
 
    return result;
  };

  auto vecType = loadOp.getVectorType();
  auto vstr = getVectorFetchType(vecType);
  source << "(reinterpret_cast<" << vstr << "*>(&(" << codegenMemref(loadOp) << "))[0]);\n";

}

void CUDAGenerator::codegen(mlir::AffineVectorStoreOp storeOp) {

  auto codegenMemref = [&](mlir::AffineVectorStoreOp storeOp) -> std::string {
    auto result = getValueName(storeOp.getMemref());
    auto map = storeOp.getAffineMap();
    auto operands = storeOp.getMapOperands();
    auto exprs = map.getResults();

    auto type = storeOp.getMemref().getType().dyn_cast<mlir::MemRefType>();
    auto memorySpace = type.getMemorySpaceAsInt();
    if (memorySpace == static_cast<int>(MemorySpace::global)) {
      auto shape = type.getShape();
      std::vector<int> strides;
      auto size = shape.size();
      for (int i = 0; i < shape.size(); i++) {
        if (i == 0) {
          strides.push_back(1);
        } else {
          strides.push_back(strides[i - 1] * shape[size - i]);
        }
      }
      result += "[";
      int index = exprs.size() - 1;
      for (auto expr : exprs) {
        std::string suffix = "";
        auto stride = strides[index--];
        if (stride >= 10240) suffix += "";
        result += this->codegen(expr, operands) + " * " + std::to_string(stride)  + suffix + " + ";
      }
      result += "0]";
    } else {
      for (auto expr : exprs) {
        result += "[" + this->codegen(expr, operands) + "]";
      }
    }
    return result;
  };

  indent();
  auto vecType = storeOp.getVectorType();
  auto vstr = getVectorFetchType(vecType);
  source << "(reinterpret_cast<" << vstr << "*>(&(" << codegenMemref(storeOp) << "))[0])";
  source << " = " << getValueName(storeOp.getValue()) << ";\n";
}

void CUDAGenerator::codegen(mlir::AffineForOp forOp) {
  
  auto lb = forOp.getConstantLowerBound();
  auto ub = forOp.getConstantUpperBound();
  auto step = forOp.getStep();
  auto iter = getValueName(forOp.getInductionVar());

  if (forOp->hasAttr(std::string("affine.loop"))) {
    auto attr = forOp->getAttr(std::string("affine.loop"));
    auto strAttr = attr.dyn_cast<mlir::StringAttr>();
    auto builder = mlir::OpBuilder(forOp->getContext());
    if (strAttr.compare(builder.getStringAttr("unroll")) == 0) {
      indent();
      source << "#pragma unroll\n";
    }
  }

  indent();
  source << "for (int " << iter << " = " << lb << "; "
               << iter << " < " << ub << "; "
               << iter << " += " << step << ") {\n";
  {
    INDENT();
    auto& ops = forOp.getBody()->getOperations();
    for (auto& op : ops) {
      if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(&op)) {
        this->codegen(forOp);
      } else if (auto ifOp = mlir::dyn_cast<mlir::AffineIfOp>(&op)) {
        this->codegen(ifOp);
      } else if (auto loadOp = mlir::dyn_cast<mlir::AffineLoadOp>(&op)) {
        this->codegen(loadOp);
      } else if (auto storeOp = mlir::dyn_cast<mlir::AffineStoreOp>(&op)) {
        this->codegen(storeOp);
      } else if (auto vecLoad = mlir::dyn_cast<mlir::AffineVectorLoadOp>(&op)) {
        this->codegen(vecLoad);
      } else if (auto vecStore = mlir::dyn_cast<mlir::AffineVectorStoreOp>(&op)) {
        this->codegen(vecStore);
      } else if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(&op)) {
        this->codegen(constOp);
      } else if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(&op)) {
        this->codegen(constOp);
      } else if (auto mulOp = mlir::dyn_cast<mlir::arith::MulFOp>(&op)) {
        this->codegen(mulOp);
      } else if (auto addOp = mlir::dyn_cast<mlir::arith::AddFOp>(&op)) {
        this->codegen(addOp);
      } else if (auto barrierOp = mlir::dyn_cast<mlir::gpu::BarrierOp>(&op)) {
        this->codegen(barrierOp);
      } else {
        auto yieldOp = mlir::dyn_cast<mlir::AffineYieldOp>(&op);
        assert(yieldOp);
      }
    }
  }
  indent();
  source << "}\n";
}

/// Print a function, first the prototype and then the body.
void CUDAGenerator::codegen(mlir::AffineParallelOp node) {

  auto&& outsideVars = collectVars(node);
  assert(outsideVars.size() != 0);
  
  int64_t totalNumber;
  std::vector<int64_t> gridDims = Analyzer::getParallelNumber(node, totalNumber);
  std::vector<int64_t> blockDims;
  node.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineParallelOp parallelOp) {
    blockDims = Analyzer::getParallelNumber(parallelOp, totalNumber);
  });

  // Annotation
  indent();
  source << "// grid dims:(";
  for (auto dim : gridDims) source << dim << ", ";
  source << ")" << ", block dims:(";
  for (auto dim : blockDims) source << dim << ", ";
  source << ")\n";

  // kernel prototype
  indent();
  source << "__global__ void " << getKernelName() << "(";
  varDeclear(outsideVars[0]);
  for (int i = 1; i < outsideVars.size(); i += 1) {
    source << ", ";
    varDeclear(outsideVars[i]);
  }
  source << ") {\n";

  {
    INDENT();
    // kernel body.
    auto& ops = node.getBody()->getOperations();
    for (auto& op : ops) {
      if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(&op)) {
        this->codegen(allocOp);
      } else if (auto applyOp = mlir::dyn_cast<mlir::AffineApplyOp>(&op)) {
        this->codegen(applyOp); 
      } else if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(&op)) {
        auto& innerOps = parallelOp.getBody()->getOperations();
        for (auto& innerOp : innerOps) {
          if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(&innerOp)) {
            this->codegen(constOp);
          } else if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(&innerOp)) {
            this->codegen(allocOp);
          } else if (auto applyOp = mlir::dyn_cast<mlir::AffineApplyOp>(&innerOp)) {
            this->codegen(applyOp); 
          } else if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(&innerOp)) {
            this->codegen(forOp);
          } else if (auto ifOp = mlir::dyn_cast<mlir::AffineIfOp>(&innerOp)) {
            this->codegen(ifOp);
          } else if (auto loadOp = mlir::dyn_cast<mlir::AffineLoadOp>(&innerOp)) {
            this->codegen(loadOp);
          } else if (auto storeOp = mlir::dyn_cast<mlir::AffineStoreOp>(&innerOp)) {
            this->codegen(storeOp);
          } else if (auto vecLoad = mlir::dyn_cast<mlir::AffineVectorLoadOp>(&innerOp)) {
            this->codegen(vecLoad);
          } else if (auto vecStore = mlir::dyn_cast<mlir::AffineVectorStoreOp>(&innerOp)) {
            this->codegen(vecStore);
          } else if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(&innerOp)) {
            this->codegen(constOp);
          } else if (auto mulOp = mlir::dyn_cast<mlir::arith::MulFOp>(&innerOp)) {
            this->codegen(mulOp);
          } else if (auto addOp = mlir::dyn_cast<mlir::arith::AddFOp>(&innerOp)) {
            this->codegen(addOp);
          } else if (auto barrierOp = mlir::dyn_cast<mlir::gpu::BarrierOp>(&innerOp)) {
            this->codegen(barrierOp);
          } else {
            auto yieldOp = mlir::dyn_cast<mlir::AffineYieldOp>(&innerOp);
            assert(yieldOp);
          }
        }
      } else {
        auto yieldOp = mlir::dyn_cast<mlir::AffineYieldOp>(&op);
        assert(yieldOp);
      }
    }
  }
  indent();
  source << "}\n";
}

/// Print a module, actually loop over the functions and print them in sequence.
void CUDAGenerator::codegen(mlir::ModuleOp node) {
  INDENT();
  auto& kernels = node.getBody()->getOperations();
  for (auto& kernel : kernels) {
    if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(kernel)) {
      this->codegen(parallelOp);
    }
  }
}


// Public API
std::string CUDAGen(mlir::ModuleOp &module) {
  source.clear();
  source.str("");
  source << "#include \"cuda_runtime.h\"\n";
  source << "namespace " + module.getName().value().str() + " {\n";
  CUDAGenerator().codegen(module); 
  source << "}\n";
  std::string sourceStr = source.str();
  if (KCGLog::level == Log::Debug) {
    llvm::errs() << sourceStr;
  }
  return std::move(sourceStr);
}

}