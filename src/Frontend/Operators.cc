#include "Frontend/Operators.h"

namespace KernelCodeGen {

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype) {
  if(dtype == "float32") return builder.getF32Type();
  if(dtype == "float64") return builder.getF64Type();
  if(dtype == "float16") return builder.getF16Type();
  return nullptr;
}

std::string toStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"float16"};
  if(type.isa<mlir::Float32Type>()) return {"float32"};
  if(type.isa<mlir::Float64Type>()) return {"float64"};
  return nullptr;
}

mlir::func::FuncOp buildFuction(mlir::ModuleOp module, mlir::OpBuilder& builder, 
 const std::string& funcName, const std::vector<mlir::Type>& inputsTypes, 
 const std::vector<mlir::Type>& outputsTypes) {

  mlir::func::FuncOp result;
  bool break_ = false;
  
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp func) {
    if (break_) return;
    // auto otherName = func.getFunctionTypeAttrName();
    auto otherName = func.getSymName();
    if (otherName == funcName) {
      // Function already exists;
      result = func;
      break_ = true;
    }
  });
  if (break_) return result;


  builder.setInsertionPointToStart(module.getBody());
  
  llvm::ArrayRef<mlir::Type> inputsTypesArray(inputsTypes);
  llvm::ArrayRef<mlir::Type> outputsTypesArray(outputsTypes);
  auto functionType = builder.getFunctionType(mlir::TypeRange(inputsTypesArray), 
    mlir::TypeRange(outputsTypesArray));

  auto funcOp = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), llvm::StringRef(funcName), functionType);

  auto& region = funcOp->getRegion(0);
  if (!region.hasOneBlock()) {
    region.emplaceBlock();
  }
  auto& body =  funcOp.front(); //? region.front()  : ;

  int nums = static_cast<int>(inputsTypes.size());
  for (int i = 0; i < nums; i++ ) {
    body.addArguments(inputsTypes[i], builder.getUnknownLoc());
  }

  return funcOp;

}

mlir::Value PlaceHolder::build(ComputeDAG* graph, 
    const std::vector<int64_t>& shapes, 
    const std::string& dtype) {
  auto builder = graph->builder;
//   llvm::ArrayRef<int64_t> shapes_ (shapes);
  auto dtype_ = getDType(builder, dtype);
  auto tType = mlir::MemRefType::get(
    shapes, dtype_, {}, static_cast<int>(MemorySpace::global));
  auto allocOp = builder.create<mlir::memref::AllocOp>(
    builder.getUnknownLoc(), tType);
  return allocOp.getResult();
}

mlir::Value Matmul::build(ComputeDAG* graph, mlir::Value A, mlir::Value B, 
  MemorySpace ms, const std::string& dtype_) {
  
  auto builder = graph->builder;
  auto typeA = A.getType();
  auto typeB = B.getType();
  int64_t m {-1}, n {-1}, k1{-1}, k2{-1};
  mlir::Attribute memorySpace;
  mlir::Type elementTypeA;

  if(typeA.isa<mlir::MemRefType>()) {
    auto shapeA = typeA.dyn_cast<mlir::MemRefType>();
    m = shapeA.getShape()[0];
    k1 = shapeA.getShape()[1];
    elementTypeA = shapeA.getElementType();
    memorySpace = shapeA.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of left operand of Matmul is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementTypeA);

  if(typeB.isa<mlir::MemRefType>()) {
    auto shapeB = typeB.dyn_cast<mlir::MemRefType>();
    k2 = shapeB.getShape()[0];
    n = shapeB.getShape()[1];
  }
  else {
    llvm::errs() << "Type of right operand of Matmul is not Memref.\n";
    return nullptr;
  }

  if (k1 != k2) {
    llvm::errs() << 
      "Can't apply Matmul Operation due to imcompatible K-dim.\n";
    return nullptr;
  }

  auto funcName = std::string({"Matmul_m"}) + std::to_string(m) + 
                  "n" + std::to_string(n) +  "k" + std::to_string(k1);

  auto emType = getDType(builder, dtype);
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(
    std::vector<int64_t>{m, n}), emType, {}, static_cast<int>(ms));

  // Create C buffer as the result.
  auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
  auto C = allocOp.getResult();

  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {typeA, typeB, typeC}, {typeC});
  // auto& bodyBlock = funcOp.getBody().front(); // the same
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);

  mlir::ValueRange operands = bodyBlock.getArguments();
  
  // void buildAffineLoopNest(OpBuilder &builder, Location loc,
  //                         ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs,
  //                         ArrayRef<int64_t> steps,
  //                         function_ref<void(OpBuilder &, Location, ValueRange)>
  //                             bodyBuilderFn = nullptr);
  mlir::SmallVector<int64_t, 3> lowerBounds(2, /*Value=*/0);
  mlir::SmallVector<int64_t, 3> steps(2, /*Value=*/1);
  mlir::SmallVector<int64_t, 3> upperBounds({m, n});
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto i = ivs[0];
      auto j = ivs[1];
      // FloatAttr Builder::getFloatAttr(Type type, double value) {
      //   return FloatAttr::get(type, value);
      // }
      // initilize to 0
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(emType, 0));

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        auto k = iv;
        auto ld_a = builder.create<mlir::AffineLoadOp>(
                      builder.getUnknownLoc(), /*A*/operands[0], mlir::ValueRange({i, k}));
        auto ld_b = builder.create<mlir::AffineLoadOp>(
                      builder.getUnknownLoc(), /*B*/operands[1], mlir::ValueRange({k, j}));
        auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, iterArgs[0]);
        builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, k1, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({zero.getResult()}), kLoopBody);

      nestedBuilder.create<mlir::AffineStoreOp>(nestedBuilder.getUnknownLoc(), 
          Cij.getResult(0), /*C*/operands[2], mlir::ValueRange({i, j}));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), operands[2]);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B, C}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}


mlir::Value Relu::build(ComputeDAG* graph, mlir::Value input, MemorySpace ms, 
  const std::string& dtype_) {
  
  auto builder = graph->builder;
  auto type = input.getType();

  mlir::Attribute memorySpace;
  mlir::Type elementType;

  llvm::ArrayRef<int64_t> shape;

  if(type.isa<mlir::MemRefType>()) {
    auto type_ = type.dyn_cast<mlir::MemRefType>();
    shape = type_.getShape();
    elementType = type_.getElementType();
    memorySpace = type_.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of input of Relu is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

  auto funcName = std::string({"Relu_Elementwise"});

  for (auto dim : shape) {
    funcName += "_" + std::to_string(dim);
  }

  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {input.getType()});
  
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();

  auto emType = getDType(builder, dtype);

  mlir::Value output;
  if (ms != MemorySpace::inplace) {
    auto typeC = mlir::MemRefType::get(shape, emType, {}, static_cast<int>(ms));
    auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
    output = allocOp.getResult();
  } else {
    output = operands[0];
  }

  mlir::SmallVector<int64_t, 8> lowerBounds(shape.size(), /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(shape.size(), /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shape.begin(), shape.end());
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {

      // initilize to 0
      auto dtypeOutput = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeOutput, 0));
      auto ld_element = nestedBuilder.create<mlir::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], ivs);
      auto max = nestedBuilder.create<mlir::arith::MaxFOp>(nestedBuilder.getUnknownLoc(), zero, ld_element);
      nestedBuilder.create<mlir::AffineStoreOp>(nestedBuilder.getUnknownLoc(), max, output, ivs);
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}


mlir::Value BatchedMatmul::build(ComputeDAG* graph, mlir::Value A, Layout layoutA,
  mlir::Value B, Layout layoutB, const std::string& dtype_) {
  auto builder = graph->builder;
  auto typeA = A.getType();
  auto typeB = B.getType();
  int64_t m {-1}, n {-1}, k1{-1}, k2{-1};
  mlir::Attribute memorySpace;
  mlir::Type elementTypeA;

  int totalDims = -1;
  std::vector<int64_t> shapeC;

  if(typeA.isa<mlir::MemRefType>()) {
    auto mrTypeA = typeA.dyn_cast<mlir::MemRefType>();
    auto shapeA = mrTypeA.getShape();
    totalDims = shapeA.size();
    if (totalDims < 2) {
      llvm::errs() << "BatchedMatmul needs at least 2 dim but got " << totalDims << "\n";
      exit(EXIT_FAILURE);
    }
    shapeC.reserve(totalDims);
    for(auto dim : shapeA) {
      shapeC.push_back(dim);
    }
    if (layoutA == Layout::rowMajor) {
      m = shapeA[totalDims - 2];
      k1 = shapeA[totalDims - 1];
    } else {
      m = shapeA[totalDims - 1];
      k1 = shapeA[totalDims - 2];      
    }

    elementTypeA = mrTypeA.getElementType();
    memorySpace = mrTypeA.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of left operand of BatchedMatmul is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementTypeA);

  if(typeB.isa<mlir::MemRefType>()) {
    auto mrTypeB = typeB.dyn_cast<mlir::MemRefType>();
    auto shapeB = mrTypeB.getShape();
    if (totalDims != shapeB.size()) {
      llvm::errs() << "BatchedMatmul: A, B dim not matched.\n";
    }

    if (layoutB == Layout::colMajor) {
      k2 = shapeB[totalDims - 1];
      n = shapeB[totalDims - 2];
    } else {
      k2 = shapeB[totalDims - 2];
      n = shapeB[totalDims - 1];
    }
    shapeC[totalDims - 1] = n;
  }
  else {
    llvm::errs() << "Type of right operand of BatchedMatmul is not Memref.\n";
    return nullptr;
  }

  if (k1 != k2) {
    llvm::errs() << 
      "Can't apply BatchedMatmul Operation due to imcompatible K-dim.\n";
    return nullptr;
  }


  // Create C buffer as the result.
  auto C = graph->create<PlaceHolder>(shapeC, dtype);

  C.getDefiningOp()->moveAfter(B.getDefiningOp());

  int batch_dim_num = shapeC.size() - 2;

  auto funcName = std::string({"BatchMatmul"});

  for (int i = 0; i < batch_dim_num; i ++) {
    funcName += "_";
    funcName += std::to_string(shapeC[i]);
  }
  char transposeA = layoutA == Layout::rowMajor ? 'N' : 'T';
  char transposeB = layoutB== Layout::rowMajor ? 'N' : 'T';
  funcName += "_m" + std::to_string(shapeC[batch_dim_num]) + 
              "_n" + std::to_string(shapeC[batch_dim_num + 1]) +  
              "_k" + std::to_string(k1) + "_" + transposeA + transposeB;

  auto emType = getDType(builder, dtype);
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shapeC), 
    emType, {}, static_cast<int>(MemorySpace::global));

  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {typeA, typeB, typeC}, {typeC});
  // auto& bodyBlock = funcOp.getBody().front(); // the same
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);

  mlir::ValueRange operands = bodyBlock.getArguments();

  mlir::SmallVector<int64_t, 8> lowerBounds(totalDims, /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(totalDims, /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shapeC.begin(), shapeC.end());
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto i = ivs[totalDims - 2];
      auto j = ivs[totalDims - 1];
      // FloatAttr Builder::getFloatAttr(Type type, double value) {
      //   return FloatAttr::get(type, value);
      // }
      // initilize to 0
      auto dtypeC = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeC, 0));
      
      std::vector<mlir::Value> indexA;
      std::vector<mlir::Value> indexB;
      std::vector<mlir::Value> indexC;

      // fill with batch dimension.
      int counter = 0;
      for (auto iv : ivs) {
        if (counter++ < totalDims - 2) {
          indexA.push_back(iv); indexB.push_back(iv); indexC.push_back(iv);
        } else {
          break;
        }
      }
      indexC.push_back(i); indexC.push_back(j);

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        auto k = iv;
        if (layoutA == Layout::rowMajor) {
          indexA.push_back(i); indexA.push_back(k);
        } else {
          indexA.push_back(k); indexA.push_back(j);
        }

        if (layoutB == Layout::rowMajor) {
          indexB.push_back(k); indexB.push_back(j);
        } else {
          indexB.push_back(j); indexB.push_back(k);
        }

        auto ld_a = builder.create<mlir::AffineLoadOp>(
                      builder.getUnknownLoc(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(indexA)));
        auto ld_b = builder.create<mlir::AffineLoadOp>(
                      builder.getUnknownLoc(), operands[1], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(indexB)));
        auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, iterArgs[0]);
        builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, k1, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({zero.getResult()}), kLoopBody);

      nestedBuilder.create<mlir::AffineStoreOp>(nestedBuilder.getUnknownLoc(), 
          Cij.getResult(0), operands[2], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(indexC)));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), operands[2]);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B, C}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

mlir::Value Transpose::build(ComputeDAG* graph, mlir::Value input) {
}

mlir::Value Softmax::build(ComputeDAG* graph, mlir::Value input, int axis, MemorySpace ms,
    const std::string& dtype_) {
  auto builder = graph->builder;
  auto type = input.getType();

  mlir::Attribute memorySpace;
  mlir::Type elementType;

  llvm::ArrayRef<int64_t> shape;

  if(type.isa<mlir::MemRefType>()) {
    auto type_ = type.dyn_cast<mlir::MemRefType>();
    shape = type_.getShape();
    elementType = type_.getElementType();
    memorySpace = type_.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of input of Softmax is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

  int totalDims = shape.size();

  if (axis < 0 || axis >= totalDims) {
    llvm::errs() << "Illegal reduction axis in Softmax.\n";
  }
  auto reduceStartAxis = axis == -1 ? totalDims - 1 : axis;

  auto funcName = std::string({"Softmax"});

  for (int i = 0; i < totalDims; i ++) {
    funcName += "_";
    funcName += std::to_string(shape[i]);
  }
  funcName += "_axis" + std::to_string(reduceStartAxis);


  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {input.getType()});
  // auto& bodyBlock = funcOp.getBody().front(); // the same
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);

  mlir::ValueRange operands = bodyBlock.getArguments();

  mlir::SmallVector<int64_t, 8> lowerBounds(reduceStartAxis, /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(reduceStartAxis, /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shape.begin(), shape.begin() + reduceStartAxis);
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {


      
      // At present, only support reduction on the last dim.
      if (totalDims - reduceStartAxis != 1) {
        llvm::errs() << "At present, only support reduction on the last dim.\n";
      }

      // initilize to 0
      auto dtypeOutput = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeOutput, 0));
      
      std::vector<mlir::Value> index;
      index.reserve(totalDims);

      // fill with batch dimension.
      int counter = 0;
      for (auto iv : ivs) {
        if (counter++ < totalDims - 1) {
          index.push_back(iv);
        } else {
          break;
        }
      }
      
      // Reduction.
      auto kLoopBody = [&](mlir::OpBuilder &kBuilder, mlir::Location kLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard kGuard(kBuilder);
        auto k = iv;
        index.push_back(k);
        auto ld = kBuilder.create<mlir::AffineLoadOp>(
                      kBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(index)));
        auto exp = kBuilder.create<mlir::math::ExpOp>(kBuilder.getUnknownLoc(), ld.getResult());
        auto add = kBuilder.create<mlir::arith::AddFOp>(kBuilder.getUnknownLoc(), exp.getResult(), iterArgs[0]);
        builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto sum = nestedBuilder.create<mlir::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, shape.back(), 1, /*iterArgs=lvm::None*/ mlir::ValueRange({zero.getResult()}), kLoopBody);

      // Elementwise.
      auto ewLoopBody = [&](mlir::OpBuilder &ewBuilder, mlir::Location ewLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard kGuard(ewBuilder);
        auto ew = iv;
        index.back() = ew;
        auto ld = ewBuilder.create<mlir::AffineLoadOp>(
                      ewBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(index)));
        auto exp = ewBuilder.create<mlir::math::ExpOp>(ewBuilder.getUnknownLoc(), ld.getResult());
        auto div = ewBuilder.create<mlir::arith::DivFOp>(ewBuilder.getUnknownLoc(), exp.getResult(), sum.getResult(0));
        
        ewBuilder.create<mlir::AffineStoreOp>(ewBuilder.getUnknownLoc(), 
            div.getResult(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(index)));
        ewBuilder.create<mlir::AffineYieldOp>(ewBuilder.getUnknownLoc());
      };
      nestedBuilder.create<mlir::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, shape.back(), 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), ewLoopBody);
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), operands[0]);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

}