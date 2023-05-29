// #include "MLIREnhance.h"

// namespace {
// using namespace mlir;
// using namespace KernelCodegen;

// struct GEMMImplement : 
//   public PassWrapper<GEMMImplement, OperationPass<ModuleOp>> {
//    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GEMMImplement)
//    GEMMImplement() {}
//    void runOnOperation() override;
// };

// void GEMMImplement::runOnOperation() {
//    ModuleOp module = getOperation();
//    if (module->hasAttr("compute_dag.gemm_kernel")) {
//     return;
//    }
//    // The walker proceeds in pre-order to process
//    module.walk<WalkOrder::PreOrder>([&](compute_dag::GEMMOp gemmOp) {
//     auto outType = gemmOp.getResult().getType();
//     int64_t m {-1}, n {-1};
//     Type dtype;
//     if(outType.isa<MemRefType>()) {
//       auto outShape = outType.dyn_cast<MemRefType>();
//       m = outShape.getShape()[0];
//       n = outShape.getShape()[1];
//       dtype = outShape.getElementType();
//     }
//     else {
//       llvm::errs() << "Unsupported tensor type of the output of the GEMM.";
//       return;
//     }

//     auto typeA = gemmOp.getOperands()[0].getType();
//     auto typeB = gemmOp.getOperands()[1].getType();
//     int64_t k {-1};
//     if(typeA.isa<MemRefType>()) {
//       auto shapeA = typeA.dyn_cast<MemRefType>();
//       k = shapeA.getShape()[1];
//     }
//     else {
//       llvm::errs() << "Unsupported tensor type of the left operand.";
//       return;
//     }

//     OpBuilder builder(module.getContext());
//     module->setAttr(std::string("compute_dag.gemm_kernel"), 
//       builder.getStringAttr("True"));

//     builder.setInsertionPointToEnd(module.getBody());
//     OpBuilder::InsertionGuard guard(builder);

//     auto int32Type = builder.getI32Type();
//     std::vector<Type> typesArray 
//       {outType, typeA, typeB, int32Type, int32Type, int32Type};
//     ArrayRef<Type> paramTypes(typesArray);
//     auto functionType = builder.getFunctionType(TypeRange(paramTypes), llvm::None);

//     auto func = builder.create<func::FuncOp>(
//       builder.getUnknownLoc(), StringRef("gemm_kernel"), functionType);

//     func->getRegion(0).push_back(new Block);
//     Block &bodyBlock = func.front();
//     int nums = static_cast<int>(paramTypes.size());
//     for (int i = 0; i < nums; i++ ) {
//       bodyBlock.addArguments(paramTypes[i], builder.getUnknownLoc());
//     }
//     ValueRange operands = bodyBlock.getArguments();
//     builder.setInsertionPointToStart(&bodyBlock);  
  
//     // // build loops
//     SmallVector<int64_t, 3> lowerBounds(3, /*Value=*/0);
//     SmallVector<int64_t, 3> steps(3, /*Value=*/1);
//     SmallVector<int64_t, 3> upperBounds({m, n, k});
//     buildAffineLoopNest(
//       builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
//       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//         auto C = operands[0];
//         auto A = operands[1];
//         auto B = operands[2];
//         auto i = ivs[0];
//         auto j = ivs[1];
//         auto k = ivs[2];
//         auto ld_a = nestedBuilder.create<AffineLoadOp>(
//           nestedBuilder.getUnknownLoc(), A, ValueRange({i, k}));
//         auto ld_b = nestedBuilder.create<AffineLoadOp>(
//           nestedBuilder.getUnknownLoc(), B, ValueRange({k, j}));
//         auto ld_accum = nestedBuilder.create<AffineLoadOp>(
//           nestedBuilder.getUnknownLoc(), C, ValueRange({i, j}));
//         if (dtype.isa<FloatType>()) {
//           auto mul = nestedBuilder.create<arith::MulFOp>(
//             nestedBuilder.getUnknownLoc(), ld_a, ld_b);
//           auto add = nestedBuilder.create<arith::AddFOp>(
//             nestedBuilder.getUnknownLoc(), mul, ld_accum);
//           nestedBuilder.create<AffineStoreOp>(
//             nestedBuilder.getUnknownLoc(), add, C, ValueRange({i, j}));
//         }
//         else {
//           auto mul = nestedBuilder.create<arith::MulIOp>(
//             nestedBuilder.getUnknownLoc(), ld_a, ld_b);
//           auto add = nestedBuilder.create<arith::AddIOp>(
//             nestedBuilder.getUnknownLoc(), mul, ld_accum);
//           nestedBuilder.create<AffineStoreOp>(
//             nestedBuilder.getUnknownLoc(), add, C, ValueRange({i, j}));            
//         }
//       }
//     );
//     builder.create<func::ReturnOp>(builder.getUnknownLoc());
//    });
// }

// std::unique_ptr<OperationPass<ModuleOp>> GEMMImplementPass() {
//    return std::make_unique<GEMMImplement>();
// }
// }

// namespace KernelCodegen {

// void ComputeDAG::operatorFuse() {
//   mlir::PassManager pm(module.getContext());
//   pm.addPass(GEMMImplementPass());
//   if (failed(pm.run(module))) {
//     llvm::errs() << "Implement GEMM failed.";
//   }
// }
  
// }