#ifndef ComputeDAG_DIALECT
#define ComputeDAG_DIALECT
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ComputeDAG dialect definition.
//===----------------------------------------------------------------------===//

def ComputeDAG_Dialect : Dialect {
    let name = "compute_dag";
    let summary = "Dialect for compute DAG optimization.";
    let description = [{
    }];
    let cppNamespace = "::mlir::compute_dag";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Base ComputeDAG operation definition.
//===----------------------------------------------------------------------===//

class ComputeDAG_Op<string mnemonic, list<Trait> traits = []> :
        Op<ComputeDAG_Dialect, mnemonic, traits>;

#endif // ComputeDAG_DIALECT