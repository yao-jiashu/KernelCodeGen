#ifndef ComputeDAG_OPS
#define ComputeDAG_OPS

include "ComputeDAGDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def GEMMOp : ComputeDAG_Op<"gemm_all2all", [NoSideEffect]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs);
  let results = (outs AnyMemRef:$res);

  let assemblyFormat = "$lhs `x` $rhs  attr-dict `:` type($lhs) `x` type($rhs) `to` type($res)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
  }];

  // let hasFolder = 1;
}

def ReluOp : ComputeDAG_Op<"relu_elementwise", [NoSideEffect]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
  }];

  // let hasFolder = 1;
}

def DelimiterOp : ComputeDAG_Op<"delimiter", [NoSideEffect]> {
  let summary = "";
  let description = [{
  }];

  let assemblyFormat = "attr-dict";
}

#endif // ComputeDAG_OPS