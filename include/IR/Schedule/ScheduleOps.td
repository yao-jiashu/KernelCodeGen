#ifndef Schedule_OPS
#define Schedule_OPS

include "ScheduleDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

def Schedule_VectorizeOp : Schedule_Op<"vector_cast", [NoSideEffect]> {
  let summary = "memref elemental type vectorizing cast operation";
  let description = [{
    The `vector_cast` operation converts a memref with a non-vector
    element type to a memref of a vector element type while not changing
    the former memref's base element type, rank, and its dimension sizes along
    all but the last dimension. The last dimension size of the source dimension
    is divided (floor division) by the vector size to obtain the corresponding
    dimension size for the target memref type. The source memref type's last
    dimension is expected to be at least vector sized. Examples:

    ```mlir
    %MV = memref.vector_cast %M : memref<8x16xf32> to memref<8x2xvector<8xf32>>
    %AV = memref.vector_cast %A : memref<?x?xf32> to memref<?x?xvector<8xf32>>
    ```
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$dest);

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let hasVerifier = 1;
  //let verifier = [{ return impl::verifyCastOp(*this, areCastCompatible); }];

  let extraClassDeclaration = [{
    /// Returns true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a memref.vector_cast is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SliceVectorOp
//===----------------------------------------------------------------------===//

def Schedule_SliceVectorOp : Schedule_Op<"vector_slice", [NoSideEffect]> {
  let summary = "memref elemental type vectorizing slice operation";
  let description = [{
    The `vector_slice` operation converts a memref with a vector
    element type to a non-memref of a element type while not changing
    the former memref's base element type, rank, and its dimension sizes along
    all but the last dimension. The last dimension size of the source dimension
    is divided (floor division) by the vector size to obtain the corresponding
    dimension size for the target memref type. The source memref type's last
    dimension is expected to be at least vector sized. Examples:

    ```mlir
    %MV = memref.vector_cast %M : memref<8x2xvector<8xf32>> to memref<8x16xf32>
    %AV = memref.vector_cast %A : memref<?x?xvector<8xf32>> to memref<?x?xf32>
    ```
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$dest);

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let hasVerifier = 1;
  //let verifier = [{ return impl::verifyCastOp(*this, areCastCompatible); }];

  let extraClassDeclaration = [{
    /// Returns true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a memref.vector_cast is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  let hasFolder = 1;
}

#endif // Schedule_OPS