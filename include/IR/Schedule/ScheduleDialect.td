#ifndef Schedule_DIALECT
#define Schedule_DIALECT
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Schedule dialect definition.
//===----------------------------------------------------------------------===//

def Schedule_Dialect : Dialect {
    let name = "schedule";
    let summary = "Dialect for schedule.";
    let description = [{
    }];
    let cppNamespace = "::mlir::schedule";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Base Schedule operation definition.
//===----------------------------------------------------------------------===//

class Schedule_Op<string mnemonic, list<Trait> traits = []> :
        Op<Schedule_Dialect, mnemonic, traits>;

#endif // Schedule_DIALECT