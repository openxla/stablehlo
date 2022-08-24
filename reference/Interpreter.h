#ifndef STABLEHLO_REFERENCE_INTERPRETER_H
#define STABLEHLO_REFERENCE_INTERPRETER_H

#include "Tensor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace stablehlo {

/// Evaluating an mlir function.
///
/// Assuming that the function under evaluation has passed verifier,
/// similarly to what's required by constant folding.
llvm::Expected<SmallVector<Tensor>> eval(func::FuncOp func,
                                         ArrayRef<Tensor> args);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INTERPRETER_H
