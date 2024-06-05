#ifndef STABLEHLO_REFERENCE_INTERPRETERPASSES_H
#define STABLEHLO_REFERENCE_INTERPRETERPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "stablehlo/reference/InterpreterPasses.h.inc"

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INTERPRETERPASSES_H
