#ifndef STABLEHLO_TO_LINALG_PASSES
#define STABLEHLO_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def StablehloLegalizeToLinalg
    : Pass<"stablehlo-legalize-to-linalg", "mlir::ModuleOp"> {
  let summary = "Legalize StableHLO to LinAlg";
  let constructor = "mlir::stablehlo::createStablehloLegalizeToLinalgPass()";
  let options = [Option<"enablePrimitiveOps", "enable-primitive-ops", "bool",
                        /*default=*/"false",
                        "Lower to primitive Linalg ops (map, reduce and "
                        "transpose) when possible, instead of linalg.generic">];
}

#endif  // STABLEHLO_TO_LINALG_PASSES
