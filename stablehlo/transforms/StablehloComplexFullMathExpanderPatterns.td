// StablehloComplexFullMathExpanderPatterns.td
include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

// Multiplication on complex inputs:
//
//       (a + bi) * (c + di) = (ac - bd) + (bc + ad)i
//
//     where a, b are real/imaginary parts of lhs, and c, d are real/imaginary
//     parts of rhs.
//
def MulOp_ComplexFullExpander : Pat<
  (StableHLO_MulOp ComplexElementType:$lhs, ComplexElementType:$rhs),
  (StableHLO_ComplexOp
    (StableHLO_SubtractOp
      (StableHLO_MulOp (StableHLO_RealOp:$a $lhs), (StableHLO_RealOp:$c $rhs)),
      (StableHLO_MulOp (StableHLO_ImagOp:$b $lhs), (StableHLO_ImagOp:$d $rhs))),
    (StableHLO_AddOp
      (StableHLO_MulOp $b, $c),
      (StableHLO_MulOp $a, $d)))>;