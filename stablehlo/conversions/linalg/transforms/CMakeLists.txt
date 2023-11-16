set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloLinalgTransforms)
add_public_tablegen_target(StablehloLinalgTransformsPassIncGen)

add_mlir_library(StablehloLinalgTransforms
  LegalizeToLinalgUtils.cpp
  StablehloLegalizeToLinalg.cpp
  StablehloToArith.cpp
  StablehloToLinalgConvolution.cpp
  StablehloToLinalgDotProduct.cpp
  StablehloToLinalgPointwise.cpp
  StablehloToLinalgRandom.cpp
  StablehloToLinalgReduce.cpp
  TypeConversion.cpp

  DEPENDS
  StablehloOpsIncGen
  StablehloLinalgTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRPass
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRSparseTensorDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransforms
  MLIRVectorDialect
)
