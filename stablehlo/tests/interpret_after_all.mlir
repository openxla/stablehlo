// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @after_all_op_test() {
  %input0 = stablehlo.after_all : !stablehlo.token
  %input1 = stablehlo.after_all : !stablehlo.token
  %result = stablehlo.after_all %input0, %input1 : !stablehlo.token
  func.return
}
