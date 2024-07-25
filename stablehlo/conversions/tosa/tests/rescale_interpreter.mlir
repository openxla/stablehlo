// RUN: stablehlo-opt %s --stablehlo-quant-legalize-to-tosa-rescale --tosa-rescale-legalize-to-stablehlo | stablehlo-translate --interpret

func.func @main() -> tensor<i1> {
  %input0 = stablehlo.constant dense<0.25> : tensor<2x2xf32>
  %input1 = stablehlo.constant dense<0.75> : tensor<2x2xf32>
  %expected = stablehlo.constant dense<1.05> : tensor<2x2xf32>

  %arg0 = stablehlo.uniform_quantize %input0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>
  %arg1 = stablehlo.uniform_quantize %input1 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>

  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  %result = stablehlo.uniform_dequantize %0: (tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>) -> tensor<2x2xf32>
  %3 = stablehlo.custom_call @check.eq(%result, %expected) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<i1>
  return %3 : tensor<i1>
}

