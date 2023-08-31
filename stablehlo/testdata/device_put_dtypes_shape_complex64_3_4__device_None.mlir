// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.733561098,-4.16122341), (0.585794032,3.87093568), (7.09892177,2.50834322), (3.2875762,-0.132075652)], [(1.10832703,1.65133715), (3.37406969,-5.53295898), (-0.780739069,2.70761037), (2.06835699,-0.587331891)], [(1.66946876,3.43079042), (-1.48735178,1.12684846), (0.889513254,-3.00452757), (2.28448224,3.44710016)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.733561098,-4.16122341), (0.585794032,3.87093568), (7.09892177,2.50834322), (3.2875762,-0.132075652)], [(1.10832703,1.65133715), (3.37406969,-5.53295898), (-0.780739069,2.70761037), (2.06835699,-0.587331891)], [(1.66946876,3.43079042), (-1.48735178,1.12684846), (0.889513254,-3.00452757), (2.28448224,3.44710016)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
}
