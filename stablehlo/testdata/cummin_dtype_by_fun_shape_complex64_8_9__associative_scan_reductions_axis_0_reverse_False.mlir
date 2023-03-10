// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xcomplex<f32>>
    %1 = call @expected() : () -> tensor<8x9xcomplex<f32>>
    %2 = call @cummin(%0) : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-1.17474532,1.45273948), (3.33831954,-3.13411498), (0.00746119535,-5.32951307), (4.34889841,-0.455768675), (3.33289957,0.335708141), (0.684122682,0.241357967), (-5.7327714,-3.34056282), (-0.217550784,0.991709291), (-0.517950118,-2.28763151)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-5.0064764,-0.851399421), (4.77812481,5.34348726), (6.07070732,-5.484610e+00), (-1.40287507,-4.63588905), (0.244590476,-2.67975283), (-4.34912395,2.22413445), (-1.7060293,-2.59180641)], [(4.420650e+00,-5.62677097), (2.44912601,-1.90460062), (2.52371383,1.37112224), (4.51757908,1.15632951), (1.17931747,0.536808252), (-0.929382741,0.101375341), (0.581020296,2.06747174), (-5.13705397,0.48004207), (-2.67845631,-1.97863626)], [(1.50587177,0.393401563), (-1.07476115,0.128974959), (-0.363187462,-2.48696709), (0.756013632,-5.17391777), (-3.060150e+00,0.270029187), (2.27615118,6.29205799), (-1.54174316,5.53210688), (2.7387476,1.48289526), (-2.79610085,4.59517193)], [(-5.04979467,-1.94924843), (-3.07939553,-2.26976657), (0.839843869,1.465680e+00), (-3.53745413,2.68166399), (-0.727278292,-1.764884), (3.97724319,3.01025844), (2.0866096,1.5009023), (6.56307173,-0.346638829), (3.91638088,3.8723197)], [(-0.944476902,-1.68290424), (1.48419631,-1.20026064), (-6.11141396,-0.0978196859), (5.97482729,0.632251322), (0.597339392,-1.13477159), (-1.45075881,3.78286362), (1.39976358,2.21757674), (2.7341888,-2.9443121), (1.63230407,7.85966825)], [(1.57729936,-3.2821548), (-1.25529408,1.75556695), (7.09212493,-2.12055445), (5.215780e+00,-0.923876166), (-3.92325759,0.0402702354), (-3.85134339,0.325153261), (0.859587609,8.3802061), (1.57020557,1.7646147), (2.97805905,-1.17994606)], [(3.15905833,-0.975567281), (-4.16509914,-3.30496812), (7.60827541,-2.64981389), (3.02684474,2.75510216), (-4.54449034,3.19854355), (3.61844182,6.39453125), (-2.24176836,2.65369487), (4.46538305,-0.469230175), (-3.43214893,0.282342345)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-1.17474532,1.45273948), (3.33831954,-3.13411498), (0.00746119535,-5.32951307), (4.34889841,-0.455768675), (3.33289957,0.335708141), (0.684122682,0.241357967), (-5.7327714,-3.34056282), (-0.217550784,0.991709291), (-0.517950118,-2.28763151)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-5.0064764,-0.851399421), (4.34889841,-0.455768675), (3.33289957,0.335708141), (-1.40287507,-4.63588905), (-5.7327714,-3.34056282), (-4.34912395,2.22413445), (-1.7060293,-2.59180641)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-5.0064764,-0.851399421), (4.34889841,-0.455768675), (1.17931747,0.536808252), (-1.40287507,-4.63588905), (-5.7327714,-3.34056282), (-5.13705397,0.48004207), (-2.67845631,-1.97863626)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-5.0064764,-0.851399421), (0.756013632,-5.17391777), (-3.060150e+00,0.270029187), (-1.40287507,-4.63588905), (-5.7327714,-3.34056282), (-5.13705397,0.48004207), (-2.79610085,4.59517193)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-5.0064764,-0.851399421), (-3.53745413,2.68166399), (-3.060150e+00,0.270029187), (-1.40287507,-4.63588905), (-5.7327714,-3.34056282), (-5.13705397,0.48004207), (-2.79610085,4.59517193)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-6.11141396,-0.0978196859), (-3.53745413,2.68166399), (-3.060150e+00,0.270029187), (-1.45075881,3.78286362), (-5.7327714,-3.34056282), (-5.13705397,0.48004207), (-2.79610085,4.59517193)], [(-5.116310e+00,2.94425654), (-3.37200522,-4.3753953), (-6.11141396,-0.0978196859), (-3.53745413,2.68166399), (-3.92325759,0.0402702354), (-3.85134339,0.325153261), (-5.7327714,-3.34056282), (-5.13705397,0.48004207), (-2.79610085,4.59517193)], [(-5.116310e+00,2.94425654), (-4.16509914,-3.30496812), (-6.11141396,-0.0978196859), (-3.53745413,2.68166399), (-4.54449034,3.19854355), (-3.85134339,0.325153261), (-5.7327714,-3.34056282), (-5.13705397,0.48004207), (-3.43214893,0.282342345)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @cummin(%arg0: tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %2 = stablehlo.real %0 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %3 = stablehlo.real %1 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %5 = stablehlo.compare  LT, %2, %3,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %6 = stablehlo.imag %0 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %7 = stablehlo.imag %1 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %8 = stablehlo.compare  LT, %6, %7,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %9 = stablehlo.select %4, %8, %5 : tensor<4x9xi1>, tensor<4x9xi1>
    %10 = stablehlo.select %9, %0, %1 : tensor<4x9xi1>, tensor<4x9xcomplex<f32>>
    %11 = "stablehlo.slice"(%10) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %12 = "stablehlo.slice"(%10) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %13 = stablehlo.real %11 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %14 = stablehlo.real %12 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %15 = stablehlo.compare  EQ, %13, %14,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %16 = stablehlo.compare  LT, %13, %14,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %17 = stablehlo.imag %11 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %18 = stablehlo.imag %12 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %19 = stablehlo.compare  LT, %17, %18,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %20 = stablehlo.select %15, %19, %16 : tensor<2x9xi1>, tensor<2x9xi1>
    %21 = stablehlo.select %20, %11, %12 : tensor<2x9xi1>, tensor<2x9xcomplex<f32>>
    %22 = "stablehlo.slice"(%21) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %23 = "stablehlo.slice"(%21) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %24 = stablehlo.real %22 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %25 = stablehlo.real %23 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %26 = stablehlo.compare  EQ, %24, %25,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %27 = stablehlo.compare  LT, %24, %25,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %28 = stablehlo.imag %22 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %29 = stablehlo.imag %23 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %30 = stablehlo.compare  LT, %28, %29,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %31 = stablehlo.select %26, %30, %27 : tensor<1x9xi1>, tensor<1x9xi1>
    %32 = stablehlo.select %31, %22, %23 : tensor<1x9xi1>, tensor<1x9xcomplex<f32>>
    %33 = "stablehlo.slice"(%32) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %34 = "stablehlo.slice"(%21) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %35 = stablehlo.real %33 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %36 = stablehlo.real %34 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %37 = stablehlo.compare  EQ, %35, %36,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %38 = stablehlo.compare  LT, %35, %36,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %39 = stablehlo.imag %33 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %40 = stablehlo.imag %34 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %41 = stablehlo.compare  LT, %39, %40,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %42 = stablehlo.select %37, %41, %38 : tensor<0x9xi1>, tensor<0x9xi1>
    %43 = stablehlo.select %42, %33, %34 : tensor<0x9xi1>, tensor<0x9xcomplex<f32>>
    %44 = "stablehlo.slice"(%21) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %45 = stablehlo.concatenate %44, %43, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<0x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %46 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %47 = stablehlo.pad %45, %46, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %48 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %49 = stablehlo.pad %32, %48, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %50 = stablehlo.add %47, %49 : tensor<2x9xcomplex<f32>>
    %51 = "stablehlo.slice"(%50) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %52 = "stablehlo.slice"(%10) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %53 = stablehlo.real %51 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %54 = stablehlo.real %52 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %55 = stablehlo.compare  EQ, %53, %54,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %56 = stablehlo.compare  LT, %53, %54,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %57 = stablehlo.imag %51 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %58 = stablehlo.imag %52 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %59 = stablehlo.compare  LT, %57, %58,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %60 = stablehlo.select %55, %59, %56 : tensor<1x9xi1>, tensor<1x9xi1>
    %61 = stablehlo.select %60, %51, %52 : tensor<1x9xi1>, tensor<1x9xcomplex<f32>>
    %62 = "stablehlo.slice"(%10) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %63 = stablehlo.concatenate %62, %61, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<1x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %64 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %65 = stablehlo.pad %63, %64, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %66 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %67 = stablehlo.pad %50, %66, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %68 = stablehlo.add %65, %67 : tensor<4x9xcomplex<f32>>
    %69 = "stablehlo.slice"(%68) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %70 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %71 = stablehlo.real %69 : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
    %72 = stablehlo.real %70 : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
    %73 = stablehlo.compare  EQ, %71, %72,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
    %74 = stablehlo.compare  LT, %71, %72,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
    %75 = stablehlo.imag %69 : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
    %76 = stablehlo.imag %70 : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
    %77 = stablehlo.compare  LT, %75, %76,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
    %78 = stablehlo.select %73, %77, %74 : tensor<3x9xi1>, tensor<3x9xi1>
    %79 = stablehlo.select %78, %69, %70 : tensor<3x9xi1>, tensor<3x9xcomplex<f32>>
    %80 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %81 = stablehlo.concatenate %80, %79, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<3x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %82 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %83 = stablehlo.pad %81, %82, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    %84 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %85 = stablehlo.pad %68, %84, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    %86 = stablehlo.add %83, %85 : tensor<8x9xcomplex<f32>>
    return %86 : tensor<8x9xcomplex<f32>>
  }
}
