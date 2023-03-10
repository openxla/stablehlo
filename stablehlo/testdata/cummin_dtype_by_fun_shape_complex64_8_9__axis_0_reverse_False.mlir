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
    %0 = stablehlo.constant dense<[[(-2.20064282,11.8359995), (-1.18065298,0.520443261), (5.54958057,2.38388181), (-5.66111183,1.3423456), (-1.09832978,2.45024061), (-5.43847418,-0.785757958), (-3.70082021,1.83477271), (2.81763911,2.11622381), (0.553307772,1.21837747)], [(2.13919377,3.62939119), (-3.41908169,3.42248225), (4.05249786,-0.655110418), (6.41896057,5.24311733), (-1.8206377,-4.54731226), (-0.287267864,-0.90525651), (4.04449129,-1.01303935), (2.02960563,-0.624596416), (-4.62202311,-1.50080669)], [(3.62444782,-0.45376417), (2.00645947,-3.74251866), (4.23192501,3.84629369), (-8.69433221E-4,-0.579477847), (3.59787154,-0.195620179), (-0.557578564,7.990080e-01), (0.187422574,-3.61917782), (-0.00638067536,0.640237808), (-3.67943764,3.53171444)], [(-3.2980957,4.45592356), (-0.847739815,1.93973708), (-2.44714737,4.18023443), (-8.686960e+00,-6.39456939), (-2.72250366,1.88301229), (-0.526008785,5.67890406), (6.58279514,4.7318759), (2.62520552,-2.60551143), (-0.919276356,5.83751535)], [(-2.13377595,0.0981623753), (0.0914229304,-5.96475124), (3.09548616,-2.05826664), (0.676138699,3.10922861), (-0.038489569,0.907395899), (4.87626696,-1.59946942), (-3.84020066,-2.89542341), (4.88541746,0.413048774), (0.641275465,-1.97885835)], [(4.99621916,-2.1540432), (1.91084325,0.0111736599), (1.72374129,2.06933546), (-2.71274638,3.83613563), (4.25857782,-6.61315966), (3.66232419,-0.763396263), (-3.35298944,-2.4677341), (-3.48226309,-3.53209853), (-2.63404799,0.434243083)], [(2.21997643,1.70412087), (2.79328108,1.28366399), (-3.28768015,-1.95127523), (0.446246326,0.31483382), (-5.18738365,0.261043727), (1.84123647,0.424709171), (-1.62078512,-2.35618782), (-4.15646744,-1.26606512), (3.53669095,0.532645643)], [(-4.71848488,-2.76528215), (-5.09705067,0.436411411), (0.027917413,-1.46016717), (0.832778394,6.65381765), (-0.354224384,4.50678301), (-3.03318095,-3.12161851), (1.88670373,6.47926903), (3.2475543,-1.14652145), (3.93028831,-0.613119602)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-2.20064282,11.8359995), (-1.18065298,0.520443261), (5.54958057,2.38388181), (-5.66111183,1.3423456), (-1.09832978,2.45024061), (-5.43847418,-0.785757958), (-3.70082021,1.83477271), (2.81763911,2.11622381), (0.553307772,1.21837747)], [(-2.20064282,11.8359995), (-3.41908169,3.42248225), (4.05249786,-0.655110418), (-5.66111183,1.3423456), (-1.8206377,-4.54731226), (-5.43847418,-0.785757958), (-3.70082021,1.83477271), (2.02960563,-0.624596416), (-4.62202311,-1.50080669)], [(-2.20064282,11.8359995), (-3.41908169,3.42248225), (4.05249786,-0.655110418), (-5.66111183,1.3423456), (-1.8206377,-4.54731226), (-5.43847418,-0.785757958), (-3.70082021,1.83477271), (-0.00638067536,0.640237808), (-4.62202311,-1.50080669)], [(-3.2980957,4.45592356), (-3.41908169,3.42248225), (-2.44714737,4.18023443), (-8.686960e+00,-6.39456939), (-2.72250366,1.88301229), (-5.43847418,-0.785757958), (-3.70082021,1.83477271), (-0.00638067536,0.640237808), (-4.62202311,-1.50080669)], [(-3.2980957,4.45592356), (-3.41908169,3.42248225), (-2.44714737,4.18023443), (-8.686960e+00,-6.39456939), (-2.72250366,1.88301229), (-5.43847418,-0.785757958), (-3.84020066,-2.89542341), (-0.00638067536,0.640237808), (-4.62202311,-1.50080669)], [(-3.2980957,4.45592356), (-3.41908169,3.42248225), (-2.44714737,4.18023443), (-8.686960e+00,-6.39456939), (-2.72250366,1.88301229), (-5.43847418,-0.785757958), (-3.84020066,-2.89542341), (-3.48226309,-3.53209853), (-4.62202311,-1.50080669)], [(-3.2980957,4.45592356), (-3.41908169,3.42248225), (-3.28768015,-1.95127523), (-8.686960e+00,-6.39456939), (-5.18738365,0.261043727), (-5.43847418,-0.785757958), (-3.84020066,-2.89542341), (-4.15646744,-1.26606512), (-4.62202311,-1.50080669)], [(-4.71848488,-2.76528215), (-5.09705067,0.436411411), (-3.28768015,-1.95127523), (-8.686960e+00,-6.39456939), (-5.18738365,0.261043727), (-5.43847418,-0.785757958), (-3.84020066,-2.89542341), (-4.15646744,-1.26606512), (-4.62202311,-1.50080669)]]> : tensor<8x9xcomplex<f32>>
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
