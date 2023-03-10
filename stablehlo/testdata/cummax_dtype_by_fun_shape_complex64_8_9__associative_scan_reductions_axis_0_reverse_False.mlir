// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xcomplex<f32>>
    %1 = call @expected() : () -> tensor<8x9xcomplex<f32>>
    %2 = call @cummax(%0) : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(6.5357995,-2.7792275), (2.73292637,0.307989419), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (-5.98038959,1.01523423), (-1.11723065,2.71194458), (1.07623279,1.01012182), (0.850069224,-6.27701187)], [(2.1942184,-2.00513959), (0.800208747,1.73119092), (0.778780341,0.89518541), (1.09725976,1.4116466), (-1.21182561,-2.80361414), (4.09927082,1.17952299), (2.38051367,-3.41611814), (1.71133399,6.45389461), (-2.80070853,0.905659317)], [(4.02120161,7.96140861), (5.38851309,-4.82463312), (2.12782097,3.07539034), (-0.415246189,0.765342056), (2.7324791,-0.458747596), (-2.25184345,-2.49996901), (-3.11704206,-7.89855337), (1.99737918,3.28718805), (0.158530965,-4.71318865)], [(-2.64649415,0.332921445), (-4.51284075,0.608356237), (-0.760128438,-0.764673054), (-0.722987771,0.345551819), (-0.560337543,0.680853366), (3.98842192,7.30660724), (-2.19115043,8.75656414), (2.13359761,1.65529561), (3.70258474,-0.390597731)], [(2.49102902,1.548015), (3.39687324,-5.277760e+00), (-3.3527267,-1.3982935), (-5.53009224,1.75702012), (-6.99671316,-6.0474906), (2.11439395,0.0363890566), (-3.87086654,2.09829926), (-6.42520809,-2.89301252), (-3.04615402,3.49825335)], [(-1.22839749,-0.0268511213), (-5.54298782,6.624180e+00), (2.83717942,7.30747604), (3.80598569,-2.48159504), (2.80593801,-0.614761472), (1.31450272,0.959255814), (2.02760506,-3.08672762), (4.30183029,-1.4150393), (4.27513123,0.601196408)], [(1.30075431,-4.30424356), (-4.59835339,1.71006858), (-0.828599691,0.949571371), (1.05611324,3.61841965), (2.67778206,1.04997039), (1.49463129,-0.914679706), (4.08357477,-2.41646504), (-2.00456047,3.07508135), (1.20975566,1.18347311)], [(3.49871397,0.800018787), (-6.4502759,5.683210e+00), (-2.74574089,-0.777059972), (-3.5486865,7.168020e-02), (-1.19251311,0.415864885), (-1.51767683,-1.90616369), (-0.274695247,-4.67734575), (-0.334966034,1.31084299), (-2.42454481,-2.20765352)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(6.5357995,-2.7792275), (2.73292637,0.307989419), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (-5.98038959,1.01523423), (-1.11723065,2.71194458), (1.07623279,1.01012182), (0.850069224,-6.27701187)], [(6.5357995,-2.7792275), (2.73292637,0.307989419), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (2.38051367,-3.41611814), (1.71133399,6.45389461), (0.850069224,-6.27701187)], [(6.5357995,-2.7792275), (5.38851309,-4.82463312), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (2.38051367,-3.41611814), (1.99737918,3.28718805), (0.850069224,-6.27701187)], [(6.5357995,-2.7792275), (5.38851309,-4.82463312), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (2.38051367,-3.41611814), (2.13359761,1.65529561), (3.70258474,-0.390597731)], [(6.5357995,-2.7792275), (5.38851309,-4.82463312), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (2.38051367,-3.41611814), (2.13359761,1.65529561), (3.70258474,-0.390597731)], [(6.5357995,-2.7792275), (5.38851309,-4.82463312), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (2.38051367,-3.41611814), (4.30183029,-1.4150393), (4.27513123,0.601196408)], [(6.5357995,-2.7792275), (5.38851309,-4.82463312), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (4.08357477,-2.41646504), (4.30183029,-1.4150393), (4.27513123,0.601196408)], [(6.5357995,-2.7792275), (5.38851309,-4.82463312), (4.29393339,-1.4793011), (4.5814867,-2.81891465), (4.28584385,-0.0542999506), (4.09927082,1.17952299), (4.08357477,-2.41646504), (4.30183029,-1.4150393), (4.27513123,0.601196408)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @cummax(%arg0: tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %2 = stablehlo.real %0 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %3 = stablehlo.real %1 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %5 = stablehlo.compare  GT, %2, %3,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %6 = stablehlo.imag %0 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %7 = stablehlo.imag %1 : (tensor<4x9xcomplex<f32>>) -> tensor<4x9xf32>
    %8 = stablehlo.compare  GT, %6, %7,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %9 = stablehlo.select %4, %8, %5 : tensor<4x9xi1>, tensor<4x9xi1>
    %10 = stablehlo.select %9, %0, %1 : tensor<4x9xi1>, tensor<4x9xcomplex<f32>>
    %11 = "stablehlo.slice"(%10) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %12 = "stablehlo.slice"(%10) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %13 = stablehlo.real %11 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %14 = stablehlo.real %12 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %15 = stablehlo.compare  EQ, %13, %14,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %16 = stablehlo.compare  GT, %13, %14,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %17 = stablehlo.imag %11 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %18 = stablehlo.imag %12 : (tensor<2x9xcomplex<f32>>) -> tensor<2x9xf32>
    %19 = stablehlo.compare  GT, %17, %18,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %20 = stablehlo.select %15, %19, %16 : tensor<2x9xi1>, tensor<2x9xi1>
    %21 = stablehlo.select %20, %11, %12 : tensor<2x9xi1>, tensor<2x9xcomplex<f32>>
    %22 = "stablehlo.slice"(%21) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %23 = "stablehlo.slice"(%21) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %24 = stablehlo.real %22 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %25 = stablehlo.real %23 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %26 = stablehlo.compare  EQ, %24, %25,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %27 = stablehlo.compare  GT, %24, %25,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %28 = stablehlo.imag %22 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %29 = stablehlo.imag %23 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %30 = stablehlo.compare  GT, %28, %29,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %31 = stablehlo.select %26, %30, %27 : tensor<1x9xi1>, tensor<1x9xi1>
    %32 = stablehlo.select %31, %22, %23 : tensor<1x9xi1>, tensor<1x9xcomplex<f32>>
    %33 = "stablehlo.slice"(%32) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %34 = "stablehlo.slice"(%21) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %35 = stablehlo.real %33 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %36 = stablehlo.real %34 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %37 = stablehlo.compare  EQ, %35, %36,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %38 = stablehlo.compare  GT, %35, %36,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %39 = stablehlo.imag %33 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %40 = stablehlo.imag %34 : (tensor<0x9xcomplex<f32>>) -> tensor<0x9xf32>
    %41 = stablehlo.compare  GT, %39, %40,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
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
    %56 = stablehlo.compare  GT, %53, %54,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %57 = stablehlo.imag %51 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %58 = stablehlo.imag %52 : (tensor<1x9xcomplex<f32>>) -> tensor<1x9xf32>
    %59 = stablehlo.compare  GT, %57, %58,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
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
    %74 = stablehlo.compare  GT, %71, %72,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
    %75 = stablehlo.imag %69 : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
    %76 = stablehlo.imag %70 : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
    %77 = stablehlo.compare  GT, %75, %76,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
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
