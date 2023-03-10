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
    %0 = stablehlo.constant dense<[[(0.0367246121,2.1384654), (-2.4269197,2.25204277), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (-0.322467983,-1.13710141), (-2.166924,-1.65235698), (-0.677196145,4.05115843), (1.37253499,-0.834335982), (-0.304116786,4.46156502)], [(7.59909439,-1.67791414), (5.73667192,-1.46152294), (-2.57594299,-0.248367906), (1.28263831,2.67226434), (-0.343415529,-2.09594059), (0.00196250016,4.77483606), (-1.19021177,2.98825359), (-3.06018949,-3.17968059), (0.837167799,0.469227463)], [(1.27063632,2.03051305), (-5.20038319,3.98520923), (2.71087551,-2.8366847), (-0.506014466,-2.45549917), (-1.06492448,1.52090871), (-0.257419616,2.41203189), (-1.2128737,-1.89222562), (-1.32079566,2.59598851), (-3.71853471,1.41927803)], [(2.28757524,1.73782885), (0.418639153,-1.65113604), (-2.29021525,-3.88727236), (-3.57268858,-1.1298486), (-0.751377642,-2.39842987), (2.61515546,-0.140380546), (-2.98863626,0.906749784), (4.06566191,2.88925934), (3.32295847,1.34868038)], [(2.8156085,2.27078032), (2.61354542,6.14098406), (1.53869736,2.00430417), (1.48748791,-1.870680e+00), (0.500480413,2.76545095), (-1.38220215,6.99490929), (4.24028683,-4.0742135), (0.294065267,5.36099768), (-2.70962286,1.12287164)], [(0.697526217,1.10667956), (3.46375346,-0.56666708), (0.714748203,1.62516105), (2.0087347,0.759792149), (1.47468913,-2.81950235), (-1.01028609,2.70139956), (3.06188107,-2.15245032), (1.96856654,-2.90111923), (0.700020552,0.340714842)], [(-0.566443622,3.76940966), (6.45704936,0.755279839), (-2.367000e+00,-2.19347596), (3.36068964,-0.415230662), (-5.489120e+00,-0.550776184), (-0.294645756,0.50101006), (-4.8256259,2.85788894), (1.2723285,-0.466926962), (-5.54240704,5.92755938)], [(1.45783305,-4.16242313), (4.08748817,-1.67441046), (3.28540468,3.55365705), (2.03922629,-4.45731258), (1.77956593,3.45115304), (0.785979033,2.36783695), (0.11686527,-2.08886147), (-1.06777203,1.54833686), (3.87946248,1.27718592)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.0367246121,2.1384654), (-2.4269197,2.25204277), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (-0.322467983,-1.13710141), (-2.166924,-1.65235698), (-0.677196145,4.05115843), (1.37253499,-0.834335982), (-0.304116786,4.46156502)], [(7.59909439,-1.67791414), (5.73667192,-1.46152294), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (-0.322467983,-1.13710141), (0.00196250016,4.77483606), (-0.677196145,4.05115843), (1.37253499,-0.834335982), (0.837167799,0.469227463)], [(7.59909439,-1.67791414), (5.73667192,-1.46152294), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (-0.322467983,-1.13710141), (0.00196250016,4.77483606), (-0.677196145,4.05115843), (1.37253499,-0.834335982), (0.837167799,0.469227463)], [(7.59909439,-1.67791414), (5.73667192,-1.46152294), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (-0.322467983,-1.13710141), (2.61515546,-0.140380546), (-0.677196145,4.05115843), (4.06566191,2.88925934), (3.32295847,1.34868038)], [(7.59909439,-1.67791414), (5.73667192,-1.46152294), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (0.500480413,2.76545095), (2.61515546,-0.140380546), (4.24028683,-4.0742135), (4.06566191,2.88925934), (3.32295847,1.34868038)], [(7.59909439,-1.67791414), (5.73667192,-1.46152294), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (1.47468913,-2.81950235), (2.61515546,-0.140380546), (4.24028683,-4.0742135), (4.06566191,2.88925934), (3.32295847,1.34868038)], [(7.59909439,-1.67791414), (6.45704936,0.755279839), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (1.47468913,-2.81950235), (2.61515546,-0.140380546), (4.24028683,-4.0742135), (4.06566191,2.88925934), (3.32295847,1.34868038)], [(7.59909439,-1.67791414), (6.45704936,0.755279839), (3.58617973,-3.21912313), (5.67304039,-0.812399446), (1.77956593,3.45115304), (2.61515546,-0.140380546), (4.24028683,-4.0742135), (4.06566191,2.88925934), (3.87946248,1.27718592)]]> : tensor<8x9xcomplex<f32>>
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
