// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xcomplex<f32>>
    %1 = call @expected() : () -> tensor<8x9xcomplex<f32>>
    %2 = call @cumsum(%0) : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(1.74231732,1.22513902), (1.26369822,2.77053022), (1.69850612,3.14500856), (0.265825123,-0.930303215), (-1.94026065,1.86232102), (-1.40567982,-0.820742249), (-2.98310256,-7.63109064), (-5.4709568,4.2049613), (0.40909791,2.82846475)], [(-1.54555774,1.83906817), (-0.894468605,0.080695048), (-1.80118322,-2.39747548), (1.47228277,3.91463947), (-6.27834511,-2.34313965), (-1.24802125,-0.0181794316), (2.55661273,-4.48410082), (0.799726545,-1.177001), (2.93339205,4.05520391)], [(0.761015713,3.09393239), (-0.642010033,-1.86784089), (-8.03944492,-5.049710e+00), (3.57996655,2.18553782), (-1.500720e+00,-4.41989088), (-3.08382368,-2.79776812), (3.20352793,-1.32676017), (-2.51952767,-4.145257), (-2.65034437,2.56483316)], [(6.37612104,-0.326897115), (-0.66182965,-0.278280199), (-2.5070672,-3.26727509), (-1.05983591,0.385778129), (1.931862,7.43623781), (-0.0216259211,-3.54014659), (-7.9497013,-1.72983968), (2.23175025,1.46199858), (-4.166821,-1.99810076)], [(-7.31446695,3.10778069), (2.11536622,-6.83897734), (0.766712129,5.85192299), (-0.326745063,2.8194983), (0.523457646,1.96032083), (0.751640677,-4.02761364), (-7.03543901,1.85012829), (-9.356120e-01,2.35867739), (-1.05426061,-2.37098265)], [(-1.90175176,-0.353284031), (-3.21852446,-0.55749023), (1.95937216,2.83737659), (2.82445931,1.86049676), (-4.89520884,-1.81733131), (3.28786874,-0.408412158), (-5.49041891,-3.01566195), (-0.0417231657,-3.31881905), (-1.87838638,-3.91749382)], [(-0.895371139,-3.12053013), (-1.14760447,5.47131205), (-1.96879148,-0.518534124), (4.87572289,-2.13756728), (0.153803989,3.83767152), (3.386400e+00,-0.199190229), (2.20877337,-2.13424945), (0.900430202,-6.354010e-01), (1.73026145,2.81834912)], [(-1.37075484,-2.70611286), (-2.76667809,-1.02258766), (0.342851758,-2.01437879), (1.76208842,7.06388474), (4.93326521,-0.0361584537), (-3.73740125,-0.394069016), (2.44618678,0.393722415), (-0.690608621,6.4777565), (0.104192138,4.15748739)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(1.74231732,1.22513902), (1.26369822,2.77053022), (1.69850612,3.14500856), (0.265825123,-0.930303215), (-1.94026065,1.86232102), (-1.40567982,-0.820742249), (-2.98310256,-7.63109064), (-5.4709568,4.2049613), (0.40909791,2.82846475)], [(0.196759582,3.06420708), (0.369229615,2.85122538), (-0.102677107,0.747533083), (1.73810792,2.98433638), (-8.21860599,-0.480818629), (-2.65370107,-0.838921666), (-0.42648983,-12.1151915), (-4.67123032,3.0279603), (3.342490e+00,6.8836689)], [(0.957775294,6.15813922), (-0.272780418,0.98338449), (-8.14212226,-4.30217648), (5.31807423,5.16987419), (-9.71932601,-4.90070963), (-5.73752499,-3.63668966), (2.7770381,-13.4419518), (-7.19075775,-1.1172967), (0.692145586,9.44850158)], [(7.33389664,5.83124256), (-0.934610068,0.705104351), (-10.6491899,-7.56945228), (4.25823879,5.55565262), (-7.78746414,2.53552818), (-5.75915051,-7.17683601), (-5.17266321,-15.1717911), (-4.95900774,0.344701767), (-3.47467542,7.4504013)], [(0.0194296837,8.93902301), (1.18075609,-6.13387298), (-9.88247776,-1.7175293), (3.93149376,8.37515068), (-7.26400661,4.49584913), (-5.00750971,-11.2044497), (-12.2081022,-13.3216629), (-5.894620e+00,2.70337915), (-4.52893591,5.07941866)], [(-1.88232231,8.58573913), (-2.03776836,-6.69136333), (-7.92310572,1.1198473), (6.75595284,10.2356472), (-12.1592159,2.67851782), (-1.71964121,-11.6128616), (-17.6985207,-16.3373241), (-5.93634319,-0.615439891), (-6.40732241,1.16192484)], [(-2.77769351,5.46520901), (-3.18537283,-1.22005129), (-9.8918972,0.601313174), (11.6316757,8.098080e+00), (-12.0054121,6.51618958), (1.66675878,-11.8120518), (-15.489747,-18.4715729), (-5.03591299,-1.2508409), (-4.67706108,3.98027396)], [(-4.14844799,2.75909615), (-5.95205069,-2.24263906), (-9.54904556,-1.41306591), (13.3937645,15.1619644), (-7.07214641,6.48003054), (-2.07064247,-12.2061214), (-13.043561,-18.0778522), (-5.72652149,5.22691584), (-4.57286882,8.13776206)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @cumsum(%arg0: tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %2 = stablehlo.add %0, %1 : tensor<4x9xcomplex<f32>>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %5 = stablehlo.add %3, %4 : tensor<2x9xcomplex<f32>>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %8 = stablehlo.add %6, %7 : tensor<1x9xcomplex<f32>>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %11 = stablehlo.add %9, %10 : tensor<0x9xcomplex<f32>>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<0x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %14 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %16 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %18 = stablehlo.add %15, %17 : tensor<2x9xcomplex<f32>>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %21 = stablehlo.add %19, %20 : tensor<1x9xcomplex<f32>>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<1x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %24 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %26 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %28 = stablehlo.add %25, %27 : tensor<4x9xcomplex<f32>>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %31 = stablehlo.add %29, %30 : tensor<3x9xcomplex<f32>>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<3x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %34 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    %36 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    %38 = stablehlo.add %35, %37 : tensor<8x9xcomplex<f32>>
    return %38 : tensor<8x9xcomplex<f32>>
  }
}
