// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xcomplex<f32>>
    %1 = call @expected() : () -> tensor<8x9xcomplex<f32>>
    %2 = call @cumprod(%0) : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.781118988,3.93758821), (-3.41993785,2.84913111), (-2.56545496,-0.925626933), (3.12654114,5.40519094), (-4.66526508,1.09738696), (6.70357608,-0.35456872), (2.27673793,5.14490843), (-6.86809111,0.343750209), (-0.909921348,3.0497582)], [(-0.2385813,-1.10970867), (-3.16484523,-0.808798849), (0.323681444,5.35255194), (0.0812853872,-2.04596543), (0.31847015,-2.55459189), (5.40177917,-0.127950341), (5.59475946,0.351106882), (1.72592223,-0.656314551), (2.74138308,2.18550634)], [(1.60982132,-1.8616215), (-2.6965363,2.37751031), (1.81461108,0.211054608), (-1.87186813,2.44813418), (-1.58922815,0.320431769), (2.49496579,4.13575029), (2.27119327,3.29366541), (0.590414464,1.68322957), (2.08161068,2.54589438)], [(2.99332952,-0.377812117), (2.44941258,-1.42010725), (2.24766016,-1.92927492), (-1.49355173,5.20451975), (-2.53146744,-1.46990156), (-0.7423141,1.68648136), (-0.370996386,-3.2195282), (-1.65768313,0.768476903), (-3.20078683,-0.530567944)], [(-1.49452949,0.760317146), (-0.0305944793,1.1298641), (2.42156863,-4.79442024), (-1.581128,1.28170466), (-4.07424641,1.51814437), (-0.735528469,2.57247019), (4.5229125,1.32325482), (-5.36087418,2.55566692), (2.6009109,4.78883648)], [(0.812102556,0.379923046), (-1.18347549,-0.488237143), (6.13666677,1.35144866), (-2.94606781,4.08570862), (6.68644809,0.566082716), (1.21718132,4.8484292), (1.45259941,0.83092767), (6.61041832,-3.09862185), (-4.83394814,3.76932645)], [(-0.602656484,-2.40166736), (1.06079745,-0.629714608), (1.61294985,-7.22106886), (1.53095639,-2.024330e+00), (1.61695457,3.33276105), (-3.74284315,4.86331606), (2.29340744,-0.360509634), (-1.04828858,2.63180161), (2.25345182,0.316261351)], [(4.987510e+00,5.38591814), (0.278971761,-0.923547923), (-2.38806319,-11.5499077), (-1.16359675,4.09007168), (-0.526184559,-0.729002833), (1.85443926,0.608784198), (0.200279832,-3.60206294), (4.0042038,-1.04570735), (-2.21081185,-0.266328722)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.781118988,3.93758821), (-3.41993785,2.84913111), (-2.56545496,-0.925626933), (3.12654114,5.40519094), (-4.66526508,1.09738696), (6.70357608,-0.35456872), (2.27673793,5.14490843), (-6.86809111,0.343750209), (-0.909921348,3.0497582)], [(4.55593634,-0.0726204216), (13.1279478,-6.25101709), (4.12407637,-14.0313387), (11.3129759,-5.95743227), (1.31762803,12.267334), (36.1658707,-2.7730267), (10.9313879,29.5839043), (-11.6281824,5.100914), (-9.15970897,6.37191677)], [(7.19905185,-8.59833526), (-20.5381298,48.0679245), (10.444973,-24.5910168), (-6.59180545,38.8472099), (-6.02485514,-19.0733814), (101.701157,142.654404), (-72.6121902,103.195099), (-15.4514561,-16.5612469), (-35.2891769,-10.0558014)], [(18.3005791,-28.4575386), (17.9552517,146.904526), (-23.9660797,-75.4234695), (-192.335876,-92.3274993), (-12.7842722,57.1395912), (-316.078186,65.6227188), (359.178406,195.491959), (38.3405571,15.5792131), (107.617836,50.9097786)], [(-5.714000e+00,56.4448776), (-166.531479,15.7925272), (-419.647308,-67.7396545), (422.444214,-100.536194), (-34.659874,-252.209152), (63.67202,-861.369079), (1365.8468,1359.47766), (-245.354187,14.4674902), (36.1058044,6.477760e+02)], [(-26.0850639,43.668148), (204.796417,62.616787), (-2483.68921,-982.827575), (-833.78772,2022.17053), (-88.9802093,-1706.00378), (4253.7876,-739.733032), (854.400757,3109.69629), (-1577.06445,855.895996), (-2616.21289,-2995.22119)], [(120.596695,36.3307533), (256.678223,-62.539566), (-11103.1318,16349.6396), (2817.04761,4783.71631), (5541.82617,-3055.08032), (-12323.7041,23456.2188), (3080.56445,6823.78125), (-599.329651,-5047.74658), (-4948.23682,-7576.99365)], [(405.802826,830.723937), (13.8476791,-254.501434), (215351.813,89196.164), (-22843.6504,5955.60986), (-5143.18604,-2432.47046), (-37133.332,35995.6563), (25196.6621,-9729.71972), (-7678.30419,-19585.4844), (8.921650e+03,18069.1641)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @cumprod(%arg0: tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<4x9xcomplex<f32>>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %5 = stablehlo.multiply %3, %4 : tensor<2x9xcomplex<f32>>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %8 = stablehlo.multiply %6, %7 : tensor<1x9xcomplex<f32>>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %11 = stablehlo.multiply %9, %10 : tensor<0x9xcomplex<f32>>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<0x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %14 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %16 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %18 = stablehlo.add %15, %17 : tensor<2x9xcomplex<f32>>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %21 = stablehlo.multiply %19, %20 : tensor<1x9xcomplex<f32>>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<1x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %24 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %26 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %28 = stablehlo.add %25, %27 : tensor<4x9xcomplex<f32>>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %31 = stablehlo.multiply %29, %30 : tensor<3x9xcomplex<f32>>
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
