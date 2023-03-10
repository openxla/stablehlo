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
    %0 = stablehlo.constant dense<[[(-4.08892918,-0.592274547), (-1.14425647,1.91067052), (-2.25763345,-3.02269602), (2.39991069,3.9459095), (-5.70192957,6.74494171), (3.16905332,2.53717709), (-0.501384735,-1.49223828), (0.884104967,1.31896174), (2.07389832,-0.408055216)], [(-0.343131691,-0.398325592), (-1.91208398,-2.82888699), (8.2058382,1.52863693), (-0.477989256,-7.1356473), (-1.14795923,0.559394419), (-4.9344492,-2.6751194), (0.510269761,-3.59979939), (-1.61603713,-3.29146242), (2.49449039,3.68833184)], [(-1.63475573,4.34523106), (2.52415752,2.52692461), (-0.774396658,-4.10873556), (0.575948417,1.05766678), (0.160311162,2.86548877), (0.429974169,3.20121646), (1.440140e-01,1.09291257E-4), (1.15421069,-4.29048109), (3.39356613,4.07246542)], [(-0.660921037,-1.60053229), (-0.274453551,-5.38995028), (0.404237062,6.81706523), (1.56395948,0.457769901), (1.67588067,-0.903431117), (1.11237192,3.51121426), (-3.54644537,3.66391945), (-0.254782468,-2.78542209), (0.0334065929,2.80789161)], [(3.10637617,-2.24091625), (-0.431779623,4.30348301), (2.46626496,2.08280706), (2.52046442,3.98609471), (2.94492626,0.59603548), (-1.85589445,-4.35431337), (-2.61496735,-0.78628987), (-1.44992769,3.24667931), (-2.70643377,1.74269211)], [(-0.993211925,5.17186069), (-4.39847517,-6.65513706), (-2.0101757,-1.64909577), (2.0322082,2.44933963), (2.47400641,1.90167332), (6.2922039,-2.12090564), (0.236437574,-2.48392987), (-2.36763692,2.49184537), (-1.87151659,4.814810e+00)], [(-0.171422139,3.31177139), (2.07540035,0.605110884), (1.95731556,0.897971212), (-0.131062135,5.34874916), (2.79879117,-1.79001343), (2.25746107,-2.21167707), (1.91943324,0.201414719), (1.4620018,-0.718965947), (2.62279963,2.66023421)], [(-1.58661473,-0.352841526), (0.336428523,4.36195803), (0.40680027,-0.615227162), (-4.74887371,-3.15494657), (-3.56358314,-1.41914356), (6.91667128,1.14076173), (-2.20926857,-2.43210053), (-2.70864248,-2.36884451), (-0.441500217,0.0555461198)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-4.08892918,-0.592274547), (-1.14425647,1.91067052), (-2.25763345,-3.02269602), (2.39991069,3.9459095), (-5.70192957,6.74494171), (3.16905332,2.53717709), (-0.501384735,-1.49223828), (0.884104967,1.31896174), (2.07389832,-0.408055216)], [(-4.43206072,-0.990600109), (-3.05634046,-0.918216467), (5.94820499,-1.49405909), (1.92192149,-3.1897378), (-6.8498888,7.30433607), (-1.76539588,-0.137942314), (0.00888502597,-5.09203768), (-0.731932163,-1.97250068), (4.56838894,3.28027654)], [(-6.06681633,3.35463095), (-0.532182932,1.60870814), (5.1738081,-5.60279465), (2.497870e+00,-2.13207102), (-6.68957758,10.1698246), (-1.33542168,3.06327415), (0.152899027,-5.09192848), (0.422278523,-6.26298189), (7.96195507,7.35274219)], [(-6.72773743,1.75409865), (-0.806636572,-3.78124213), (5.57804537,1.21427059), (4.06182957,-1.67430115), (-5.01369667,9.26639366), (-0.22304976,6.57448863), (-3.39354658,-1.42800903), (0.167496085,-9.04840373), (7.99536133,10.1606331)], [(-3.62136126,-0.486817598), (-1.23841619,0.522240877), (8.04431056,3.29707766), (6.58229399,2.31179357), (-2.06877041,9.86242866), (-2.07894421,2.22017527), (-6.00851393,-2.21429896), (-1.2824316,-5.80172443), (5.28892756,11.9033251)], [(-4.61457348,4.68504333), (-5.63689137,-6.13289642), (6.03413486,1.64798188), (8.61450195,4.76113319), (0.405236244,11.7641029), (4.2132597,0.0992698669), (-5.77207661,-4.69822884), (-3.65006828,-3.3098793), (3.41741085,16.7181358)], [(-4.78599548,7.99681473), (-3.56149101,-5.5277853), (7.9914503,2.54595304), (8.48343944,10.1098824), (3.20402741,9.97408962), (6.47072077,-2.11240721), (-3.85264349,-4.49681425), (-2.18806648,-4.02884531), (6.04021072,19.3783703)], [(-6.372610e+00,7.64397335), (-3.22506213,-1.16582751), (8.39825058,1.93072593), (3.73456621,6.95493602), (-0.359555721,8.55494499), (13.3873911,-0.971645355), (-6.06191158,-6.92891455), (-4.89670897,-6.397690e+00), (5.598710e+00,19.4339142)]]> : tensor<8x9xcomplex<f32>>
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
