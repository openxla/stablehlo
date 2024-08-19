// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xcomplex<f64>>
    %1 = call @expected() : () -> tensor<5x7xcomplex<f64>>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>):
      %3 = stablehlo.real %arg0 : (tensor<complex<f64>>) -> tensor<f64>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %4 = stablehlo.compare  EQ, %3, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %5 = stablehlo.select %4, %cst_0, %3 : tensor<i1>, tensor<f64>
      %6 = stablehlo.compare  NE, %3, %3,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
      %7 = stablehlo.select %6, %cst_1, %5 : tensor<i1>, tensor<f64>
      %8 = stablehlo.imag %arg0 : (tensor<complex<f64>>) -> tensor<f64>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %9 = stablehlo.compare  EQ, %8, %cst_2,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %10 = stablehlo.select %9, %cst_3, %8 : tensor<i1>, tensor<f64>
      %11 = stablehlo.compare  NE, %8, %8,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
      %12 = stablehlo.select %11, %cst_4, %10 : tensor<i1>, tensor<f64>
      %13 = stablehlo.real %arg1 : (tensor<complex<f64>>) -> tensor<f64>
      %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %14 = stablehlo.compare  EQ, %13, %cst_5,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %15 = stablehlo.select %14, %cst_6, %13 : tensor<i1>, tensor<f64>
      %16 = stablehlo.compare  NE, %13, %13,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_7 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
      %17 = stablehlo.select %16, %cst_7, %15 : tensor<i1>, tensor<f64>
      %18 = stablehlo.imag %arg1 : (tensor<complex<f64>>) -> tensor<f64>
      %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %19 = stablehlo.compare  EQ, %18, %cst_8,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %20 = stablehlo.select %19, %cst_9, %18 : tensor<i1>, tensor<f64>
      %21 = stablehlo.compare  NE, %18, %18,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_10 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
      %22 = stablehlo.select %21, %cst_10, %20 : tensor<i1>, tensor<f64>
      %23 = stablehlo.compare  LT, %12, %22,  TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %24 = stablehlo.compare  LT, %7, %17,  TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %25 = stablehlo.compare  EQ, %7, %17,  TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %26 = stablehlo.and %25, %23 : tensor<i1>
      %27 = stablehlo.or %24, %26 : tensor<i1>
      stablehlo.return %27 : tensor<i1>
    }) : (tensor<5x7xcomplex<f64>>) -> tensor<5x7xcomplex<f64>>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xcomplex<f64>>, tensor<5x7xcomplex<f64>>) -> ()
    return %2 : tensor<5x7xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<5x7xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.64944264624640258,1.8644958567225149), (-1.1994910003008084,-2.3947153534106054), (0.87678242649798709,3.4186975878474462), (-4.941559697600062,3.3040654072929283), (1.7344473357004015,1.5114585492183048), (3.9110869865337605,-6.5450301921577632), (-0.24921953991846346,1.0605301430490701)], [(-0.39456495128517594,2.0464516580769061), (-0.60532504429574174,0.14739522385347475), (4.7120907212489795,-3.7016351244260384), (0.30677575551742364,-3.5517602003857069), (4.7934713839026593,2.2411522548710421), (0.73813718680452056,-7.5334013751630948), (-0.74477976131012891,-3.2448148397921441)], [(4.4030828729636671,1.8617995584501481), (1.4776385552561986,-0.12614478725582773), (0.31361908711910358,0.85138624510558436), (0.55218300919771535,-0.88608320277690233), (4.9668445386980427,1.132838136625913), (-3.6894334284890551,-2.3681588926945638), (4.1082916491323447,1.5775315939191725)], [(0.69313580871902969,-2.1876246473781782), (-5.3367303481182207,-2.3690572089334503), (-6.5735594674279181,-5.4145718413220916), (2.3084125663360737,-0.42271919316078965), (-2.9888835466028341,-1.1330456020838757), (-0.20012046711908324,-1.4094380287854835), (-1.188100880018089,-3.5018452039490056)], [(3.1605309513611379,-0.18253662964002815), (-0.94389012989099152,4.1663781255558483), (1.0883816404967923,1.920608285851281), (-5.2918227675522331,0.8041070865420803), (1.5068728813231811,-0.68677102184695249), (0.49901319800208671,-1.5155025860734601), (1.4989977944138713,2.6472286983937945)]]> : tensor<5x7xcomplex<f64>>
    return %cst : tensor<5x7xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<5x7xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.39456495128517594,2.0464516580769061), (-5.3367303481182207,-2.3690572089334503), (-6.5735594674279181,-5.4145718413220916), (-5.2918227675522331,0.8041070865420803), (-2.9888835466028341,-1.1330456020838757), (-3.6894334284890551,-2.3681588926945638), (-1.188100880018089,-3.5018452039490056)], [(0.64944264624640258,1.8644958567225149), (-1.1994910003008084,-2.3947153534106054), (0.31361908711910358,0.85138624510558436), (-4.941559697600062,3.3040654072929283), (1.5068728813231811,-0.68677102184695249), (-0.20012046711908324,-1.4094380287854835), (-0.74477976131012891,-3.2448148397921441)], [(0.69313580871902969,-2.1876246473781782), (-0.94389012989099152,4.1663781255558483), (0.87678242649798709,3.4186975878474462), (0.30677575551742364,-3.5517602003857069), (1.7344473357004015,1.5114585492183048), (0.49901319800208671,-1.5155025860734601), (-0.24921953991846346,1.0605301430490701)], [(3.1605309513611379,-0.18253662964002815), (-0.60532504429574174,0.14739522385347475), (1.0883816404967923,1.920608285851281), (0.55218300919771535,-0.88608320277690233), (4.7934713839026593,2.2411522548710421), (0.73813718680452056,-7.5334013751630948), (1.4989977944138713,2.6472286983937945)], [(4.4030828729636671,1.8617995584501481), (1.4776385552561986,-0.12614478725582773), (4.7120907212489795,-3.7016351244260384), (2.3084125663360737,-0.42271919316078965), (4.9668445386980427,1.132838136625913), (3.9110869865337605,-6.5450301921577632), (4.1082916491323447,1.5775315939191725)]]> : tensor<5x7xcomplex<f64>>
    return %cst : tensor<5x7xcomplex<f64>>
  }
}
