// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-3.28254175,1.17971408), (4.88530779,-0.133065134), (6.09067678,-0.695391059)], [(-0.452481329,-1.43866086), (5.69688702,-4.74869299), (-0.735809385,-0.741267144)], [(-4.63528156,-2.12715411), (4.72008276,-1.30464411), (0.0831500887,2.1310811)], [(-3.05067658,-3.7326386), (3.1067102,2.63032556), (1.07599819,1.06427312)]]> : tensor<4x3xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(2.04384804,2.52163649), (1.34624648,-2.58588719), (3.17275858,-0.135297522), (4.8619442,-3.1084168), (-0.3081882,5.67325211), (0.651394546,0.400263518)], [(1.5483948,-1.90760767), (-0.1668313,-4.15355349), (-3.54817295,3.27089429), (3.70867682,2.1964066), (4.23657036,-0.011417916), (3.80263972,-0.222120628)], [(-1.42191422,-1.96880734), (-3.10092974,-0.730867445), (1.58704257,-2.82402587), (-2.18051314,-6.26070308), (-4.60540342,6.75190783), (3.14610624,0.480063885)]]> : tensor<3x6xcomplex<f32>>
    return %0, %1 : tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-12.4027882,-26.3940926), (-22.131218,-12.4878387), (-19.4514141,2.3346858), (-11.5166359,-10.4397697), (-8.34055137,24.7204628), (35.4328346,-1.40040898)], [(2.05220938,-19.7989883), (-23.2638092,-20.8004246), (-9.57244682,31.8813171), (21.8496647,-4.46384811), (40.7760468,-23.8611641), (18.9304466,-23.1265545)], [(4.78735685,-30.2541656), (-16.6474838,-16.9339237), (-21.3245544,17.0934887), (4.38298607,4.42757034), (18.7066841,-40.4757347), (14.7295551,-2.50594521)], [(13.570653,-20.8069839), (-5.91099643,-14.5657024), (-25.0975494,-11.9507418), (-16.3734264,-1.14364624), (23.1669312,-2.68518162), (14.7790794,9.52448463)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
}

