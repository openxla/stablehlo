// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %2 = call @expected() : () -> tensor<1x125xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x3FBDD84189C3E1B45AC53E3D44B97942063CE4B34BC4463A74BFDFC13EC19742ACBA9F3B26C1DDB74F3BCAC5E7C0883EF4309AC45D3EC0BED7BCA63FEDC8B13C444527BA3A42233323C315B32AC8A5457441B3BEB134EFBBC3BC9C3D21BD4FC0B33FC3450C441D3C1BBD7EB88F45FCC46FC451BC904459C3CB348945823F693CA431BDBC05C3FFB31CC0C9C43043853C94BF82B4FABE67C1AA456A442EC3D140484389B9DB393FC0E2B6F4284240AC44F444D0C14EC28345F3C252B9FDBE23C253B808AF24BC22BCAE3C67BDF6BBFE43BC412E371FBEA03A64406DC60841C54517C045C44B3CBB3A77AEFFBD8BC0C93C2544583A2AC435409DC4"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<4.394530e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0xC3C5D84189C3E1B45AC53E3D44B97942063CE4B34BC4463A74BFDFC13EC19742ACBA9F3B26C1DDB74F3BCAC5E7C0883EF4309AC45D3EC0BED7BCA63FEDC8B13C444527BA3A42233323C315B32AC8A5457441B3BEB134EFBBC3BC9C3D21BD4FC0B33FC3450C441D3C1BBD7EB88F45FCC46FC451BC904459C3CB348945823F693CA431BDBC05C3FFB31CC0C9C43043853C94BF82B4FABE67C1AA456A442EC3D140484389B9DB393FC0E2B6F4284240AC44F444D0C14EC28345F3C252B9FDBE23C253B808AF24BC22BCAE3C67BDF6BBFE43BC412E371FBEA03A64406DC60841C54517C045C44B3CBB3A77AEFFBD8BC0C93C2544583A2AC435409DC4"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

