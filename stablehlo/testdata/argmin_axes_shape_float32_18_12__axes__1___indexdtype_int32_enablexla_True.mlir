// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<18x12xf32>
    %1 = call @expected() : () -> tensor<18xi32>
    %2 = call @argmin(%0) : (tensor<18x12xf32>) -> tensor<18xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<18xi32>, tensor<18xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<18x12xf32> {
    %0 = stablehlo.constant dense<"0x2A66C03EDE0C103F33FABA3F781A8CBFAF62C0C0F291A140F52D0FBF7A635ABE2D7400C10C839DBD77D777BF9A3B1C3F8ED8B8BF790F0C3F3200C1BED15D7B40D5A711C0B4742540E20507C022BDAC3FAC3E8F4077ACD13F3A448ABF7E33A7C0BD2E1DBEB6DA4D3F22A688BFCDB017BED0B449C02500293DB672D8BE62866BC082A22040C12BAF3D93AA2C3F6A14DABFA8ECCFC0289493BF0F405CBFAD6C41403D506B40AD9185BF7FF4C13F50291DC0DE663C3F9864334072EC09C07A5C3B40F368F2BFE6D82AC09C8242C0C6AD81C0B22D4BBF1E6AB63F9B18B43E343133C080A9124078AD9A3FA366953F0A24263FF9A8933F2A49FBBFCB102EBDEF312840A4E98BBFE9CF6340C35FC63F52795EBF00BAEF3F01664FC094290A40B7874DC016D6633FC2215C400F2BE5BFB9B0E73E344A36C025DF01C0B362FFBF89B706409C2B063F846D2EC0AE03433E5A79B3BDDCED503FF5B6593D4DEA9CC0EBD0CABF561B3040A4C58ABF7D584340CE3906C052BA673FE206E73E85FA294064E3D43FF26C45BCE74F2FC02BEF12C130C5DDC0BA29453EB0222E40866811403A0414C04076E5BF315191C0B1B0EE408413AAC049BEEE3F2AF88440D73F91BED87023C005F42BBE429A13BF405E7BC06330BC3FD3AD853EFC36AD407B70284035A75D4004A32BBFDFECA14059C697C08A08E53FBF7BA4C0F0E29B3F5529CFBB35922AC15F75A3C0604CE6BF884D85C0C623AB4018267F40C14F6A40DAC1EBBE992836C0278FA73F872F1140CD77113DCA70CF40A73543C0F8C2BDC06440C93E08C56340FC3A89C09A0F1440708E3E3F9A70A7BEEAE170407E9C12BF27333740594C714069555CBFBC5068404F7D3CC0FCED4440AFD69A400CB7B03EC490013FE58247C06DAB1D403A5199403FF4FD3F48FC8D40A25FD73F37699A4041880040B43F223F330E123FC03C9A3F29751A40BA02A63F51C631C09E665CC08B0D1B407BE29C3FFF5E534046F319C0C266A2C0B1CD303F2BE814C0D7D29FC0FEEA47C0347B84C09C96C0BF715196C0B1CD633F0BB6BF3F5C0FFFBF311B68C0E988BDBECBA8E7BE5B3FB4C054D17E40E56BCAC03F380BC0A955BCBF28BDBBBF1CEE56BE11BD3DC0587645C06D744BBF6D82B9C0953C8A3DAF9E6D40EDF897BE462825BF50CF9A3FA5DD6CC0AB211AC0499912C0AAAF2CC01971353F261B9440598121C084E3B0BF"> : tensor<18x12xf32>
    return %0 : tensor<18x12xf32>
  }
  func.func private @expected() -> tensor<18xi32> {
    %0 = stablehlo.constant dense<[8, 11, 7, 0, 3, 9, 4, 2, 2, 6, 7, 9, 0, 3, 10, 1, 2, 4]> : tensor<18xi32>
    return %0 : tensor<18xi32>
  }
  func.func private @argmin(%arg0: tensor<18x12xf32>) -> tensor<18xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<18x12xi32>
    %1 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [1] : (tensor<18x12xf32>, tensor<18x12xi32>, tensor<f32>, tensor<i32>) -> (tensor<18xf32>, tensor<18xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  LT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<f32>, tensor<i32>
    }
    return %3#1 : tensor<18xi32>
  }
}
