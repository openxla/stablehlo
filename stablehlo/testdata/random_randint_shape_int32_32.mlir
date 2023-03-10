// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<32xi32>
    %2 = stablehlo.constant dense<2147483647> : tensor<i32>
    %3 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %4 = stablehlo.constant dense<2147483647> : tensor<i32>
    %5 = call @clip(%2, %3, %4) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<5> : tensor<i32>
    %7 = stablehlo.compare  GT, %6, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = stablehlo.constant dense<-5> : tensor<i32>
    %9 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %10 = stablehlo.constant dense<2147483647> : tensor<i32>
    %11 = call @clip_0(%8, %9, %10) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = stablehlo.convert %11 : tensor<i32>
    %13 = stablehlo.constant dense<5> : tensor<i32>
    %14 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %15 = stablehlo.constant dense<2147483647> : tensor<i32>
    %16 = call @clip_1(%13, %14, %15) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %17 = stablehlo.convert %16 : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %19 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.iota dim = 0 : tensor<4xui32>
    %21 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %24 = stablehlo.reshape %23 : (tensor<1xui32>) -> tensor<ui32>
    %25 = "stablehlo.slice"(%20) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %26 = "stablehlo.slice"(%20) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %27 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %28 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %29 = stablehlo.xor %22, %24 : tensor<ui32>
    %30 = stablehlo.constant dense<466688986> : tensor<ui32>
    %31 = stablehlo.xor %29, %30 : tensor<ui32>
    %32 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %33 = stablehlo.add %25, %32 : tensor<2xui32>
    %34 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %35 = stablehlo.add %26, %34 : tensor<2xui32>
    %36 = stablehlo.constant dense<0> : tensor<i32>
    %37 = stablehlo.constant dense<0> : tensor<i32>
    %38:9 = stablehlo.while(%iterArg = %37, %iterArg_0 = %36, %iterArg_1 = %33, %iterArg_2 = %35, %iterArg_3 = %24, %iterArg_4 = %31, %iterArg_5 = %22, %iterArg_6 = %27, %iterArg_7 = %28) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %116 = stablehlo.constant dense<5> : tensor<i32>
      %117 = stablehlo.compare  LT, %iterArg, %116,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %117 : tensor<i1>
    } do {
      %116 = stablehlo.constant dense<1> : tensor<i32>
      %117 = stablehlo.add %iterArg_0, %116 : tensor<i32>
      %118 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %119 = stablehlo.reshape %118 : (tensor<1xui32>) -> tensor<ui32>
      %120 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %121 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %122 = stablehlo.shift_left %iterArg_2, %121 : tensor<2xui32>
      %123 = stablehlo.constant dense<32> : tensor<ui32>
      %124 = stablehlo.subtract %123, %119 : tensor<ui32>
      %125 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %126 = stablehlo.shift_right_logical %iterArg_2, %125 : tensor<2xui32>
      %127 = stablehlo.or %122, %126 : tensor<2xui32>
      %128 = stablehlo.xor %120, %127 : tensor<2xui32>
      %129 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %130 = stablehlo.reshape %129 : (tensor<1xui32>) -> tensor<ui32>
      %131 = stablehlo.add %120, %128 : tensor<2xui32>
      %132 = stablehlo.broadcast_in_dim %130, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %133 = stablehlo.shift_left %128, %132 : tensor<2xui32>
      %134 = stablehlo.constant dense<32> : tensor<ui32>
      %135 = stablehlo.subtract %134, %130 : tensor<ui32>
      %136 = stablehlo.broadcast_in_dim %135, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %137 = stablehlo.shift_right_logical %128, %136 : tensor<2xui32>
      %138 = stablehlo.or %133, %137 : tensor<2xui32>
      %139 = stablehlo.xor %131, %138 : tensor<2xui32>
      %140 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %141 = stablehlo.reshape %140 : (tensor<1xui32>) -> tensor<ui32>
      %142 = stablehlo.add %131, %139 : tensor<2xui32>
      %143 = stablehlo.broadcast_in_dim %141, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %144 = stablehlo.shift_left %139, %143 : tensor<2xui32>
      %145 = stablehlo.constant dense<32> : tensor<ui32>
      %146 = stablehlo.subtract %145, %141 : tensor<ui32>
      %147 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %148 = stablehlo.shift_right_logical %139, %147 : tensor<2xui32>
      %149 = stablehlo.or %144, %148 : tensor<2xui32>
      %150 = stablehlo.xor %142, %149 : tensor<2xui32>
      %151 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %152 = stablehlo.reshape %151 : (tensor<1xui32>) -> tensor<ui32>
      %153 = stablehlo.add %142, %150 : tensor<2xui32>
      %154 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %155 = stablehlo.shift_left %150, %154 : tensor<2xui32>
      %156 = stablehlo.constant dense<32> : tensor<ui32>
      %157 = stablehlo.subtract %156, %152 : tensor<ui32>
      %158 = stablehlo.broadcast_in_dim %157, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %159 = stablehlo.shift_right_logical %150, %158 : tensor<2xui32>
      %160 = stablehlo.or %155, %159 : tensor<2xui32>
      %161 = stablehlo.xor %153, %160 : tensor<2xui32>
      %162 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %163 = stablehlo.add %153, %162 : tensor<2xui32>
      %164 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %165 = stablehlo.add %161, %164 : tensor<2xui32>
      %166 = stablehlo.constant dense<1> : tensor<i32>
      %167 = stablehlo.add %iterArg_0, %166 : tensor<i32>
      %168 = stablehlo.convert %167 : (tensor<i32>) -> tensor<ui32>
      %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %170 = stablehlo.add %165, %169 : tensor<2xui32>
      %171 = stablehlo.constant dense<1> : tensor<i32>
      %172 = stablehlo.add %iterArg, %171 : tensor<i32>
      stablehlo.return %172, %117, %163, %170, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %39 = stablehlo.concatenate %38#2, %38#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %40 = stablehlo.reshape %39 : (tensor<4xui32>) -> tensor<2x2xui32>
    %41 = "stablehlo.slice"(%40) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %42 = stablehlo.reshape %41 : (tensor<1x2xui32>) -> tensor<2xui32>
    %43 = "stablehlo.slice"(%40) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %44 = stablehlo.reshape %43 : (tensor<1x2xui32>) -> tensor<2xui32>
    %45 = stablehlo.iota dim = 0 : tensor<32xui32>
    %46 = "stablehlo.slice"(%42) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %47 = stablehlo.reshape %46 : (tensor<1xui32>) -> tensor<ui32>
    %48 = "stablehlo.slice"(%42) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %49 = stablehlo.reshape %48 : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.slice"(%45) {limit_indices = dense<16> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<32xui32>) -> tensor<16xui32>
    %51 = "stablehlo.slice"(%45) {limit_indices = dense<32> : tensor<1xi64>, start_indices = dense<16> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<32xui32>) -> tensor<16xui32>
    %52 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %53 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %54 = stablehlo.xor %47, %49 : tensor<ui32>
    %55 = stablehlo.constant dense<466688986> : tensor<ui32>
    %56 = stablehlo.xor %54, %55 : tensor<ui32>
    %57 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<ui32>) -> tensor<16xui32>
    %58 = stablehlo.add %50, %57 : tensor<16xui32>
    %59 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<ui32>) -> tensor<16xui32>
    %60 = stablehlo.add %51, %59 : tensor<16xui32>
    %61 = stablehlo.constant dense<0> : tensor<i32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63:9 = stablehlo.while(%iterArg = %62, %iterArg_0 = %61, %iterArg_1 = %58, %iterArg_2 = %60, %iterArg_3 = %49, %iterArg_4 = %56, %iterArg_5 = %47, %iterArg_6 = %52, %iterArg_7 = %53) : tensor<i32>, tensor<i32>, tensor<16xui32>, tensor<16xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %116 = stablehlo.constant dense<5> : tensor<i32>
      %117 = stablehlo.compare  LT, %iterArg, %116,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %117 : tensor<i1>
    } do {
      %116 = stablehlo.constant dense<1> : tensor<i32>
      %117 = stablehlo.add %iterArg_0, %116 : tensor<i32>
      %118 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %119 = stablehlo.reshape %118 : (tensor<1xui32>) -> tensor<ui32>
      %120 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<16xui32>
      %121 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %122 = stablehlo.shift_left %iterArg_2, %121 : tensor<16xui32>
      %123 = stablehlo.constant dense<32> : tensor<ui32>
      %124 = stablehlo.subtract %123, %119 : tensor<ui32>
      %125 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %126 = stablehlo.shift_right_logical %iterArg_2, %125 : tensor<16xui32>
      %127 = stablehlo.or %122, %126 : tensor<16xui32>
      %128 = stablehlo.xor %120, %127 : tensor<16xui32>
      %129 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %130 = stablehlo.reshape %129 : (tensor<1xui32>) -> tensor<ui32>
      %131 = stablehlo.add %120, %128 : tensor<16xui32>
      %132 = stablehlo.broadcast_in_dim %130, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %133 = stablehlo.shift_left %128, %132 : tensor<16xui32>
      %134 = stablehlo.constant dense<32> : tensor<ui32>
      %135 = stablehlo.subtract %134, %130 : tensor<ui32>
      %136 = stablehlo.broadcast_in_dim %135, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %137 = stablehlo.shift_right_logical %128, %136 : tensor<16xui32>
      %138 = stablehlo.or %133, %137 : tensor<16xui32>
      %139 = stablehlo.xor %131, %138 : tensor<16xui32>
      %140 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %141 = stablehlo.reshape %140 : (tensor<1xui32>) -> tensor<ui32>
      %142 = stablehlo.add %131, %139 : tensor<16xui32>
      %143 = stablehlo.broadcast_in_dim %141, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %144 = stablehlo.shift_left %139, %143 : tensor<16xui32>
      %145 = stablehlo.constant dense<32> : tensor<ui32>
      %146 = stablehlo.subtract %145, %141 : tensor<ui32>
      %147 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %148 = stablehlo.shift_right_logical %139, %147 : tensor<16xui32>
      %149 = stablehlo.or %144, %148 : tensor<16xui32>
      %150 = stablehlo.xor %142, %149 : tensor<16xui32>
      %151 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %152 = stablehlo.reshape %151 : (tensor<1xui32>) -> tensor<ui32>
      %153 = stablehlo.add %142, %150 : tensor<16xui32>
      %154 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %155 = stablehlo.shift_left %150, %154 : tensor<16xui32>
      %156 = stablehlo.constant dense<32> : tensor<ui32>
      %157 = stablehlo.subtract %156, %152 : tensor<ui32>
      %158 = stablehlo.broadcast_in_dim %157, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %159 = stablehlo.shift_right_logical %150, %158 : tensor<16xui32>
      %160 = stablehlo.or %155, %159 : tensor<16xui32>
      %161 = stablehlo.xor %153, %160 : tensor<16xui32>
      %162 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %163 = stablehlo.add %153, %162 : tensor<16xui32>
      %164 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %165 = stablehlo.add %161, %164 : tensor<16xui32>
      %166 = stablehlo.constant dense<1> : tensor<i32>
      %167 = stablehlo.add %iterArg_0, %166 : tensor<i32>
      %168 = stablehlo.convert %167 : (tensor<i32>) -> tensor<ui32>
      %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %170 = stablehlo.add %165, %169 : tensor<16xui32>
      %171 = stablehlo.constant dense<1> : tensor<i32>
      %172 = stablehlo.add %iterArg, %171 : tensor<i32>
      stablehlo.return %172, %117, %163, %170, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<16xui32>, tensor<16xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %64 = stablehlo.concatenate %63#2, %63#3, dim = 0 : (tensor<16xui32>, tensor<16xui32>) -> tensor<32xui32>
    %65 = stablehlo.iota dim = 0 : tensor<32xui32>
    %66 = "stablehlo.slice"(%44) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %67 = stablehlo.reshape %66 : (tensor<1xui32>) -> tensor<ui32>
    %68 = "stablehlo.slice"(%44) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %69 = stablehlo.reshape %68 : (tensor<1xui32>) -> tensor<ui32>
    %70 = "stablehlo.slice"(%65) {limit_indices = dense<16> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<32xui32>) -> tensor<16xui32>
    %71 = "stablehlo.slice"(%65) {limit_indices = dense<32> : tensor<1xi64>, start_indices = dense<16> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<32xui32>) -> tensor<16xui32>
    %72 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %73 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %74 = stablehlo.xor %67, %69 : tensor<ui32>
    %75 = stablehlo.constant dense<466688986> : tensor<ui32>
    %76 = stablehlo.xor %74, %75 : tensor<ui32>
    %77 = stablehlo.broadcast_in_dim %67, dims = [] : (tensor<ui32>) -> tensor<16xui32>
    %78 = stablehlo.add %70, %77 : tensor<16xui32>
    %79 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<ui32>) -> tensor<16xui32>
    %80 = stablehlo.add %71, %79 : tensor<16xui32>
    %81 = stablehlo.constant dense<0> : tensor<i32>
    %82 = stablehlo.constant dense<0> : tensor<i32>
    %83:9 = stablehlo.while(%iterArg = %82, %iterArg_0 = %81, %iterArg_1 = %78, %iterArg_2 = %80, %iterArg_3 = %69, %iterArg_4 = %76, %iterArg_5 = %67, %iterArg_6 = %72, %iterArg_7 = %73) : tensor<i32>, tensor<i32>, tensor<16xui32>, tensor<16xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %116 = stablehlo.constant dense<5> : tensor<i32>
      %117 = stablehlo.compare  LT, %iterArg, %116,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %117 : tensor<i1>
    } do {
      %116 = stablehlo.constant dense<1> : tensor<i32>
      %117 = stablehlo.add %iterArg_0, %116 : tensor<i32>
      %118 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %119 = stablehlo.reshape %118 : (tensor<1xui32>) -> tensor<ui32>
      %120 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<16xui32>
      %121 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %122 = stablehlo.shift_left %iterArg_2, %121 : tensor<16xui32>
      %123 = stablehlo.constant dense<32> : tensor<ui32>
      %124 = stablehlo.subtract %123, %119 : tensor<ui32>
      %125 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %126 = stablehlo.shift_right_logical %iterArg_2, %125 : tensor<16xui32>
      %127 = stablehlo.or %122, %126 : tensor<16xui32>
      %128 = stablehlo.xor %120, %127 : tensor<16xui32>
      %129 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %130 = stablehlo.reshape %129 : (tensor<1xui32>) -> tensor<ui32>
      %131 = stablehlo.add %120, %128 : tensor<16xui32>
      %132 = stablehlo.broadcast_in_dim %130, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %133 = stablehlo.shift_left %128, %132 : tensor<16xui32>
      %134 = stablehlo.constant dense<32> : tensor<ui32>
      %135 = stablehlo.subtract %134, %130 : tensor<ui32>
      %136 = stablehlo.broadcast_in_dim %135, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %137 = stablehlo.shift_right_logical %128, %136 : tensor<16xui32>
      %138 = stablehlo.or %133, %137 : tensor<16xui32>
      %139 = stablehlo.xor %131, %138 : tensor<16xui32>
      %140 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %141 = stablehlo.reshape %140 : (tensor<1xui32>) -> tensor<ui32>
      %142 = stablehlo.add %131, %139 : tensor<16xui32>
      %143 = stablehlo.broadcast_in_dim %141, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %144 = stablehlo.shift_left %139, %143 : tensor<16xui32>
      %145 = stablehlo.constant dense<32> : tensor<ui32>
      %146 = stablehlo.subtract %145, %141 : tensor<ui32>
      %147 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %148 = stablehlo.shift_right_logical %139, %147 : tensor<16xui32>
      %149 = stablehlo.or %144, %148 : tensor<16xui32>
      %150 = stablehlo.xor %142, %149 : tensor<16xui32>
      %151 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %152 = stablehlo.reshape %151 : (tensor<1xui32>) -> tensor<ui32>
      %153 = stablehlo.add %142, %150 : tensor<16xui32>
      %154 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %155 = stablehlo.shift_left %150, %154 : tensor<16xui32>
      %156 = stablehlo.constant dense<32> : tensor<ui32>
      %157 = stablehlo.subtract %156, %152 : tensor<ui32>
      %158 = stablehlo.broadcast_in_dim %157, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %159 = stablehlo.shift_right_logical %150, %158 : tensor<16xui32>
      %160 = stablehlo.or %155, %159 : tensor<16xui32>
      %161 = stablehlo.xor %153, %160 : tensor<16xui32>
      %162 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %163 = stablehlo.add %153, %162 : tensor<16xui32>
      %164 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %165 = stablehlo.add %161, %164 : tensor<16xui32>
      %166 = stablehlo.constant dense<1> : tensor<i32>
      %167 = stablehlo.add %iterArg_0, %166 : tensor<i32>
      %168 = stablehlo.convert %167 : (tensor<i32>) -> tensor<ui32>
      %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %170 = stablehlo.add %165, %169 : tensor<16xui32>
      %171 = stablehlo.constant dense<1> : tensor<i32>
      %172 = stablehlo.add %iterArg, %171 : tensor<i32>
      stablehlo.return %172, %117, %163, %170, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<16xui32>, tensor<16xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %84 = stablehlo.concatenate %83#2, %83#3, dim = 0 : (tensor<16xui32>, tensor<16xui32>) -> tensor<32xui32>
    %85 = stablehlo.subtract %19, %18 : tensor<1xi32>
    %86 = stablehlo.convert %85 : (tensor<1xi32>) -> tensor<1xui32>
    %87 = stablehlo.compare  LE, %19, %18,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %88 = stablehlo.constant dense<1> : tensor<ui32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %90 = stablehlo.select %87, %89, %86 : tensor<1xi1>, tensor<1xui32>
    %91 = stablehlo.compare  GT, %19, %18,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %92 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %93 = stablehlo.and %92, %91 : tensor<1xi1>
    %94 = stablehlo.constant dense<1> : tensor<ui32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %96 = stablehlo.add %90, %95 : tensor<1xui32>
    %97 = stablehlo.select %93, %96, %90 : tensor<1xi1>, tensor<1xui32>
    %98 = stablehlo.constant dense<65536> : tensor<ui32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %100 = stablehlo.remainder %99, %97 : tensor<1xui32>
    %101 = stablehlo.multiply %100, %100 : tensor<1xui32>
    %102 = stablehlo.remainder %101, %97 : tensor<1xui32>
    %103 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<1xui32>) -> tensor<32xui32>
    %104 = stablehlo.remainder %64, %103 : tensor<32xui32>
    %105 = stablehlo.broadcast_in_dim %102, dims = [0] : (tensor<1xui32>) -> tensor<32xui32>
    %106 = stablehlo.multiply %104, %105 : tensor<32xui32>
    %107 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<1xui32>) -> tensor<32xui32>
    %108 = stablehlo.remainder %84, %107 : tensor<32xui32>
    %109 = stablehlo.add %106, %108 : tensor<32xui32>
    %110 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<1xui32>) -> tensor<32xui32>
    %111 = stablehlo.remainder %109, %110 : tensor<32xui32>
    %112 = stablehlo.convert %111 : (tensor<32xui32>) -> tensor<32xi32>
    %113 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi32>) -> tensor<32xi32>
    %114 = stablehlo.add %113, %112 : tensor<32xi32>
    %115 = stablehlo.custom_call @check.eq(%114, %1) : (tensor<32xi32>, tensor<32xi32>) -> tensor<i1>
    return %115 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<32xi32> {
    %0 = stablehlo.constant dense<[-5, 1, 3, 0, -3, 0, -3, 1, 2, 4, -4, -3, 2, -2, 4, 3, 0, -5, -4, 1, 4, -2, 2, 1, 0, 3, -2, -2, -5, 2, 3, -4]> : tensor<32xi32>
    return %0 : tensor<32xi32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
  func.func private @clip_0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
  func.func private @clip_1(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
}
