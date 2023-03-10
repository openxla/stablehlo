// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<5x4xi32>
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
    %18 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %19 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
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
      %118 = stablehlo.constant dense<5> : tensor<i32>
      %119 = stablehlo.compare  LT, %iterArg, %118,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %119 : tensor<i1>
    } do {
      %118 = stablehlo.constant dense<1> : tensor<i32>
      %119 = stablehlo.add %iterArg_0, %118 : tensor<i32>
      %120 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %121 = stablehlo.reshape %120 : (tensor<1xui32>) -> tensor<ui32>
      %122 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %123 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %124 = stablehlo.shift_left %iterArg_2, %123 : tensor<2xui32>
      %125 = stablehlo.constant dense<32> : tensor<ui32>
      %126 = stablehlo.subtract %125, %121 : tensor<ui32>
      %127 = stablehlo.broadcast_in_dim %126, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %128 = stablehlo.shift_right_logical %iterArg_2, %127 : tensor<2xui32>
      %129 = stablehlo.or %124, %128 : tensor<2xui32>
      %130 = stablehlo.xor %122, %129 : tensor<2xui32>
      %131 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %132 = stablehlo.reshape %131 : (tensor<1xui32>) -> tensor<ui32>
      %133 = stablehlo.add %122, %130 : tensor<2xui32>
      %134 = stablehlo.broadcast_in_dim %132, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %135 = stablehlo.shift_left %130, %134 : tensor<2xui32>
      %136 = stablehlo.constant dense<32> : tensor<ui32>
      %137 = stablehlo.subtract %136, %132 : tensor<ui32>
      %138 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %139 = stablehlo.shift_right_logical %130, %138 : tensor<2xui32>
      %140 = stablehlo.or %135, %139 : tensor<2xui32>
      %141 = stablehlo.xor %133, %140 : tensor<2xui32>
      %142 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %143 = stablehlo.reshape %142 : (tensor<1xui32>) -> tensor<ui32>
      %144 = stablehlo.add %133, %141 : tensor<2xui32>
      %145 = stablehlo.broadcast_in_dim %143, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %146 = stablehlo.shift_left %141, %145 : tensor<2xui32>
      %147 = stablehlo.constant dense<32> : tensor<ui32>
      %148 = stablehlo.subtract %147, %143 : tensor<ui32>
      %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %150 = stablehlo.shift_right_logical %141, %149 : tensor<2xui32>
      %151 = stablehlo.or %146, %150 : tensor<2xui32>
      %152 = stablehlo.xor %144, %151 : tensor<2xui32>
      %153 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %154 = stablehlo.reshape %153 : (tensor<1xui32>) -> tensor<ui32>
      %155 = stablehlo.add %144, %152 : tensor<2xui32>
      %156 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %157 = stablehlo.shift_left %152, %156 : tensor<2xui32>
      %158 = stablehlo.constant dense<32> : tensor<ui32>
      %159 = stablehlo.subtract %158, %154 : tensor<ui32>
      %160 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %161 = stablehlo.shift_right_logical %152, %160 : tensor<2xui32>
      %162 = stablehlo.or %157, %161 : tensor<2xui32>
      %163 = stablehlo.xor %155, %162 : tensor<2xui32>
      %164 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %165 = stablehlo.add %155, %164 : tensor<2xui32>
      %166 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %167 = stablehlo.add %163, %166 : tensor<2xui32>
      %168 = stablehlo.constant dense<1> : tensor<i32>
      %169 = stablehlo.add %iterArg_0, %168 : tensor<i32>
      %170 = stablehlo.convert %169 : (tensor<i32>) -> tensor<ui32>
      %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %172 = stablehlo.add %167, %171 : tensor<2xui32>
      %173 = stablehlo.constant dense<1> : tensor<i32>
      %174 = stablehlo.add %iterArg, %173 : tensor<i32>
      stablehlo.return %174, %119, %165, %172, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %39 = stablehlo.concatenate %38#2, %38#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %40 = stablehlo.reshape %39 : (tensor<4xui32>) -> tensor<2x2xui32>
    %41 = "stablehlo.slice"(%40) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %42 = stablehlo.reshape %41 : (tensor<1x2xui32>) -> tensor<2xui32>
    %43 = "stablehlo.slice"(%40) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %44 = stablehlo.reshape %43 : (tensor<1x2xui32>) -> tensor<2xui32>
    %45 = stablehlo.iota dim = 0 : tensor<20xui32>
    %46 = "stablehlo.slice"(%42) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %47 = stablehlo.reshape %46 : (tensor<1xui32>) -> tensor<ui32>
    %48 = "stablehlo.slice"(%42) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %49 = stablehlo.reshape %48 : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.slice"(%45) {limit_indices = dense<10> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<20xui32>) -> tensor<10xui32>
    %51 = "stablehlo.slice"(%45) {limit_indices = dense<20> : tensor<1xi64>, start_indices = dense<10> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<20xui32>) -> tensor<10xui32>
    %52 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %53 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %54 = stablehlo.xor %47, %49 : tensor<ui32>
    %55 = stablehlo.constant dense<466688986> : tensor<ui32>
    %56 = stablehlo.xor %54, %55 : tensor<ui32>
    %57 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %58 = stablehlo.add %50, %57 : tensor<10xui32>
    %59 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %60 = stablehlo.add %51, %59 : tensor<10xui32>
    %61 = stablehlo.constant dense<0> : tensor<i32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63:9 = stablehlo.while(%iterArg = %62, %iterArg_0 = %61, %iterArg_1 = %58, %iterArg_2 = %60, %iterArg_3 = %49, %iterArg_4 = %56, %iterArg_5 = %47, %iterArg_6 = %52, %iterArg_7 = %53) : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %118 = stablehlo.constant dense<5> : tensor<i32>
      %119 = stablehlo.compare  LT, %iterArg, %118,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %119 : tensor<i1>
    } do {
      %118 = stablehlo.constant dense<1> : tensor<i32>
      %119 = stablehlo.add %iterArg_0, %118 : tensor<i32>
      %120 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %121 = stablehlo.reshape %120 : (tensor<1xui32>) -> tensor<ui32>
      %122 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<10xui32>
      %123 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %124 = stablehlo.shift_left %iterArg_2, %123 : tensor<10xui32>
      %125 = stablehlo.constant dense<32> : tensor<ui32>
      %126 = stablehlo.subtract %125, %121 : tensor<ui32>
      %127 = stablehlo.broadcast_in_dim %126, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %128 = stablehlo.shift_right_logical %iterArg_2, %127 : tensor<10xui32>
      %129 = stablehlo.or %124, %128 : tensor<10xui32>
      %130 = stablehlo.xor %122, %129 : tensor<10xui32>
      %131 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %132 = stablehlo.reshape %131 : (tensor<1xui32>) -> tensor<ui32>
      %133 = stablehlo.add %122, %130 : tensor<10xui32>
      %134 = stablehlo.broadcast_in_dim %132, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %135 = stablehlo.shift_left %130, %134 : tensor<10xui32>
      %136 = stablehlo.constant dense<32> : tensor<ui32>
      %137 = stablehlo.subtract %136, %132 : tensor<ui32>
      %138 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %139 = stablehlo.shift_right_logical %130, %138 : tensor<10xui32>
      %140 = stablehlo.or %135, %139 : tensor<10xui32>
      %141 = stablehlo.xor %133, %140 : tensor<10xui32>
      %142 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %143 = stablehlo.reshape %142 : (tensor<1xui32>) -> tensor<ui32>
      %144 = stablehlo.add %133, %141 : tensor<10xui32>
      %145 = stablehlo.broadcast_in_dim %143, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %146 = stablehlo.shift_left %141, %145 : tensor<10xui32>
      %147 = stablehlo.constant dense<32> : tensor<ui32>
      %148 = stablehlo.subtract %147, %143 : tensor<ui32>
      %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %150 = stablehlo.shift_right_logical %141, %149 : tensor<10xui32>
      %151 = stablehlo.or %146, %150 : tensor<10xui32>
      %152 = stablehlo.xor %144, %151 : tensor<10xui32>
      %153 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %154 = stablehlo.reshape %153 : (tensor<1xui32>) -> tensor<ui32>
      %155 = stablehlo.add %144, %152 : tensor<10xui32>
      %156 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %157 = stablehlo.shift_left %152, %156 : tensor<10xui32>
      %158 = stablehlo.constant dense<32> : tensor<ui32>
      %159 = stablehlo.subtract %158, %154 : tensor<ui32>
      %160 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %161 = stablehlo.shift_right_logical %152, %160 : tensor<10xui32>
      %162 = stablehlo.or %157, %161 : tensor<10xui32>
      %163 = stablehlo.xor %155, %162 : tensor<10xui32>
      %164 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %165 = stablehlo.add %155, %164 : tensor<10xui32>
      %166 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %167 = stablehlo.add %163, %166 : tensor<10xui32>
      %168 = stablehlo.constant dense<1> : tensor<i32>
      %169 = stablehlo.add %iterArg_0, %168 : tensor<i32>
      %170 = stablehlo.convert %169 : (tensor<i32>) -> tensor<ui32>
      %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %172 = stablehlo.add %167, %171 : tensor<10xui32>
      %173 = stablehlo.constant dense<1> : tensor<i32>
      %174 = stablehlo.add %iterArg, %173 : tensor<i32>
      stablehlo.return %174, %119, %165, %172, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %64 = stablehlo.concatenate %63#2, %63#3, dim = 0 : (tensor<10xui32>, tensor<10xui32>) -> tensor<20xui32>
    %65 = stablehlo.reshape %64 : (tensor<20xui32>) -> tensor<5x4xui32>
    %66 = stablehlo.iota dim = 0 : tensor<20xui32>
    %67 = "stablehlo.slice"(%44) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %68 = stablehlo.reshape %67 : (tensor<1xui32>) -> tensor<ui32>
    %69 = "stablehlo.slice"(%44) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %70 = stablehlo.reshape %69 : (tensor<1xui32>) -> tensor<ui32>
    %71 = "stablehlo.slice"(%66) {limit_indices = dense<10> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<20xui32>) -> tensor<10xui32>
    %72 = "stablehlo.slice"(%66) {limit_indices = dense<20> : tensor<1xi64>, start_indices = dense<10> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<20xui32>) -> tensor<10xui32>
    %73 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %74 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %75 = stablehlo.xor %68, %70 : tensor<ui32>
    %76 = stablehlo.constant dense<466688986> : tensor<ui32>
    %77 = stablehlo.xor %75, %76 : tensor<ui32>
    %78 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %79 = stablehlo.add %71, %78 : tensor<10xui32>
    %80 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %81 = stablehlo.add %72, %80 : tensor<10xui32>
    %82 = stablehlo.constant dense<0> : tensor<i32>
    %83 = stablehlo.constant dense<0> : tensor<i32>
    %84:9 = stablehlo.while(%iterArg = %83, %iterArg_0 = %82, %iterArg_1 = %79, %iterArg_2 = %81, %iterArg_3 = %70, %iterArg_4 = %77, %iterArg_5 = %68, %iterArg_6 = %73, %iterArg_7 = %74) : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %118 = stablehlo.constant dense<5> : tensor<i32>
      %119 = stablehlo.compare  LT, %iterArg, %118,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %119 : tensor<i1>
    } do {
      %118 = stablehlo.constant dense<1> : tensor<i32>
      %119 = stablehlo.add %iterArg_0, %118 : tensor<i32>
      %120 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %121 = stablehlo.reshape %120 : (tensor<1xui32>) -> tensor<ui32>
      %122 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<10xui32>
      %123 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %124 = stablehlo.shift_left %iterArg_2, %123 : tensor<10xui32>
      %125 = stablehlo.constant dense<32> : tensor<ui32>
      %126 = stablehlo.subtract %125, %121 : tensor<ui32>
      %127 = stablehlo.broadcast_in_dim %126, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %128 = stablehlo.shift_right_logical %iterArg_2, %127 : tensor<10xui32>
      %129 = stablehlo.or %124, %128 : tensor<10xui32>
      %130 = stablehlo.xor %122, %129 : tensor<10xui32>
      %131 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %132 = stablehlo.reshape %131 : (tensor<1xui32>) -> tensor<ui32>
      %133 = stablehlo.add %122, %130 : tensor<10xui32>
      %134 = stablehlo.broadcast_in_dim %132, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %135 = stablehlo.shift_left %130, %134 : tensor<10xui32>
      %136 = stablehlo.constant dense<32> : tensor<ui32>
      %137 = stablehlo.subtract %136, %132 : tensor<ui32>
      %138 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %139 = stablehlo.shift_right_logical %130, %138 : tensor<10xui32>
      %140 = stablehlo.or %135, %139 : tensor<10xui32>
      %141 = stablehlo.xor %133, %140 : tensor<10xui32>
      %142 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %143 = stablehlo.reshape %142 : (tensor<1xui32>) -> tensor<ui32>
      %144 = stablehlo.add %133, %141 : tensor<10xui32>
      %145 = stablehlo.broadcast_in_dim %143, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %146 = stablehlo.shift_left %141, %145 : tensor<10xui32>
      %147 = stablehlo.constant dense<32> : tensor<ui32>
      %148 = stablehlo.subtract %147, %143 : tensor<ui32>
      %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %150 = stablehlo.shift_right_logical %141, %149 : tensor<10xui32>
      %151 = stablehlo.or %146, %150 : tensor<10xui32>
      %152 = stablehlo.xor %144, %151 : tensor<10xui32>
      %153 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %154 = stablehlo.reshape %153 : (tensor<1xui32>) -> tensor<ui32>
      %155 = stablehlo.add %144, %152 : tensor<10xui32>
      %156 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %157 = stablehlo.shift_left %152, %156 : tensor<10xui32>
      %158 = stablehlo.constant dense<32> : tensor<ui32>
      %159 = stablehlo.subtract %158, %154 : tensor<ui32>
      %160 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %161 = stablehlo.shift_right_logical %152, %160 : tensor<10xui32>
      %162 = stablehlo.or %157, %161 : tensor<10xui32>
      %163 = stablehlo.xor %155, %162 : tensor<10xui32>
      %164 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %165 = stablehlo.add %155, %164 : tensor<10xui32>
      %166 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %167 = stablehlo.add %163, %166 : tensor<10xui32>
      %168 = stablehlo.constant dense<1> : tensor<i32>
      %169 = stablehlo.add %iterArg_0, %168 : tensor<i32>
      %170 = stablehlo.convert %169 : (tensor<i32>) -> tensor<ui32>
      %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %172 = stablehlo.add %167, %171 : tensor<10xui32>
      %173 = stablehlo.constant dense<1> : tensor<i32>
      %174 = stablehlo.add %iterArg, %173 : tensor<i32>
      stablehlo.return %174, %119, %165, %172, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %85 = stablehlo.concatenate %84#2, %84#3, dim = 0 : (tensor<10xui32>, tensor<10xui32>) -> tensor<20xui32>
    %86 = stablehlo.reshape %85 : (tensor<20xui32>) -> tensor<5x4xui32>
    %87 = stablehlo.subtract %19, %18 : tensor<1x1xi32>
    %88 = stablehlo.convert %87 : (tensor<1x1xi32>) -> tensor<1x1xui32>
    %89 = stablehlo.compare  LE, %19, %18,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %90 = stablehlo.constant dense<1> : tensor<ui32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %92 = stablehlo.select %89, %91, %88 : tensor<1x1xi1>, tensor<1x1xui32>
    %93 = stablehlo.compare  GT, %19, %18,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %94 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %95 = stablehlo.and %94, %93 : tensor<1x1xi1>
    %96 = stablehlo.constant dense<1> : tensor<ui32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %98 = stablehlo.add %92, %97 : tensor<1x1xui32>
    %99 = stablehlo.select %95, %98, %92 : tensor<1x1xi1>, tensor<1x1xui32>
    %100 = stablehlo.constant dense<65536> : tensor<ui32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %102 = stablehlo.remainder %101, %99 : tensor<1x1xui32>
    %103 = stablehlo.multiply %102, %102 : tensor<1x1xui32>
    %104 = stablehlo.remainder %103, %99 : tensor<1x1xui32>
    %105 = stablehlo.broadcast_in_dim %99, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<5x4xui32>
    %106 = stablehlo.remainder %65, %105 : tensor<5x4xui32>
    %107 = stablehlo.broadcast_in_dim %104, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<5x4xui32>
    %108 = stablehlo.multiply %106, %107 : tensor<5x4xui32>
    %109 = stablehlo.broadcast_in_dim %99, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<5x4xui32>
    %110 = stablehlo.remainder %86, %109 : tensor<5x4xui32>
    %111 = stablehlo.add %108, %110 : tensor<5x4xui32>
    %112 = stablehlo.broadcast_in_dim %99, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<5x4xui32>
    %113 = stablehlo.remainder %111, %112 : tensor<5x4xui32>
    %114 = stablehlo.convert %113 : (tensor<5x4xui32>) -> tensor<5x4xi32>
    %115 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<5x4xi32>
    %116 = stablehlo.add %115, %114 : tensor<5x4xi32>
    %117 = stablehlo.custom_call @check.eq(%116, %1) : (tensor<5x4xi32>, tensor<5x4xi32>) -> tensor<i1>
    return %117 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<5x4xi32> {
    %0 = stablehlo.constant dense<[[1, -5, 1, 4], [-4, 1, -2, 1], [4, -2, 2, -4], [2, 3, 4, 1], [1, -1, 2, 3]]> : tensor<5x4xi32>
    return %0 : tensor<5x4xi32>
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
