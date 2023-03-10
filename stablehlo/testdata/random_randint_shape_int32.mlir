// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<i32>
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
    %18 = stablehlo.iota dim = 0 : tensor<4xui32>
    %19 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %20 = stablehlo.reshape %19 : (tensor<1xui32>) -> tensor<ui32>
    %21 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = "stablehlo.slice"(%18) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %24 = "stablehlo.slice"(%18) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %25 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %26 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %27 = stablehlo.xor %20, %22 : tensor<ui32>
    %28 = stablehlo.constant dense<466688986> : tensor<ui32>
    %29 = stablehlo.xor %27, %28 : tensor<ui32>
    %30 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %31 = stablehlo.add %23, %30 : tensor<2xui32>
    %32 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %33 = stablehlo.add %24, %32 : tensor<2xui32>
    %34 = stablehlo.constant dense<0> : tensor<i32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36:9 = stablehlo.while(%iterArg = %35, %iterArg_0 = %34, %iterArg_1 = %31, %iterArg_2 = %33, %iterArg_3 = %22, %iterArg_4 = %29, %iterArg_5 = %20, %iterArg_6 = %25, %iterArg_7 = %26) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %117 = stablehlo.constant dense<5> : tensor<i32>
      %118 = stablehlo.compare  LT, %iterArg, %117,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %118 : tensor<i1>
    } do {
      %117 = stablehlo.constant dense<1> : tensor<i32>
      %118 = stablehlo.add %iterArg_0, %117 : tensor<i32>
      %119 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %120 = stablehlo.reshape %119 : (tensor<1xui32>) -> tensor<ui32>
      %121 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %122 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %123 = stablehlo.shift_left %iterArg_2, %122 : tensor<2xui32>
      %124 = stablehlo.constant dense<32> : tensor<ui32>
      %125 = stablehlo.subtract %124, %120 : tensor<ui32>
      %126 = stablehlo.broadcast_in_dim %125, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %127 = stablehlo.shift_right_logical %iterArg_2, %126 : tensor<2xui32>
      %128 = stablehlo.or %123, %127 : tensor<2xui32>
      %129 = stablehlo.xor %121, %128 : tensor<2xui32>
      %130 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %131 = stablehlo.reshape %130 : (tensor<1xui32>) -> tensor<ui32>
      %132 = stablehlo.add %121, %129 : tensor<2xui32>
      %133 = stablehlo.broadcast_in_dim %131, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %134 = stablehlo.shift_left %129, %133 : tensor<2xui32>
      %135 = stablehlo.constant dense<32> : tensor<ui32>
      %136 = stablehlo.subtract %135, %131 : tensor<ui32>
      %137 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %138 = stablehlo.shift_right_logical %129, %137 : tensor<2xui32>
      %139 = stablehlo.or %134, %138 : tensor<2xui32>
      %140 = stablehlo.xor %132, %139 : tensor<2xui32>
      %141 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %142 = stablehlo.reshape %141 : (tensor<1xui32>) -> tensor<ui32>
      %143 = stablehlo.add %132, %140 : tensor<2xui32>
      %144 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %145 = stablehlo.shift_left %140, %144 : tensor<2xui32>
      %146 = stablehlo.constant dense<32> : tensor<ui32>
      %147 = stablehlo.subtract %146, %142 : tensor<ui32>
      %148 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %149 = stablehlo.shift_right_logical %140, %148 : tensor<2xui32>
      %150 = stablehlo.or %145, %149 : tensor<2xui32>
      %151 = stablehlo.xor %143, %150 : tensor<2xui32>
      %152 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %153 = stablehlo.reshape %152 : (tensor<1xui32>) -> tensor<ui32>
      %154 = stablehlo.add %143, %151 : tensor<2xui32>
      %155 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %156 = stablehlo.shift_left %151, %155 : tensor<2xui32>
      %157 = stablehlo.constant dense<32> : tensor<ui32>
      %158 = stablehlo.subtract %157, %153 : tensor<ui32>
      %159 = stablehlo.broadcast_in_dim %158, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %160 = stablehlo.shift_right_logical %151, %159 : tensor<2xui32>
      %161 = stablehlo.or %156, %160 : tensor<2xui32>
      %162 = stablehlo.xor %154, %161 : tensor<2xui32>
      %163 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %164 = stablehlo.add %154, %163 : tensor<2xui32>
      %165 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %166 = stablehlo.add %162, %165 : tensor<2xui32>
      %167 = stablehlo.constant dense<1> : tensor<i32>
      %168 = stablehlo.add %iterArg_0, %167 : tensor<i32>
      %169 = stablehlo.convert %168 : (tensor<i32>) -> tensor<ui32>
      %170 = stablehlo.broadcast_in_dim %169, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %171 = stablehlo.add %166, %170 : tensor<2xui32>
      %172 = stablehlo.constant dense<1> : tensor<i32>
      %173 = stablehlo.add %iterArg, %172 : tensor<i32>
      stablehlo.return %173, %118, %164, %171, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %37 = stablehlo.concatenate %36#2, %36#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %38 = stablehlo.reshape %37 : (tensor<4xui32>) -> tensor<2x2xui32>
    %39 = "stablehlo.slice"(%38) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %40 = stablehlo.reshape %39 : (tensor<1x2xui32>) -> tensor<2xui32>
    %41 = "stablehlo.slice"(%38) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %42 = stablehlo.reshape %41 : (tensor<1x2xui32>) -> tensor<2xui32>
    %43 = stablehlo.constant dense<0> : tensor<1xui32>
    %44 = stablehlo.iota dim = 0 : tensor<1xui32>
    %45 = "stablehlo.slice"(%40) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %46 = stablehlo.reshape %45 : (tensor<1xui32>) -> tensor<ui32>
    %47 = "stablehlo.slice"(%40) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %48 = stablehlo.reshape %47 : (tensor<1xui32>) -> tensor<ui32>
    %49 = stablehlo.concatenate %44, %43, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %50 = "stablehlo.slice"(%49) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %51 = "stablehlo.slice"(%49) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %52 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %53 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %54 = stablehlo.xor %46, %48 : tensor<ui32>
    %55 = stablehlo.constant dense<466688986> : tensor<ui32>
    %56 = stablehlo.xor %54, %55 : tensor<ui32>
    %57 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %58 = stablehlo.add %50, %57 : tensor<1xui32>
    %59 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %60 = stablehlo.add %51, %59 : tensor<1xui32>
    %61 = stablehlo.constant dense<0> : tensor<i32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63:9 = stablehlo.while(%iterArg = %62, %iterArg_0 = %61, %iterArg_1 = %58, %iterArg_2 = %60, %iterArg_3 = %48, %iterArg_4 = %56, %iterArg_5 = %46, %iterArg_6 = %52, %iterArg_7 = %53) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %117 = stablehlo.constant dense<5> : tensor<i32>
      %118 = stablehlo.compare  LT, %iterArg, %117,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %118 : tensor<i1>
    } do {
      %117 = stablehlo.constant dense<1> : tensor<i32>
      %118 = stablehlo.add %iterArg_0, %117 : tensor<i32>
      %119 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %120 = stablehlo.reshape %119 : (tensor<1xui32>) -> tensor<ui32>
      %121 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1xui32>
      %122 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %123 = stablehlo.shift_left %iterArg_2, %122 : tensor<1xui32>
      %124 = stablehlo.constant dense<32> : tensor<ui32>
      %125 = stablehlo.subtract %124, %120 : tensor<ui32>
      %126 = stablehlo.broadcast_in_dim %125, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %127 = stablehlo.shift_right_logical %iterArg_2, %126 : tensor<1xui32>
      %128 = stablehlo.or %123, %127 : tensor<1xui32>
      %129 = stablehlo.xor %121, %128 : tensor<1xui32>
      %130 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %131 = stablehlo.reshape %130 : (tensor<1xui32>) -> tensor<ui32>
      %132 = stablehlo.add %121, %129 : tensor<1xui32>
      %133 = stablehlo.broadcast_in_dim %131, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %134 = stablehlo.shift_left %129, %133 : tensor<1xui32>
      %135 = stablehlo.constant dense<32> : tensor<ui32>
      %136 = stablehlo.subtract %135, %131 : tensor<ui32>
      %137 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %138 = stablehlo.shift_right_logical %129, %137 : tensor<1xui32>
      %139 = stablehlo.or %134, %138 : tensor<1xui32>
      %140 = stablehlo.xor %132, %139 : tensor<1xui32>
      %141 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %142 = stablehlo.reshape %141 : (tensor<1xui32>) -> tensor<ui32>
      %143 = stablehlo.add %132, %140 : tensor<1xui32>
      %144 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %145 = stablehlo.shift_left %140, %144 : tensor<1xui32>
      %146 = stablehlo.constant dense<32> : tensor<ui32>
      %147 = stablehlo.subtract %146, %142 : tensor<ui32>
      %148 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %149 = stablehlo.shift_right_logical %140, %148 : tensor<1xui32>
      %150 = stablehlo.or %145, %149 : tensor<1xui32>
      %151 = stablehlo.xor %143, %150 : tensor<1xui32>
      %152 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %153 = stablehlo.reshape %152 : (tensor<1xui32>) -> tensor<ui32>
      %154 = stablehlo.add %143, %151 : tensor<1xui32>
      %155 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %156 = stablehlo.shift_left %151, %155 : tensor<1xui32>
      %157 = stablehlo.constant dense<32> : tensor<ui32>
      %158 = stablehlo.subtract %157, %153 : tensor<ui32>
      %159 = stablehlo.broadcast_in_dim %158, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %160 = stablehlo.shift_right_logical %151, %159 : tensor<1xui32>
      %161 = stablehlo.or %156, %160 : tensor<1xui32>
      %162 = stablehlo.xor %154, %161 : tensor<1xui32>
      %163 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %164 = stablehlo.add %154, %163 : tensor<1xui32>
      %165 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %166 = stablehlo.add %162, %165 : tensor<1xui32>
      %167 = stablehlo.constant dense<1> : tensor<i32>
      %168 = stablehlo.add %iterArg_0, %167 : tensor<i32>
      %169 = stablehlo.convert %168 : (tensor<i32>) -> tensor<ui32>
      %170 = stablehlo.broadcast_in_dim %169, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %171 = stablehlo.add %166, %170 : tensor<1xui32>
      %172 = stablehlo.constant dense<1> : tensor<i32>
      %173 = stablehlo.add %iterArg, %172 : tensor<i32>
      stablehlo.return %173, %118, %164, %171, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %64 = stablehlo.concatenate %63#2, %63#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %65 = stablehlo.constant dense<0> : tensor<i32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %67 = "stablehlo.gather"(%64, %66) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
    %68 = stablehlo.reshape %67 : (tensor<1xui32>) -> tensor<ui32>
    %69 = stablehlo.constant dense<0> : tensor<1xui32>
    %70 = stablehlo.iota dim = 0 : tensor<1xui32>
    %71 = "stablehlo.slice"(%42) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %72 = stablehlo.reshape %71 : (tensor<1xui32>) -> tensor<ui32>
    %73 = "stablehlo.slice"(%42) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %74 = stablehlo.reshape %73 : (tensor<1xui32>) -> tensor<ui32>
    %75 = stablehlo.concatenate %70, %69, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %76 = "stablehlo.slice"(%75) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %77 = "stablehlo.slice"(%75) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %78 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %79 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %80 = stablehlo.xor %72, %74 : tensor<ui32>
    %81 = stablehlo.constant dense<466688986> : tensor<ui32>
    %82 = stablehlo.xor %80, %81 : tensor<ui32>
    %83 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %84 = stablehlo.add %76, %83 : tensor<1xui32>
    %85 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %86 = stablehlo.add %77, %85 : tensor<1xui32>
    %87 = stablehlo.constant dense<0> : tensor<i32>
    %88 = stablehlo.constant dense<0> : tensor<i32>
    %89:9 = stablehlo.while(%iterArg = %88, %iterArg_0 = %87, %iterArg_1 = %84, %iterArg_2 = %86, %iterArg_3 = %74, %iterArg_4 = %82, %iterArg_5 = %72, %iterArg_6 = %78, %iterArg_7 = %79) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %117 = stablehlo.constant dense<5> : tensor<i32>
      %118 = stablehlo.compare  LT, %iterArg, %117,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %118 : tensor<i1>
    } do {
      %117 = stablehlo.constant dense<1> : tensor<i32>
      %118 = stablehlo.add %iterArg_0, %117 : tensor<i32>
      %119 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %120 = stablehlo.reshape %119 : (tensor<1xui32>) -> tensor<ui32>
      %121 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1xui32>
      %122 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %123 = stablehlo.shift_left %iterArg_2, %122 : tensor<1xui32>
      %124 = stablehlo.constant dense<32> : tensor<ui32>
      %125 = stablehlo.subtract %124, %120 : tensor<ui32>
      %126 = stablehlo.broadcast_in_dim %125, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %127 = stablehlo.shift_right_logical %iterArg_2, %126 : tensor<1xui32>
      %128 = stablehlo.or %123, %127 : tensor<1xui32>
      %129 = stablehlo.xor %121, %128 : tensor<1xui32>
      %130 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %131 = stablehlo.reshape %130 : (tensor<1xui32>) -> tensor<ui32>
      %132 = stablehlo.add %121, %129 : tensor<1xui32>
      %133 = stablehlo.broadcast_in_dim %131, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %134 = stablehlo.shift_left %129, %133 : tensor<1xui32>
      %135 = stablehlo.constant dense<32> : tensor<ui32>
      %136 = stablehlo.subtract %135, %131 : tensor<ui32>
      %137 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %138 = stablehlo.shift_right_logical %129, %137 : tensor<1xui32>
      %139 = stablehlo.or %134, %138 : tensor<1xui32>
      %140 = stablehlo.xor %132, %139 : tensor<1xui32>
      %141 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %142 = stablehlo.reshape %141 : (tensor<1xui32>) -> tensor<ui32>
      %143 = stablehlo.add %132, %140 : tensor<1xui32>
      %144 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %145 = stablehlo.shift_left %140, %144 : tensor<1xui32>
      %146 = stablehlo.constant dense<32> : tensor<ui32>
      %147 = stablehlo.subtract %146, %142 : tensor<ui32>
      %148 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %149 = stablehlo.shift_right_logical %140, %148 : tensor<1xui32>
      %150 = stablehlo.or %145, %149 : tensor<1xui32>
      %151 = stablehlo.xor %143, %150 : tensor<1xui32>
      %152 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %153 = stablehlo.reshape %152 : (tensor<1xui32>) -> tensor<ui32>
      %154 = stablehlo.add %143, %151 : tensor<1xui32>
      %155 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %156 = stablehlo.shift_left %151, %155 : tensor<1xui32>
      %157 = stablehlo.constant dense<32> : tensor<ui32>
      %158 = stablehlo.subtract %157, %153 : tensor<ui32>
      %159 = stablehlo.broadcast_in_dim %158, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %160 = stablehlo.shift_right_logical %151, %159 : tensor<1xui32>
      %161 = stablehlo.or %156, %160 : tensor<1xui32>
      %162 = stablehlo.xor %154, %161 : tensor<1xui32>
      %163 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %164 = stablehlo.add %154, %163 : tensor<1xui32>
      %165 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %166 = stablehlo.add %162, %165 : tensor<1xui32>
      %167 = stablehlo.constant dense<1> : tensor<i32>
      %168 = stablehlo.add %iterArg_0, %167 : tensor<i32>
      %169 = stablehlo.convert %168 : (tensor<i32>) -> tensor<ui32>
      %170 = stablehlo.broadcast_in_dim %169, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %171 = stablehlo.add %166, %170 : tensor<1xui32>
      %172 = stablehlo.constant dense<1> : tensor<i32>
      %173 = stablehlo.add %iterArg, %172 : tensor<i32>
      stablehlo.return %173, %118, %164, %171, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %90 = stablehlo.concatenate %89#2, %89#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %91 = stablehlo.constant dense<0> : tensor<i32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %93 = "stablehlo.gather"(%90, %92) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
    %94 = stablehlo.reshape %93 : (tensor<1xui32>) -> tensor<ui32>
    %95 = stablehlo.subtract %17, %12 : tensor<i32>
    %96 = stablehlo.convert %95 : (tensor<i32>) -> tensor<ui32>
    %97 = stablehlo.compare  LE, %17, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %98 = stablehlo.constant dense<1> : tensor<ui32>
    %99 = stablehlo.select %97, %98, %96 : tensor<i1>, tensor<ui32>
    %100 = stablehlo.compare  GT, %17, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %101 = stablehlo.and %7, %100 : tensor<i1>
    %102 = stablehlo.constant dense<1> : tensor<ui32>
    %103 = stablehlo.add %99, %102 : tensor<ui32>
    %104 = stablehlo.select %101, %103, %99 : tensor<i1>, tensor<ui32>
    %105 = stablehlo.constant dense<65536> : tensor<ui32>
    %106 = stablehlo.remainder %105, %104 : tensor<ui32>
    %107 = stablehlo.multiply %106, %106 : tensor<ui32>
    %108 = stablehlo.remainder %107, %104 : tensor<ui32>
    %109 = stablehlo.remainder %68, %104 : tensor<ui32>
    %110 = stablehlo.multiply %109, %108 : tensor<ui32>
    %111 = stablehlo.remainder %94, %104 : tensor<ui32>
    %112 = stablehlo.add %110, %111 : tensor<ui32>
    %113 = stablehlo.remainder %112, %104 : tensor<ui32>
    %114 = stablehlo.convert %113 : (tensor<ui32>) -> tensor<i32>
    %115 = stablehlo.add %12, %114 : tensor<i32>
    %116 = stablehlo.custom_call @check.eq(%115, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %116 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<-4> : tensor<i32>
    return %0 : tensor<i32>
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
