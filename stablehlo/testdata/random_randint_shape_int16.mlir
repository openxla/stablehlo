// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<i16>
    %2 = stablehlo.constant dense<32767> : tensor<i16>
    %3 = stablehlo.constant dense<-32768> : tensor<i16>
    %4 = stablehlo.constant dense<32767> : tensor<i16>
    %5 = call @clip(%2, %3, %4) : (tensor<i16>, tensor<i16>, tensor<i16>) -> tensor<i16>
    %6 = stablehlo.convert %5 : (tensor<i16>) -> tensor<i32>
    %7 = stablehlo.constant dense<5> : tensor<i32>
    %8 = stablehlo.compare  GT, %7, %6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.constant dense<-5> : tensor<i32>
    %10 = stablehlo.constant dense<-32768> : tensor<i32>
    %11 = stablehlo.constant dense<32767> : tensor<i32>
    %12 = call @clip_0(%9, %10, %11) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = stablehlo.convert %12 : (tensor<i32>) -> tensor<i16>
    %14 = stablehlo.constant dense<5> : tensor<i32>
    %15 = stablehlo.constant dense<-32768> : tensor<i32>
    %16 = stablehlo.constant dense<32767> : tensor<i32>
    %17 = call @clip_1(%14, %15, %16) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %18 = stablehlo.convert %17 : (tensor<i32>) -> tensor<i16>
    %19 = stablehlo.iota dim = 0 : tensor<4xui32>
    %20 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %21 = stablehlo.reshape %20 : (tensor<1xui32>) -> tensor<ui32>
    %22 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %23 = stablehlo.reshape %22 : (tensor<1xui32>) -> tensor<ui32>
    %24 = "stablehlo.slice"(%19) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %25 = "stablehlo.slice"(%19) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %26 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %27 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %28 = stablehlo.xor %21, %23 : tensor<ui32>
    %29 = stablehlo.constant dense<466688986> : tensor<ui32>
    %30 = stablehlo.xor %28, %29 : tensor<ui32>
    %31 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %32 = stablehlo.add %24, %31 : tensor<2xui32>
    %33 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %34 = stablehlo.add %25, %33 : tensor<2xui32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<0> : tensor<i32>
    %37:9 = stablehlo.while(%iterArg = %36, %iterArg_0 = %35, %iterArg_1 = %32, %iterArg_2 = %34, %iterArg_3 = %23, %iterArg_4 = %30, %iterArg_5 = %21, %iterArg_6 = %26, %iterArg_7 = %27) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %148 = stablehlo.constant dense<5> : tensor<i32>
      %149 = stablehlo.compare  LT, %iterArg, %148,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %149 : tensor<i1>
    } do {
      %148 = stablehlo.constant dense<1> : tensor<i32>
      %149 = stablehlo.add %iterArg_0, %148 : tensor<i32>
      %150 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %151 = stablehlo.reshape %150 : (tensor<1xui32>) -> tensor<ui32>
      %152 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %153 = stablehlo.broadcast_in_dim %151, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %154 = stablehlo.shift_left %iterArg_2, %153 : tensor<2xui32>
      %155 = stablehlo.constant dense<32> : tensor<ui32>
      %156 = stablehlo.subtract %155, %151 : tensor<ui32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %158 = stablehlo.shift_right_logical %iterArg_2, %157 : tensor<2xui32>
      %159 = stablehlo.or %154, %158 : tensor<2xui32>
      %160 = stablehlo.xor %152, %159 : tensor<2xui32>
      %161 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %162 = stablehlo.reshape %161 : (tensor<1xui32>) -> tensor<ui32>
      %163 = stablehlo.add %152, %160 : tensor<2xui32>
      %164 = stablehlo.broadcast_in_dim %162, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %165 = stablehlo.shift_left %160, %164 : tensor<2xui32>
      %166 = stablehlo.constant dense<32> : tensor<ui32>
      %167 = stablehlo.subtract %166, %162 : tensor<ui32>
      %168 = stablehlo.broadcast_in_dim %167, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %169 = stablehlo.shift_right_logical %160, %168 : tensor<2xui32>
      %170 = stablehlo.or %165, %169 : tensor<2xui32>
      %171 = stablehlo.xor %163, %170 : tensor<2xui32>
      %172 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %173 = stablehlo.reshape %172 : (tensor<1xui32>) -> tensor<ui32>
      %174 = stablehlo.add %163, %171 : tensor<2xui32>
      %175 = stablehlo.broadcast_in_dim %173, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %176 = stablehlo.shift_left %171, %175 : tensor<2xui32>
      %177 = stablehlo.constant dense<32> : tensor<ui32>
      %178 = stablehlo.subtract %177, %173 : tensor<ui32>
      %179 = stablehlo.broadcast_in_dim %178, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %180 = stablehlo.shift_right_logical %171, %179 : tensor<2xui32>
      %181 = stablehlo.or %176, %180 : tensor<2xui32>
      %182 = stablehlo.xor %174, %181 : tensor<2xui32>
      %183 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %184 = stablehlo.reshape %183 : (tensor<1xui32>) -> tensor<ui32>
      %185 = stablehlo.add %174, %182 : tensor<2xui32>
      %186 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %187 = stablehlo.shift_left %182, %186 : tensor<2xui32>
      %188 = stablehlo.constant dense<32> : tensor<ui32>
      %189 = stablehlo.subtract %188, %184 : tensor<ui32>
      %190 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %191 = stablehlo.shift_right_logical %182, %190 : tensor<2xui32>
      %192 = stablehlo.or %187, %191 : tensor<2xui32>
      %193 = stablehlo.xor %185, %192 : tensor<2xui32>
      %194 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %195 = stablehlo.add %185, %194 : tensor<2xui32>
      %196 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %197 = stablehlo.add %193, %196 : tensor<2xui32>
      %198 = stablehlo.constant dense<1> : tensor<i32>
      %199 = stablehlo.add %iterArg_0, %198 : tensor<i32>
      %200 = stablehlo.convert %199 : (tensor<i32>) -> tensor<ui32>
      %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %202 = stablehlo.add %197, %201 : tensor<2xui32>
      %203 = stablehlo.constant dense<1> : tensor<i32>
      %204 = stablehlo.add %iterArg, %203 : tensor<i32>
      stablehlo.return %204, %149, %195, %202, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %38 = stablehlo.concatenate %37#2, %37#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %39 = stablehlo.reshape %38 : (tensor<4xui32>) -> tensor<2x2xui32>
    %40 = "stablehlo.slice"(%39) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %41 = stablehlo.reshape %40 : (tensor<1x2xui32>) -> tensor<2xui32>
    %42 = "stablehlo.slice"(%39) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %43 = stablehlo.reshape %42 : (tensor<1x2xui32>) -> tensor<2xui32>
    %44 = stablehlo.constant dense<0> : tensor<1xui32>
    %45 = stablehlo.iota dim = 0 : tensor<1xui32>
    %46 = "stablehlo.slice"(%41) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %47 = stablehlo.reshape %46 : (tensor<1xui32>) -> tensor<ui32>
    %48 = "stablehlo.slice"(%41) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %49 = stablehlo.reshape %48 : (tensor<1xui32>) -> tensor<ui32>
    %50 = stablehlo.concatenate %45, %44, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %51 = "stablehlo.slice"(%50) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %52 = "stablehlo.slice"(%50) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %53 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %54 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %55 = stablehlo.xor %47, %49 : tensor<ui32>
    %56 = stablehlo.constant dense<466688986> : tensor<ui32>
    %57 = stablehlo.xor %55, %56 : tensor<ui32>
    %58 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %59 = stablehlo.add %51, %58 : tensor<1xui32>
    %60 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %61 = stablehlo.add %52, %60 : tensor<1xui32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63 = stablehlo.constant dense<0> : tensor<i32>
    %64:9 = stablehlo.while(%iterArg = %63, %iterArg_0 = %62, %iterArg_1 = %59, %iterArg_2 = %61, %iterArg_3 = %49, %iterArg_4 = %57, %iterArg_5 = %47, %iterArg_6 = %53, %iterArg_7 = %54) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %148 = stablehlo.constant dense<5> : tensor<i32>
      %149 = stablehlo.compare  LT, %iterArg, %148,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %149 : tensor<i1>
    } do {
      %148 = stablehlo.constant dense<1> : tensor<i32>
      %149 = stablehlo.add %iterArg_0, %148 : tensor<i32>
      %150 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %151 = stablehlo.reshape %150 : (tensor<1xui32>) -> tensor<ui32>
      %152 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1xui32>
      %153 = stablehlo.broadcast_in_dim %151, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %154 = stablehlo.shift_left %iterArg_2, %153 : tensor<1xui32>
      %155 = stablehlo.constant dense<32> : tensor<ui32>
      %156 = stablehlo.subtract %155, %151 : tensor<ui32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %158 = stablehlo.shift_right_logical %iterArg_2, %157 : tensor<1xui32>
      %159 = stablehlo.or %154, %158 : tensor<1xui32>
      %160 = stablehlo.xor %152, %159 : tensor<1xui32>
      %161 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %162 = stablehlo.reshape %161 : (tensor<1xui32>) -> tensor<ui32>
      %163 = stablehlo.add %152, %160 : tensor<1xui32>
      %164 = stablehlo.broadcast_in_dim %162, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %165 = stablehlo.shift_left %160, %164 : tensor<1xui32>
      %166 = stablehlo.constant dense<32> : tensor<ui32>
      %167 = stablehlo.subtract %166, %162 : tensor<ui32>
      %168 = stablehlo.broadcast_in_dim %167, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %169 = stablehlo.shift_right_logical %160, %168 : tensor<1xui32>
      %170 = stablehlo.or %165, %169 : tensor<1xui32>
      %171 = stablehlo.xor %163, %170 : tensor<1xui32>
      %172 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %173 = stablehlo.reshape %172 : (tensor<1xui32>) -> tensor<ui32>
      %174 = stablehlo.add %163, %171 : tensor<1xui32>
      %175 = stablehlo.broadcast_in_dim %173, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %176 = stablehlo.shift_left %171, %175 : tensor<1xui32>
      %177 = stablehlo.constant dense<32> : tensor<ui32>
      %178 = stablehlo.subtract %177, %173 : tensor<ui32>
      %179 = stablehlo.broadcast_in_dim %178, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %180 = stablehlo.shift_right_logical %171, %179 : tensor<1xui32>
      %181 = stablehlo.or %176, %180 : tensor<1xui32>
      %182 = stablehlo.xor %174, %181 : tensor<1xui32>
      %183 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %184 = stablehlo.reshape %183 : (tensor<1xui32>) -> tensor<ui32>
      %185 = stablehlo.add %174, %182 : tensor<1xui32>
      %186 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %187 = stablehlo.shift_left %182, %186 : tensor<1xui32>
      %188 = stablehlo.constant dense<32> : tensor<ui32>
      %189 = stablehlo.subtract %188, %184 : tensor<ui32>
      %190 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %191 = stablehlo.shift_right_logical %182, %190 : tensor<1xui32>
      %192 = stablehlo.or %187, %191 : tensor<1xui32>
      %193 = stablehlo.xor %185, %192 : tensor<1xui32>
      %194 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %195 = stablehlo.add %185, %194 : tensor<1xui32>
      %196 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %197 = stablehlo.add %193, %196 : tensor<1xui32>
      %198 = stablehlo.constant dense<1> : tensor<i32>
      %199 = stablehlo.add %iterArg_0, %198 : tensor<i32>
      %200 = stablehlo.convert %199 : (tensor<i32>) -> tensor<ui32>
      %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %202 = stablehlo.add %197, %201 : tensor<1xui32>
      %203 = stablehlo.constant dense<1> : tensor<i32>
      %204 = stablehlo.add %iterArg, %203 : tensor<i32>
      stablehlo.return %204, %149, %195, %202, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %65 = stablehlo.concatenate %64#2, %64#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %66 = stablehlo.constant dense<0> : tensor<i32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %68 = "stablehlo.gather"(%65, %67) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [1] : (tensor<1xui32>) -> tensor<1x1xui32>
    %70 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %71 = stablehlo.constant dense<16> : tensor<ui32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %73 = stablehlo.multiply %72, %70 : tensor<2x1xui32>
    %74 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<2x1xui32>
    %75 = stablehlo.shift_right_logical %74, %73 : tensor<2x1xui32>
    %76 = stablehlo.constant dense<65535> : tensor<ui32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %78 = stablehlo.and %77, %75 : tensor<2x1xui32>
    %79 = stablehlo.transpose %78, dims = [1, 0] : (tensor<2x1xui32>) -> tensor<1x2xui32>
    %80 = stablehlo.reshape %79 : (tensor<1x2xui32>) -> tensor<2xui32>
    %81 = stablehlo.convert %80 : (tensor<2xui32>) -> tensor<2xui16>
    %82 = stablehlo.constant dense<0> : tensor<i32>
    %83 = stablehlo.dynamic_slice %81, %82, sizes = [1] : (tensor<2xui16>, tensor<i32>) -> tensor<1xui16>
    %84 = stablehlo.reshape %83 : (tensor<1xui16>) -> tensor<ui16>
    %85 = stablehlo.constant dense<0> : tensor<1xui32>
    %86 = stablehlo.iota dim = 0 : tensor<1xui32>
    %87 = "stablehlo.slice"(%43) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %88 = stablehlo.reshape %87 : (tensor<1xui32>) -> tensor<ui32>
    %89 = "stablehlo.slice"(%43) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %90 = stablehlo.reshape %89 : (tensor<1xui32>) -> tensor<ui32>
    %91 = stablehlo.concatenate %86, %85, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %92 = "stablehlo.slice"(%91) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %93 = "stablehlo.slice"(%91) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %94 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %95 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %96 = stablehlo.xor %88, %90 : tensor<ui32>
    %97 = stablehlo.constant dense<466688986> : tensor<ui32>
    %98 = stablehlo.xor %96, %97 : tensor<ui32>
    %99 = stablehlo.broadcast_in_dim %88, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %100 = stablehlo.add %92, %99 : tensor<1xui32>
    %101 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %102 = stablehlo.add %93, %101 : tensor<1xui32>
    %103 = stablehlo.constant dense<0> : tensor<i32>
    %104 = stablehlo.constant dense<0> : tensor<i32>
    %105:9 = stablehlo.while(%iterArg = %104, %iterArg_0 = %103, %iterArg_1 = %100, %iterArg_2 = %102, %iterArg_3 = %90, %iterArg_4 = %98, %iterArg_5 = %88, %iterArg_6 = %94, %iterArg_7 = %95) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %148 = stablehlo.constant dense<5> : tensor<i32>
      %149 = stablehlo.compare  LT, %iterArg, %148,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %149 : tensor<i1>
    } do {
      %148 = stablehlo.constant dense<1> : tensor<i32>
      %149 = stablehlo.add %iterArg_0, %148 : tensor<i32>
      %150 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %151 = stablehlo.reshape %150 : (tensor<1xui32>) -> tensor<ui32>
      %152 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1xui32>
      %153 = stablehlo.broadcast_in_dim %151, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %154 = stablehlo.shift_left %iterArg_2, %153 : tensor<1xui32>
      %155 = stablehlo.constant dense<32> : tensor<ui32>
      %156 = stablehlo.subtract %155, %151 : tensor<ui32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %158 = stablehlo.shift_right_logical %iterArg_2, %157 : tensor<1xui32>
      %159 = stablehlo.or %154, %158 : tensor<1xui32>
      %160 = stablehlo.xor %152, %159 : tensor<1xui32>
      %161 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %162 = stablehlo.reshape %161 : (tensor<1xui32>) -> tensor<ui32>
      %163 = stablehlo.add %152, %160 : tensor<1xui32>
      %164 = stablehlo.broadcast_in_dim %162, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %165 = stablehlo.shift_left %160, %164 : tensor<1xui32>
      %166 = stablehlo.constant dense<32> : tensor<ui32>
      %167 = stablehlo.subtract %166, %162 : tensor<ui32>
      %168 = stablehlo.broadcast_in_dim %167, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %169 = stablehlo.shift_right_logical %160, %168 : tensor<1xui32>
      %170 = stablehlo.or %165, %169 : tensor<1xui32>
      %171 = stablehlo.xor %163, %170 : tensor<1xui32>
      %172 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %173 = stablehlo.reshape %172 : (tensor<1xui32>) -> tensor<ui32>
      %174 = stablehlo.add %163, %171 : tensor<1xui32>
      %175 = stablehlo.broadcast_in_dim %173, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %176 = stablehlo.shift_left %171, %175 : tensor<1xui32>
      %177 = stablehlo.constant dense<32> : tensor<ui32>
      %178 = stablehlo.subtract %177, %173 : tensor<ui32>
      %179 = stablehlo.broadcast_in_dim %178, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %180 = stablehlo.shift_right_logical %171, %179 : tensor<1xui32>
      %181 = stablehlo.or %176, %180 : tensor<1xui32>
      %182 = stablehlo.xor %174, %181 : tensor<1xui32>
      %183 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %184 = stablehlo.reshape %183 : (tensor<1xui32>) -> tensor<ui32>
      %185 = stablehlo.add %174, %182 : tensor<1xui32>
      %186 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %187 = stablehlo.shift_left %182, %186 : tensor<1xui32>
      %188 = stablehlo.constant dense<32> : tensor<ui32>
      %189 = stablehlo.subtract %188, %184 : tensor<ui32>
      %190 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %191 = stablehlo.shift_right_logical %182, %190 : tensor<1xui32>
      %192 = stablehlo.or %187, %191 : tensor<1xui32>
      %193 = stablehlo.xor %185, %192 : tensor<1xui32>
      %194 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %195 = stablehlo.add %185, %194 : tensor<1xui32>
      %196 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %197 = stablehlo.add %193, %196 : tensor<1xui32>
      %198 = stablehlo.constant dense<1> : tensor<i32>
      %199 = stablehlo.add %iterArg_0, %198 : tensor<i32>
      %200 = stablehlo.convert %199 : (tensor<i32>) -> tensor<ui32>
      %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %202 = stablehlo.add %197, %201 : tensor<1xui32>
      %203 = stablehlo.constant dense<1> : tensor<i32>
      %204 = stablehlo.add %iterArg, %203 : tensor<i32>
      stablehlo.return %204, %149, %195, %202, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %106 = stablehlo.concatenate %105#2, %105#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %107 = stablehlo.constant dense<0> : tensor<i32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %109 = "stablehlo.gather"(%106, %108) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [1] : (tensor<1xui32>) -> tensor<1x1xui32>
    %111 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %112 = stablehlo.constant dense<16> : tensor<ui32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %114 = stablehlo.multiply %113, %111 : tensor<2x1xui32>
    %115 = stablehlo.broadcast_in_dim %110, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<2x1xui32>
    %116 = stablehlo.shift_right_logical %115, %114 : tensor<2x1xui32>
    %117 = stablehlo.constant dense<65535> : tensor<ui32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %119 = stablehlo.and %118, %116 : tensor<2x1xui32>
    %120 = stablehlo.transpose %119, dims = [1, 0] : (tensor<2x1xui32>) -> tensor<1x2xui32>
    %121 = stablehlo.reshape %120 : (tensor<1x2xui32>) -> tensor<2xui32>
    %122 = stablehlo.convert %121 : (tensor<2xui32>) -> tensor<2xui16>
    %123 = stablehlo.constant dense<0> : tensor<i32>
    %124 = stablehlo.dynamic_slice %122, %123, sizes = [1] : (tensor<2xui16>, tensor<i32>) -> tensor<1xui16>
    %125 = stablehlo.reshape %124 : (tensor<1xui16>) -> tensor<ui16>
    %126 = stablehlo.subtract %18, %13 : tensor<i16>
    %127 = stablehlo.convert %126 : (tensor<i16>) -> tensor<ui16>
    %128 = stablehlo.compare  LE, %18, %13,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %129 = stablehlo.constant dense<1> : tensor<ui16>
    %130 = stablehlo.select %128, %129, %127 : tensor<i1>, tensor<ui16>
    %131 = stablehlo.compare  GT, %18, %13,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %132 = stablehlo.and %8, %131 : tensor<i1>
    %133 = stablehlo.constant dense<1> : tensor<ui16>
    %134 = stablehlo.add %130, %133 : tensor<ui16>
    %135 = stablehlo.select %132, %134, %130 : tensor<i1>, tensor<ui16>
    %136 = stablehlo.constant dense<256> : tensor<ui16>
    %137 = stablehlo.remainder %136, %135 : tensor<ui16>
    %138 = stablehlo.multiply %137, %137 : tensor<ui16>
    %139 = stablehlo.remainder %138, %135 : tensor<ui16>
    %140 = stablehlo.remainder %84, %135 : tensor<ui16>
    %141 = stablehlo.multiply %140, %139 : tensor<ui16>
    %142 = stablehlo.remainder %125, %135 : tensor<ui16>
    %143 = stablehlo.add %141, %142 : tensor<ui16>
    %144 = stablehlo.remainder %143, %135 : tensor<ui16>
    %145 = stablehlo.convert %144 : (tensor<ui16>) -> tensor<i16>
    %146 = stablehlo.add %13, %145 : tensor<i16>
    %147 = stablehlo.custom_call @check.eq(%146, %1) : (tensor<i16>, tensor<i16>) -> tensor<i1>
    return %147 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<i16> {
    %0 = stablehlo.constant dense<0> : tensor<i16>
    return %0 : tensor<i16>
  }
  func.func private @clip(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<i16>) -> tensor<i16> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i16>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i16>
    return %1 : tensor<i16>
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
