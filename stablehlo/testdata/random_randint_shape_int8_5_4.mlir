// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<5x4xi8>
    %2 = stablehlo.constant dense<127> : tensor<i8>
    %3 = stablehlo.constant dense<-128> : tensor<i8>
    %4 = stablehlo.constant dense<127> : tensor<i8>
    %5 = call @clip(%2, %3, %4) : (tensor<i8>, tensor<i8>, tensor<i8>) -> tensor<i8>
    %6 = stablehlo.convert %5 : (tensor<i8>) -> tensor<i32>
    %7 = stablehlo.constant dense<5> : tensor<i32>
    %8 = stablehlo.compare  GT, %7, %6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.constant dense<-5> : tensor<i32>
    %10 = stablehlo.constant dense<-128> : tensor<i32>
    %11 = stablehlo.constant dense<127> : tensor<i32>
    %12 = call @clip_0(%9, %10, %11) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = stablehlo.convert %12 : (tensor<i32>) -> tensor<i8>
    %14 = stablehlo.constant dense<5> : tensor<i32>
    %15 = stablehlo.constant dense<-128> : tensor<i32>
    %16 = stablehlo.constant dense<127> : tensor<i32>
    %17 = call @clip_1(%14, %15, %16) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %18 = stablehlo.convert %17 : (tensor<i32>) -> tensor<i8>
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i8>) -> tensor<1x1xi8>
    %20 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i8>) -> tensor<1x1xi8>
    %21 = stablehlo.iota dim = 0 : tensor<4xui32>
    %22 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %23 = stablehlo.reshape %22 : (tensor<1xui32>) -> tensor<ui32>
    %24 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %25 = stablehlo.reshape %24 : (tensor<1xui32>) -> tensor<ui32>
    %26 = "stablehlo.slice"(%21) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %27 = "stablehlo.slice"(%21) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %28 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %29 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %30 = stablehlo.xor %23, %25 : tensor<ui32>
    %31 = stablehlo.constant dense<466688986> : tensor<ui32>
    %32 = stablehlo.xor %30, %31 : tensor<ui32>
    %33 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %34 = stablehlo.add %26, %33 : tensor<2xui32>
    %35 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %36 = stablehlo.add %27, %35 : tensor<2xui32>
    %37 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.constant dense<0> : tensor<i32>
    %39:9 = stablehlo.while(%iterArg = %38, %iterArg_0 = %37, %iterArg_1 = %34, %iterArg_2 = %36, %iterArg_3 = %25, %iterArg_4 = %32, %iterArg_5 = %23, %iterArg_6 = %28, %iterArg_7 = %29) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %157 = stablehlo.constant dense<5> : tensor<i32>
      %158 = stablehlo.compare  LT, %iterArg, %157,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %158 : tensor<i1>
    } do {
      %157 = stablehlo.constant dense<1> : tensor<i32>
      %158 = stablehlo.add %iterArg_0, %157 : tensor<i32>
      %159 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %160 = stablehlo.reshape %159 : (tensor<1xui32>) -> tensor<ui32>
      %161 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %162 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %163 = stablehlo.shift_left %iterArg_2, %162 : tensor<2xui32>
      %164 = stablehlo.constant dense<32> : tensor<ui32>
      %165 = stablehlo.subtract %164, %160 : tensor<ui32>
      %166 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %167 = stablehlo.shift_right_logical %iterArg_2, %166 : tensor<2xui32>
      %168 = stablehlo.or %163, %167 : tensor<2xui32>
      %169 = stablehlo.xor %161, %168 : tensor<2xui32>
      %170 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %171 = stablehlo.reshape %170 : (tensor<1xui32>) -> tensor<ui32>
      %172 = stablehlo.add %161, %169 : tensor<2xui32>
      %173 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %174 = stablehlo.shift_left %169, %173 : tensor<2xui32>
      %175 = stablehlo.constant dense<32> : tensor<ui32>
      %176 = stablehlo.subtract %175, %171 : tensor<ui32>
      %177 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %178 = stablehlo.shift_right_logical %169, %177 : tensor<2xui32>
      %179 = stablehlo.or %174, %178 : tensor<2xui32>
      %180 = stablehlo.xor %172, %179 : tensor<2xui32>
      %181 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %182 = stablehlo.reshape %181 : (tensor<1xui32>) -> tensor<ui32>
      %183 = stablehlo.add %172, %180 : tensor<2xui32>
      %184 = stablehlo.broadcast_in_dim %182, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %185 = stablehlo.shift_left %180, %184 : tensor<2xui32>
      %186 = stablehlo.constant dense<32> : tensor<ui32>
      %187 = stablehlo.subtract %186, %182 : tensor<ui32>
      %188 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %189 = stablehlo.shift_right_logical %180, %188 : tensor<2xui32>
      %190 = stablehlo.or %185, %189 : tensor<2xui32>
      %191 = stablehlo.xor %183, %190 : tensor<2xui32>
      %192 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %193 = stablehlo.reshape %192 : (tensor<1xui32>) -> tensor<ui32>
      %194 = stablehlo.add %183, %191 : tensor<2xui32>
      %195 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %196 = stablehlo.shift_left %191, %195 : tensor<2xui32>
      %197 = stablehlo.constant dense<32> : tensor<ui32>
      %198 = stablehlo.subtract %197, %193 : tensor<ui32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %200 = stablehlo.shift_right_logical %191, %199 : tensor<2xui32>
      %201 = stablehlo.or %196, %200 : tensor<2xui32>
      %202 = stablehlo.xor %194, %201 : tensor<2xui32>
      %203 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %204 = stablehlo.add %194, %203 : tensor<2xui32>
      %205 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %206 = stablehlo.add %202, %205 : tensor<2xui32>
      %207 = stablehlo.constant dense<1> : tensor<i32>
      %208 = stablehlo.add %iterArg_0, %207 : tensor<i32>
      %209 = stablehlo.convert %208 : (tensor<i32>) -> tensor<ui32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %211 = stablehlo.add %206, %210 : tensor<2xui32>
      %212 = stablehlo.constant dense<1> : tensor<i32>
      %213 = stablehlo.add %iterArg, %212 : tensor<i32>
      stablehlo.return %213, %158, %204, %211, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %40 = stablehlo.concatenate %39#2, %39#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %41 = stablehlo.reshape %40 : (tensor<4xui32>) -> tensor<2x2xui32>
    %42 = "stablehlo.slice"(%41) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %43 = stablehlo.reshape %42 : (tensor<1x2xui32>) -> tensor<2xui32>
    %44 = "stablehlo.slice"(%41) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %45 = stablehlo.reshape %44 : (tensor<1x2xui32>) -> tensor<2xui32>
    %46 = stablehlo.constant dense<0> : tensor<1xui32>
    %47 = stablehlo.iota dim = 0 : tensor<5xui32>
    %48 = "stablehlo.slice"(%43) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %49 = stablehlo.reshape %48 : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.slice"(%43) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %51 = stablehlo.reshape %50 : (tensor<1xui32>) -> tensor<ui32>
    %52 = stablehlo.concatenate %47, %46, dim = 0 : (tensor<5xui32>, tensor<1xui32>) -> tensor<6xui32>
    %53 = "stablehlo.slice"(%52) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
    %54 = "stablehlo.slice"(%52) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
    %55 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %56 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %57 = stablehlo.xor %49, %51 : tensor<ui32>
    %58 = stablehlo.constant dense<466688986> : tensor<ui32>
    %59 = stablehlo.xor %57, %58 : tensor<ui32>
    %60 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %61 = stablehlo.add %53, %60 : tensor<3xui32>
    %62 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %63 = stablehlo.add %54, %62 : tensor<3xui32>
    %64 = stablehlo.constant dense<0> : tensor<i32>
    %65 = stablehlo.constant dense<0> : tensor<i32>
    %66:9 = stablehlo.while(%iterArg = %65, %iterArg_0 = %64, %iterArg_1 = %61, %iterArg_2 = %63, %iterArg_3 = %51, %iterArg_4 = %59, %iterArg_5 = %49, %iterArg_6 = %55, %iterArg_7 = %56) : tensor<i32>, tensor<i32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %157 = stablehlo.constant dense<5> : tensor<i32>
      %158 = stablehlo.compare  LT, %iterArg, %157,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %158 : tensor<i1>
    } do {
      %157 = stablehlo.constant dense<1> : tensor<i32>
      %158 = stablehlo.add %iterArg_0, %157 : tensor<i32>
      %159 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %160 = stablehlo.reshape %159 : (tensor<1xui32>) -> tensor<ui32>
      %161 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<3xui32>
      %162 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %163 = stablehlo.shift_left %iterArg_2, %162 : tensor<3xui32>
      %164 = stablehlo.constant dense<32> : tensor<ui32>
      %165 = stablehlo.subtract %164, %160 : tensor<ui32>
      %166 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %167 = stablehlo.shift_right_logical %iterArg_2, %166 : tensor<3xui32>
      %168 = stablehlo.or %163, %167 : tensor<3xui32>
      %169 = stablehlo.xor %161, %168 : tensor<3xui32>
      %170 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %171 = stablehlo.reshape %170 : (tensor<1xui32>) -> tensor<ui32>
      %172 = stablehlo.add %161, %169 : tensor<3xui32>
      %173 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %174 = stablehlo.shift_left %169, %173 : tensor<3xui32>
      %175 = stablehlo.constant dense<32> : tensor<ui32>
      %176 = stablehlo.subtract %175, %171 : tensor<ui32>
      %177 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %178 = stablehlo.shift_right_logical %169, %177 : tensor<3xui32>
      %179 = stablehlo.or %174, %178 : tensor<3xui32>
      %180 = stablehlo.xor %172, %179 : tensor<3xui32>
      %181 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %182 = stablehlo.reshape %181 : (tensor<1xui32>) -> tensor<ui32>
      %183 = stablehlo.add %172, %180 : tensor<3xui32>
      %184 = stablehlo.broadcast_in_dim %182, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %185 = stablehlo.shift_left %180, %184 : tensor<3xui32>
      %186 = stablehlo.constant dense<32> : tensor<ui32>
      %187 = stablehlo.subtract %186, %182 : tensor<ui32>
      %188 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %189 = stablehlo.shift_right_logical %180, %188 : tensor<3xui32>
      %190 = stablehlo.or %185, %189 : tensor<3xui32>
      %191 = stablehlo.xor %183, %190 : tensor<3xui32>
      %192 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %193 = stablehlo.reshape %192 : (tensor<1xui32>) -> tensor<ui32>
      %194 = stablehlo.add %183, %191 : tensor<3xui32>
      %195 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %196 = stablehlo.shift_left %191, %195 : tensor<3xui32>
      %197 = stablehlo.constant dense<32> : tensor<ui32>
      %198 = stablehlo.subtract %197, %193 : tensor<ui32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %200 = stablehlo.shift_right_logical %191, %199 : tensor<3xui32>
      %201 = stablehlo.or %196, %200 : tensor<3xui32>
      %202 = stablehlo.xor %194, %201 : tensor<3xui32>
      %203 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %204 = stablehlo.add %194, %203 : tensor<3xui32>
      %205 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %206 = stablehlo.add %202, %205 : tensor<3xui32>
      %207 = stablehlo.constant dense<1> : tensor<i32>
      %208 = stablehlo.add %iterArg_0, %207 : tensor<i32>
      %209 = stablehlo.convert %208 : (tensor<i32>) -> tensor<ui32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %211 = stablehlo.add %206, %210 : tensor<3xui32>
      %212 = stablehlo.constant dense<1> : tensor<i32>
      %213 = stablehlo.add %iterArg, %212 : tensor<i32>
      stablehlo.return %213, %158, %204, %211, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %67 = stablehlo.concatenate %66#2, %66#3, dim = 0 : (tensor<3xui32>, tensor<3xui32>) -> tensor<6xui32>
    %68 = stablehlo.constant dense<0> : tensor<i32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %70 = "stablehlo.gather"(%67, %69) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<5> : tensor<1xi64>} : (tensor<6xui32>, tensor<1xi32>) -> tensor<5xui32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [1] : (tensor<5xui32>) -> tensor<1x5xui32>
    %72 = stablehlo.iota dim = 0 : tensor<4x1xui32>
    %73 = stablehlo.constant dense<8> : tensor<ui32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<ui32>) -> tensor<4x1xui32>
    %75 = stablehlo.multiply %74, %72 : tensor<4x1xui32>
    %76 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<1x5xui32>) -> tensor<4x5xui32>
    %77 = stablehlo.broadcast_in_dim %75, dims = [0, 1] : (tensor<4x1xui32>) -> tensor<4x5xui32>
    %78 = stablehlo.shift_right_logical %76, %77 : tensor<4x5xui32>
    %79 = stablehlo.constant dense<255> : tensor<ui32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<ui32>) -> tensor<4x5xui32>
    %81 = stablehlo.and %80, %78 : tensor<4x5xui32>
    %82 = stablehlo.transpose %81, dims = [1, 0] : (tensor<4x5xui32>) -> tensor<5x4xui32>
    %83 = stablehlo.reshape %82 : (tensor<5x4xui32>) -> tensor<20xui32>
    %84 = stablehlo.convert %83 : (tensor<20xui32>) -> tensor<20xui8>
    %85 = stablehlo.reshape %84 : (tensor<20xui8>) -> tensor<5x4xui8>
    %86 = stablehlo.constant dense<0> : tensor<1xui32>
    %87 = stablehlo.iota dim = 0 : tensor<5xui32>
    %88 = "stablehlo.slice"(%45) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %89 = stablehlo.reshape %88 : (tensor<1xui32>) -> tensor<ui32>
    %90 = "stablehlo.slice"(%45) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %91 = stablehlo.reshape %90 : (tensor<1xui32>) -> tensor<ui32>
    %92 = stablehlo.concatenate %87, %86, dim = 0 : (tensor<5xui32>, tensor<1xui32>) -> tensor<6xui32>
    %93 = "stablehlo.slice"(%92) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
    %94 = "stablehlo.slice"(%92) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
    %95 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %96 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %97 = stablehlo.xor %89, %91 : tensor<ui32>
    %98 = stablehlo.constant dense<466688986> : tensor<ui32>
    %99 = stablehlo.xor %97, %98 : tensor<ui32>
    %100 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %101 = stablehlo.add %93, %100 : tensor<3xui32>
    %102 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %103 = stablehlo.add %94, %102 : tensor<3xui32>
    %104 = stablehlo.constant dense<0> : tensor<i32>
    %105 = stablehlo.constant dense<0> : tensor<i32>
    %106:9 = stablehlo.while(%iterArg = %105, %iterArg_0 = %104, %iterArg_1 = %101, %iterArg_2 = %103, %iterArg_3 = %91, %iterArg_4 = %99, %iterArg_5 = %89, %iterArg_6 = %95, %iterArg_7 = %96) : tensor<i32>, tensor<i32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %157 = stablehlo.constant dense<5> : tensor<i32>
      %158 = stablehlo.compare  LT, %iterArg, %157,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %158 : tensor<i1>
    } do {
      %157 = stablehlo.constant dense<1> : tensor<i32>
      %158 = stablehlo.add %iterArg_0, %157 : tensor<i32>
      %159 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %160 = stablehlo.reshape %159 : (tensor<1xui32>) -> tensor<ui32>
      %161 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<3xui32>
      %162 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %163 = stablehlo.shift_left %iterArg_2, %162 : tensor<3xui32>
      %164 = stablehlo.constant dense<32> : tensor<ui32>
      %165 = stablehlo.subtract %164, %160 : tensor<ui32>
      %166 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %167 = stablehlo.shift_right_logical %iterArg_2, %166 : tensor<3xui32>
      %168 = stablehlo.or %163, %167 : tensor<3xui32>
      %169 = stablehlo.xor %161, %168 : tensor<3xui32>
      %170 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %171 = stablehlo.reshape %170 : (tensor<1xui32>) -> tensor<ui32>
      %172 = stablehlo.add %161, %169 : tensor<3xui32>
      %173 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %174 = stablehlo.shift_left %169, %173 : tensor<3xui32>
      %175 = stablehlo.constant dense<32> : tensor<ui32>
      %176 = stablehlo.subtract %175, %171 : tensor<ui32>
      %177 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %178 = stablehlo.shift_right_logical %169, %177 : tensor<3xui32>
      %179 = stablehlo.or %174, %178 : tensor<3xui32>
      %180 = stablehlo.xor %172, %179 : tensor<3xui32>
      %181 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %182 = stablehlo.reshape %181 : (tensor<1xui32>) -> tensor<ui32>
      %183 = stablehlo.add %172, %180 : tensor<3xui32>
      %184 = stablehlo.broadcast_in_dim %182, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %185 = stablehlo.shift_left %180, %184 : tensor<3xui32>
      %186 = stablehlo.constant dense<32> : tensor<ui32>
      %187 = stablehlo.subtract %186, %182 : tensor<ui32>
      %188 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %189 = stablehlo.shift_right_logical %180, %188 : tensor<3xui32>
      %190 = stablehlo.or %185, %189 : tensor<3xui32>
      %191 = stablehlo.xor %183, %190 : tensor<3xui32>
      %192 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %193 = stablehlo.reshape %192 : (tensor<1xui32>) -> tensor<ui32>
      %194 = stablehlo.add %183, %191 : tensor<3xui32>
      %195 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %196 = stablehlo.shift_left %191, %195 : tensor<3xui32>
      %197 = stablehlo.constant dense<32> : tensor<ui32>
      %198 = stablehlo.subtract %197, %193 : tensor<ui32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %200 = stablehlo.shift_right_logical %191, %199 : tensor<3xui32>
      %201 = stablehlo.or %196, %200 : tensor<3xui32>
      %202 = stablehlo.xor %194, %201 : tensor<3xui32>
      %203 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %204 = stablehlo.add %194, %203 : tensor<3xui32>
      %205 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %206 = stablehlo.add %202, %205 : tensor<3xui32>
      %207 = stablehlo.constant dense<1> : tensor<i32>
      %208 = stablehlo.add %iterArg_0, %207 : tensor<i32>
      %209 = stablehlo.convert %208 : (tensor<i32>) -> tensor<ui32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [] : (tensor<ui32>) -> tensor<3xui32>
      %211 = stablehlo.add %206, %210 : tensor<3xui32>
      %212 = stablehlo.constant dense<1> : tensor<i32>
      %213 = stablehlo.add %iterArg, %212 : tensor<i32>
      stablehlo.return %213, %158, %204, %211, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %107 = stablehlo.concatenate %106#2, %106#3, dim = 0 : (tensor<3xui32>, tensor<3xui32>) -> tensor<6xui32>
    %108 = stablehlo.constant dense<0> : tensor<i32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %110 = "stablehlo.gather"(%107, %109) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<5> : tensor<1xi64>} : (tensor<6xui32>, tensor<1xi32>) -> tensor<5xui32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [1] : (tensor<5xui32>) -> tensor<1x5xui32>
    %112 = stablehlo.iota dim = 0 : tensor<4x1xui32>
    %113 = stablehlo.constant dense<8> : tensor<ui32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [] : (tensor<ui32>) -> tensor<4x1xui32>
    %115 = stablehlo.multiply %114, %112 : tensor<4x1xui32>
    %116 = stablehlo.broadcast_in_dim %111, dims = [0, 1] : (tensor<1x5xui32>) -> tensor<4x5xui32>
    %117 = stablehlo.broadcast_in_dim %115, dims = [0, 1] : (tensor<4x1xui32>) -> tensor<4x5xui32>
    %118 = stablehlo.shift_right_logical %116, %117 : tensor<4x5xui32>
    %119 = stablehlo.constant dense<255> : tensor<ui32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<ui32>) -> tensor<4x5xui32>
    %121 = stablehlo.and %120, %118 : tensor<4x5xui32>
    %122 = stablehlo.transpose %121, dims = [1, 0] : (tensor<4x5xui32>) -> tensor<5x4xui32>
    %123 = stablehlo.reshape %122 : (tensor<5x4xui32>) -> tensor<20xui32>
    %124 = stablehlo.convert %123 : (tensor<20xui32>) -> tensor<20xui8>
    %125 = stablehlo.reshape %124 : (tensor<20xui8>) -> tensor<5x4xui8>
    %126 = stablehlo.subtract %20, %19 : tensor<1x1xi8>
    %127 = stablehlo.convert %126 : (tensor<1x1xi8>) -> tensor<1x1xui8>
    %128 = stablehlo.compare  LE, %20, %19,  SIGNED : (tensor<1x1xi8>, tensor<1x1xi8>) -> tensor<1x1xi1>
    %129 = stablehlo.constant dense<1> : tensor<ui8>
    %130 = stablehlo.broadcast_in_dim %129, dims = [] : (tensor<ui8>) -> tensor<1x1xui8>
    %131 = stablehlo.select %128, %130, %127 : tensor<1x1xi1>, tensor<1x1xui8>
    %132 = stablehlo.compare  GT, %20, %19,  SIGNED : (tensor<1x1xi8>, tensor<1x1xi8>) -> tensor<1x1xi1>
    %133 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %134 = stablehlo.and %133, %132 : tensor<1x1xi1>
    %135 = stablehlo.constant dense<1> : tensor<ui8>
    %136 = stablehlo.broadcast_in_dim %135, dims = [] : (tensor<ui8>) -> tensor<1x1xui8>
    %137 = stablehlo.add %131, %136 : tensor<1x1xui8>
    %138 = stablehlo.select %134, %137, %131 : tensor<1x1xi1>, tensor<1x1xui8>
    %139 = stablehlo.constant dense<16> : tensor<ui8>
    %140 = stablehlo.broadcast_in_dim %139, dims = [] : (tensor<ui8>) -> tensor<1x1xui8>
    %141 = stablehlo.remainder %140, %138 : tensor<1x1xui8>
    %142 = stablehlo.multiply %141, %141 : tensor<1x1xui8>
    %143 = stablehlo.remainder %142, %138 : tensor<1x1xui8>
    %144 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<1x1xui8>) -> tensor<5x4xui8>
    %145 = stablehlo.remainder %85, %144 : tensor<5x4xui8>
    %146 = stablehlo.broadcast_in_dim %143, dims = [0, 1] : (tensor<1x1xui8>) -> tensor<5x4xui8>
    %147 = stablehlo.multiply %145, %146 : tensor<5x4xui8>
    %148 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<1x1xui8>) -> tensor<5x4xui8>
    %149 = stablehlo.remainder %125, %148 : tensor<5x4xui8>
    %150 = stablehlo.add %147, %149 : tensor<5x4xui8>
    %151 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<1x1xui8>) -> tensor<5x4xui8>
    %152 = stablehlo.remainder %150, %151 : tensor<5x4xui8>
    %153 = stablehlo.convert %152 : (tensor<5x4xui8>) -> tensor<5x4xi8>
    %154 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x1xi8>) -> tensor<5x4xi8>
    %155 = stablehlo.add %154, %153 : tensor<5x4xi8>
    %156 = stablehlo.custom_call @check.eq(%155, %1) : (tensor<5x4xi8>, tensor<5x4xi8>) -> tensor<i1>
    return %156 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<5x4xi8> {
    %0 = stablehlo.constant dense<[[1, -2, -3, -5], [-3, 1, -5, -1], [-5, 0, -2, -1], [-1, -4, 0, -2], [-2, -1, -4, -3]]> : tensor<5x4xi8>
    return %0 : tensor<5x4xi8>
  }
  func.func private @clip(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<i8>) -> tensor<i8> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i8>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i8>
    return %1 : tensor<i8>
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
