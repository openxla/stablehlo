// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<32xi16>
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
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i16>) -> tensor<1xi16>
    %20 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i16>) -> tensor<1xi16>
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
      %145 = stablehlo.constant dense<5> : tensor<i32>
      %146 = stablehlo.compare  LT, %iterArg, %145,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %146 : tensor<i1>
    } do {
      %145 = stablehlo.constant dense<1> : tensor<i32>
      %146 = stablehlo.add %iterArg_0, %145 : tensor<i32>
      %147 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %148 = stablehlo.reshape %147 : (tensor<1xui32>) -> tensor<ui32>
      %149 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %150 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %151 = stablehlo.shift_left %iterArg_2, %150 : tensor<2xui32>
      %152 = stablehlo.constant dense<32> : tensor<ui32>
      %153 = stablehlo.subtract %152, %148 : tensor<ui32>
      %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %155 = stablehlo.shift_right_logical %iterArg_2, %154 : tensor<2xui32>
      %156 = stablehlo.or %151, %155 : tensor<2xui32>
      %157 = stablehlo.xor %149, %156 : tensor<2xui32>
      %158 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %159 = stablehlo.reshape %158 : (tensor<1xui32>) -> tensor<ui32>
      %160 = stablehlo.add %149, %157 : tensor<2xui32>
      %161 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %162 = stablehlo.shift_left %157, %161 : tensor<2xui32>
      %163 = stablehlo.constant dense<32> : tensor<ui32>
      %164 = stablehlo.subtract %163, %159 : tensor<ui32>
      %165 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %166 = stablehlo.shift_right_logical %157, %165 : tensor<2xui32>
      %167 = stablehlo.or %162, %166 : tensor<2xui32>
      %168 = stablehlo.xor %160, %167 : tensor<2xui32>
      %169 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %170 = stablehlo.reshape %169 : (tensor<1xui32>) -> tensor<ui32>
      %171 = stablehlo.add %160, %168 : tensor<2xui32>
      %172 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %173 = stablehlo.shift_left %168, %172 : tensor<2xui32>
      %174 = stablehlo.constant dense<32> : tensor<ui32>
      %175 = stablehlo.subtract %174, %170 : tensor<ui32>
      %176 = stablehlo.broadcast_in_dim %175, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %177 = stablehlo.shift_right_logical %168, %176 : tensor<2xui32>
      %178 = stablehlo.or %173, %177 : tensor<2xui32>
      %179 = stablehlo.xor %171, %178 : tensor<2xui32>
      %180 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %181 = stablehlo.reshape %180 : (tensor<1xui32>) -> tensor<ui32>
      %182 = stablehlo.add %171, %179 : tensor<2xui32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %184 = stablehlo.shift_left %179, %183 : tensor<2xui32>
      %185 = stablehlo.constant dense<32> : tensor<ui32>
      %186 = stablehlo.subtract %185, %181 : tensor<ui32>
      %187 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %188 = stablehlo.shift_right_logical %179, %187 : tensor<2xui32>
      %189 = stablehlo.or %184, %188 : tensor<2xui32>
      %190 = stablehlo.xor %182, %189 : tensor<2xui32>
      %191 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %192 = stablehlo.add %182, %191 : tensor<2xui32>
      %193 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %194 = stablehlo.add %190, %193 : tensor<2xui32>
      %195 = stablehlo.constant dense<1> : tensor<i32>
      %196 = stablehlo.add %iterArg_0, %195 : tensor<i32>
      %197 = stablehlo.convert %196 : (tensor<i32>) -> tensor<ui32>
      %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %199 = stablehlo.add %194, %198 : tensor<2xui32>
      %200 = stablehlo.constant dense<1> : tensor<i32>
      %201 = stablehlo.add %iterArg, %200 : tensor<i32>
      stablehlo.return %201, %146, %192, %199, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %40 = stablehlo.concatenate %39#2, %39#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %41 = stablehlo.reshape %40 : (tensor<4xui32>) -> tensor<2x2xui32>
    %42 = "stablehlo.slice"(%41) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %43 = stablehlo.reshape %42 : (tensor<1x2xui32>) -> tensor<2xui32>
    %44 = "stablehlo.slice"(%41) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %45 = stablehlo.reshape %44 : (tensor<1x2xui32>) -> tensor<2xui32>
    %46 = stablehlo.iota dim = 0 : tensor<16xui32>
    %47 = "stablehlo.slice"(%43) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %48 = stablehlo.reshape %47 : (tensor<1xui32>) -> tensor<ui32>
    %49 = "stablehlo.slice"(%43) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %50 = stablehlo.reshape %49 : (tensor<1xui32>) -> tensor<ui32>
    %51 = "stablehlo.slice"(%46) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<16xui32>) -> tensor<8xui32>
    %52 = "stablehlo.slice"(%46) {limit_indices = dense<16> : tensor<1xi64>, start_indices = dense<8> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<16xui32>) -> tensor<8xui32>
    %53 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %54 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %55 = stablehlo.xor %48, %50 : tensor<ui32>
    %56 = stablehlo.constant dense<466688986> : tensor<ui32>
    %57 = stablehlo.xor %55, %56 : tensor<ui32>
    %58 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %59 = stablehlo.add %51, %58 : tensor<8xui32>
    %60 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %61 = stablehlo.add %52, %60 : tensor<8xui32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63 = stablehlo.constant dense<0> : tensor<i32>
    %64:9 = stablehlo.while(%iterArg = %63, %iterArg_0 = %62, %iterArg_1 = %59, %iterArg_2 = %61, %iterArg_3 = %50, %iterArg_4 = %57, %iterArg_5 = %48, %iterArg_6 = %53, %iterArg_7 = %54) : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %145 = stablehlo.constant dense<5> : tensor<i32>
      %146 = stablehlo.compare  LT, %iterArg, %145,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %146 : tensor<i1>
    } do {
      %145 = stablehlo.constant dense<1> : tensor<i32>
      %146 = stablehlo.add %iterArg_0, %145 : tensor<i32>
      %147 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %148 = stablehlo.reshape %147 : (tensor<1xui32>) -> tensor<ui32>
      %149 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<8xui32>
      %150 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %151 = stablehlo.shift_left %iterArg_2, %150 : tensor<8xui32>
      %152 = stablehlo.constant dense<32> : tensor<ui32>
      %153 = stablehlo.subtract %152, %148 : tensor<ui32>
      %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %155 = stablehlo.shift_right_logical %iterArg_2, %154 : tensor<8xui32>
      %156 = stablehlo.or %151, %155 : tensor<8xui32>
      %157 = stablehlo.xor %149, %156 : tensor<8xui32>
      %158 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %159 = stablehlo.reshape %158 : (tensor<1xui32>) -> tensor<ui32>
      %160 = stablehlo.add %149, %157 : tensor<8xui32>
      %161 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %162 = stablehlo.shift_left %157, %161 : tensor<8xui32>
      %163 = stablehlo.constant dense<32> : tensor<ui32>
      %164 = stablehlo.subtract %163, %159 : tensor<ui32>
      %165 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %166 = stablehlo.shift_right_logical %157, %165 : tensor<8xui32>
      %167 = stablehlo.or %162, %166 : tensor<8xui32>
      %168 = stablehlo.xor %160, %167 : tensor<8xui32>
      %169 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %170 = stablehlo.reshape %169 : (tensor<1xui32>) -> tensor<ui32>
      %171 = stablehlo.add %160, %168 : tensor<8xui32>
      %172 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %173 = stablehlo.shift_left %168, %172 : tensor<8xui32>
      %174 = stablehlo.constant dense<32> : tensor<ui32>
      %175 = stablehlo.subtract %174, %170 : tensor<ui32>
      %176 = stablehlo.broadcast_in_dim %175, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %177 = stablehlo.shift_right_logical %168, %176 : tensor<8xui32>
      %178 = stablehlo.or %173, %177 : tensor<8xui32>
      %179 = stablehlo.xor %171, %178 : tensor<8xui32>
      %180 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %181 = stablehlo.reshape %180 : (tensor<1xui32>) -> tensor<ui32>
      %182 = stablehlo.add %171, %179 : tensor<8xui32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %184 = stablehlo.shift_left %179, %183 : tensor<8xui32>
      %185 = stablehlo.constant dense<32> : tensor<ui32>
      %186 = stablehlo.subtract %185, %181 : tensor<ui32>
      %187 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %188 = stablehlo.shift_right_logical %179, %187 : tensor<8xui32>
      %189 = stablehlo.or %184, %188 : tensor<8xui32>
      %190 = stablehlo.xor %182, %189 : tensor<8xui32>
      %191 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %192 = stablehlo.add %182, %191 : tensor<8xui32>
      %193 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %194 = stablehlo.add %190, %193 : tensor<8xui32>
      %195 = stablehlo.constant dense<1> : tensor<i32>
      %196 = stablehlo.add %iterArg_0, %195 : tensor<i32>
      %197 = stablehlo.convert %196 : (tensor<i32>) -> tensor<ui32>
      %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %199 = stablehlo.add %194, %198 : tensor<8xui32>
      %200 = stablehlo.constant dense<1> : tensor<i32>
      %201 = stablehlo.add %iterArg, %200 : tensor<i32>
      stablehlo.return %201, %146, %192, %199, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %65 = stablehlo.concatenate %64#2, %64#3, dim = 0 : (tensor<8xui32>, tensor<8xui32>) -> tensor<16xui32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [1] : (tensor<16xui32>) -> tensor<1x16xui32>
    %67 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %68 = stablehlo.constant dense<16> : tensor<ui32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %70 = stablehlo.multiply %69, %67 : tensor<2x1xui32>
    %71 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<1x16xui32>) -> tensor<2x16xui32>
    %72 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x16xui32>
    %73 = stablehlo.shift_right_logical %71, %72 : tensor<2x16xui32>
    %74 = stablehlo.constant dense<65535> : tensor<ui32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<ui32>) -> tensor<2x16xui32>
    %76 = stablehlo.and %75, %73 : tensor<2x16xui32>
    %77 = stablehlo.transpose %76, dims = [1, 0] : (tensor<2x16xui32>) -> tensor<16x2xui32>
    %78 = stablehlo.reshape %77 : (tensor<16x2xui32>) -> tensor<32xui32>
    %79 = stablehlo.convert %78 : (tensor<32xui32>) -> tensor<32xui16>
    %80 = stablehlo.iota dim = 0 : tensor<16xui32>
    %81 = "stablehlo.slice"(%45) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %82 = stablehlo.reshape %81 : (tensor<1xui32>) -> tensor<ui32>
    %83 = "stablehlo.slice"(%45) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %84 = stablehlo.reshape %83 : (tensor<1xui32>) -> tensor<ui32>
    %85 = "stablehlo.slice"(%80) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<16xui32>) -> tensor<8xui32>
    %86 = "stablehlo.slice"(%80) {limit_indices = dense<16> : tensor<1xi64>, start_indices = dense<8> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<16xui32>) -> tensor<8xui32>
    %87 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %88 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %89 = stablehlo.xor %82, %84 : tensor<ui32>
    %90 = stablehlo.constant dense<466688986> : tensor<ui32>
    %91 = stablehlo.xor %89, %90 : tensor<ui32>
    %92 = stablehlo.broadcast_in_dim %82, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %93 = stablehlo.add %85, %92 : tensor<8xui32>
    %94 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %95 = stablehlo.add %86, %94 : tensor<8xui32>
    %96 = stablehlo.constant dense<0> : tensor<i32>
    %97 = stablehlo.constant dense<0> : tensor<i32>
    %98:9 = stablehlo.while(%iterArg = %97, %iterArg_0 = %96, %iterArg_1 = %93, %iterArg_2 = %95, %iterArg_3 = %84, %iterArg_4 = %91, %iterArg_5 = %82, %iterArg_6 = %87, %iterArg_7 = %88) : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %145 = stablehlo.constant dense<5> : tensor<i32>
      %146 = stablehlo.compare  LT, %iterArg, %145,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %146 : tensor<i1>
    } do {
      %145 = stablehlo.constant dense<1> : tensor<i32>
      %146 = stablehlo.add %iterArg_0, %145 : tensor<i32>
      %147 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %148 = stablehlo.reshape %147 : (tensor<1xui32>) -> tensor<ui32>
      %149 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<8xui32>
      %150 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %151 = stablehlo.shift_left %iterArg_2, %150 : tensor<8xui32>
      %152 = stablehlo.constant dense<32> : tensor<ui32>
      %153 = stablehlo.subtract %152, %148 : tensor<ui32>
      %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %155 = stablehlo.shift_right_logical %iterArg_2, %154 : tensor<8xui32>
      %156 = stablehlo.or %151, %155 : tensor<8xui32>
      %157 = stablehlo.xor %149, %156 : tensor<8xui32>
      %158 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %159 = stablehlo.reshape %158 : (tensor<1xui32>) -> tensor<ui32>
      %160 = stablehlo.add %149, %157 : tensor<8xui32>
      %161 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %162 = stablehlo.shift_left %157, %161 : tensor<8xui32>
      %163 = stablehlo.constant dense<32> : tensor<ui32>
      %164 = stablehlo.subtract %163, %159 : tensor<ui32>
      %165 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %166 = stablehlo.shift_right_logical %157, %165 : tensor<8xui32>
      %167 = stablehlo.or %162, %166 : tensor<8xui32>
      %168 = stablehlo.xor %160, %167 : tensor<8xui32>
      %169 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %170 = stablehlo.reshape %169 : (tensor<1xui32>) -> tensor<ui32>
      %171 = stablehlo.add %160, %168 : tensor<8xui32>
      %172 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %173 = stablehlo.shift_left %168, %172 : tensor<8xui32>
      %174 = stablehlo.constant dense<32> : tensor<ui32>
      %175 = stablehlo.subtract %174, %170 : tensor<ui32>
      %176 = stablehlo.broadcast_in_dim %175, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %177 = stablehlo.shift_right_logical %168, %176 : tensor<8xui32>
      %178 = stablehlo.or %173, %177 : tensor<8xui32>
      %179 = stablehlo.xor %171, %178 : tensor<8xui32>
      %180 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %181 = stablehlo.reshape %180 : (tensor<1xui32>) -> tensor<ui32>
      %182 = stablehlo.add %171, %179 : tensor<8xui32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %184 = stablehlo.shift_left %179, %183 : tensor<8xui32>
      %185 = stablehlo.constant dense<32> : tensor<ui32>
      %186 = stablehlo.subtract %185, %181 : tensor<ui32>
      %187 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %188 = stablehlo.shift_right_logical %179, %187 : tensor<8xui32>
      %189 = stablehlo.or %184, %188 : tensor<8xui32>
      %190 = stablehlo.xor %182, %189 : tensor<8xui32>
      %191 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %192 = stablehlo.add %182, %191 : tensor<8xui32>
      %193 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %194 = stablehlo.add %190, %193 : tensor<8xui32>
      %195 = stablehlo.constant dense<1> : tensor<i32>
      %196 = stablehlo.add %iterArg_0, %195 : tensor<i32>
      %197 = stablehlo.convert %196 : (tensor<i32>) -> tensor<ui32>
      %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %199 = stablehlo.add %194, %198 : tensor<8xui32>
      %200 = stablehlo.constant dense<1> : tensor<i32>
      %201 = stablehlo.add %iterArg, %200 : tensor<i32>
      stablehlo.return %201, %146, %192, %199, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %99 = stablehlo.concatenate %98#2, %98#3, dim = 0 : (tensor<8xui32>, tensor<8xui32>) -> tensor<16xui32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [1] : (tensor<16xui32>) -> tensor<1x16xui32>
    %101 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %102 = stablehlo.constant dense<16> : tensor<ui32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %104 = stablehlo.multiply %103, %101 : tensor<2x1xui32>
    %105 = stablehlo.broadcast_in_dim %100, dims = [0, 1] : (tensor<1x16xui32>) -> tensor<2x16xui32>
    %106 = stablehlo.broadcast_in_dim %104, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x16xui32>
    %107 = stablehlo.shift_right_logical %105, %106 : tensor<2x16xui32>
    %108 = stablehlo.constant dense<65535> : tensor<ui32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<ui32>) -> tensor<2x16xui32>
    %110 = stablehlo.and %109, %107 : tensor<2x16xui32>
    %111 = stablehlo.transpose %110, dims = [1, 0] : (tensor<2x16xui32>) -> tensor<16x2xui32>
    %112 = stablehlo.reshape %111 : (tensor<16x2xui32>) -> tensor<32xui32>
    %113 = stablehlo.convert %112 : (tensor<32xui32>) -> tensor<32xui16>
    %114 = stablehlo.subtract %20, %19 : tensor<1xi16>
    %115 = stablehlo.convert %114 : (tensor<1xi16>) -> tensor<1xui16>
    %116 = stablehlo.compare  LE, %20, %19,  SIGNED : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi1>
    %117 = stablehlo.constant dense<1> : tensor<ui16>
    %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<ui16>) -> tensor<1xui16>
    %119 = stablehlo.select %116, %118, %115 : tensor<1xi1>, tensor<1xui16>
    %120 = stablehlo.compare  GT, %20, %19,  SIGNED : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi1>
    %121 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %122 = stablehlo.and %121, %120 : tensor<1xi1>
    %123 = stablehlo.constant dense<1> : tensor<ui16>
    %124 = stablehlo.broadcast_in_dim %123, dims = [] : (tensor<ui16>) -> tensor<1xui16>
    %125 = stablehlo.add %119, %124 : tensor<1xui16>
    %126 = stablehlo.select %122, %125, %119 : tensor<1xi1>, tensor<1xui16>
    %127 = stablehlo.constant dense<256> : tensor<ui16>
    %128 = stablehlo.broadcast_in_dim %127, dims = [] : (tensor<ui16>) -> tensor<1xui16>
    %129 = stablehlo.remainder %128, %126 : tensor<1xui16>
    %130 = stablehlo.multiply %129, %129 : tensor<1xui16>
    %131 = stablehlo.remainder %130, %126 : tensor<1xui16>
    %132 = stablehlo.broadcast_in_dim %126, dims = [0] : (tensor<1xui16>) -> tensor<32xui16>
    %133 = stablehlo.remainder %79, %132 : tensor<32xui16>
    %134 = stablehlo.broadcast_in_dim %131, dims = [0] : (tensor<1xui16>) -> tensor<32xui16>
    %135 = stablehlo.multiply %133, %134 : tensor<32xui16>
    %136 = stablehlo.broadcast_in_dim %126, dims = [0] : (tensor<1xui16>) -> tensor<32xui16>
    %137 = stablehlo.remainder %113, %136 : tensor<32xui16>
    %138 = stablehlo.add %135, %137 : tensor<32xui16>
    %139 = stablehlo.broadcast_in_dim %126, dims = [0] : (tensor<1xui16>) -> tensor<32xui16>
    %140 = stablehlo.remainder %138, %139 : tensor<32xui16>
    %141 = stablehlo.convert %140 : (tensor<32xui16>) -> tensor<32xi16>
    %142 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1xi16>) -> tensor<32xi16>
    %143 = stablehlo.add %142, %141 : tensor<32xi16>
    %144 = stablehlo.custom_call @check.eq(%143, %1) : (tensor<32xi16>, tensor<32xi16>) -> tensor<i1>
    return %144 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<32xi16> {
    %0 = stablehlo.constant dense<[0, -4, 4, -2, -5, 3, -1, -1, -5, -3, -5, 3, -3, 1, 2, 4, 1, -2, -4, -4, 0, -2, -4, 3, 2, -3, 0, -1, 3, 0, -1, -4]> : tensor<32xi16>
    return %0 : tensor<32xi16>
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
