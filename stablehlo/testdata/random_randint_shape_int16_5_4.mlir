// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<5x4xi16>
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
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i16>) -> tensor<1x1xi16>
    %20 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i16>) -> tensor<1x1xi16>
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
      %147 = stablehlo.constant dense<5> : tensor<i32>
      %148 = stablehlo.compare  LT, %iterArg, %147,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %148 : tensor<i1>
    } do {
      %147 = stablehlo.constant dense<1> : tensor<i32>
      %148 = stablehlo.add %iterArg_0, %147 : tensor<i32>
      %149 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %150 = stablehlo.reshape %149 : (tensor<1xui32>) -> tensor<ui32>
      %151 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %152 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %153 = stablehlo.shift_left %iterArg_2, %152 : tensor<2xui32>
      %154 = stablehlo.constant dense<32> : tensor<ui32>
      %155 = stablehlo.subtract %154, %150 : tensor<ui32>
      %156 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %157 = stablehlo.shift_right_logical %iterArg_2, %156 : tensor<2xui32>
      %158 = stablehlo.or %153, %157 : tensor<2xui32>
      %159 = stablehlo.xor %151, %158 : tensor<2xui32>
      %160 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %161 = stablehlo.reshape %160 : (tensor<1xui32>) -> tensor<ui32>
      %162 = stablehlo.add %151, %159 : tensor<2xui32>
      %163 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %164 = stablehlo.shift_left %159, %163 : tensor<2xui32>
      %165 = stablehlo.constant dense<32> : tensor<ui32>
      %166 = stablehlo.subtract %165, %161 : tensor<ui32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %168 = stablehlo.shift_right_logical %159, %167 : tensor<2xui32>
      %169 = stablehlo.or %164, %168 : tensor<2xui32>
      %170 = stablehlo.xor %162, %169 : tensor<2xui32>
      %171 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %172 = stablehlo.reshape %171 : (tensor<1xui32>) -> tensor<ui32>
      %173 = stablehlo.add %162, %170 : tensor<2xui32>
      %174 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %175 = stablehlo.shift_left %170, %174 : tensor<2xui32>
      %176 = stablehlo.constant dense<32> : tensor<ui32>
      %177 = stablehlo.subtract %176, %172 : tensor<ui32>
      %178 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %179 = stablehlo.shift_right_logical %170, %178 : tensor<2xui32>
      %180 = stablehlo.or %175, %179 : tensor<2xui32>
      %181 = stablehlo.xor %173, %180 : tensor<2xui32>
      %182 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %183 = stablehlo.reshape %182 : (tensor<1xui32>) -> tensor<ui32>
      %184 = stablehlo.add %173, %181 : tensor<2xui32>
      %185 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %186 = stablehlo.shift_left %181, %185 : tensor<2xui32>
      %187 = stablehlo.constant dense<32> : tensor<ui32>
      %188 = stablehlo.subtract %187, %183 : tensor<ui32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %190 = stablehlo.shift_right_logical %181, %189 : tensor<2xui32>
      %191 = stablehlo.or %186, %190 : tensor<2xui32>
      %192 = stablehlo.xor %184, %191 : tensor<2xui32>
      %193 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %194 = stablehlo.add %184, %193 : tensor<2xui32>
      %195 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %196 = stablehlo.add %192, %195 : tensor<2xui32>
      %197 = stablehlo.constant dense<1> : tensor<i32>
      %198 = stablehlo.add %iterArg_0, %197 : tensor<i32>
      %199 = stablehlo.convert %198 : (tensor<i32>) -> tensor<ui32>
      %200 = stablehlo.broadcast_in_dim %199, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %201 = stablehlo.add %196, %200 : tensor<2xui32>
      %202 = stablehlo.constant dense<1> : tensor<i32>
      %203 = stablehlo.add %iterArg, %202 : tensor<i32>
      stablehlo.return %203, %148, %194, %201, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %40 = stablehlo.concatenate %39#2, %39#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %41 = stablehlo.reshape %40 : (tensor<4xui32>) -> tensor<2x2xui32>
    %42 = "stablehlo.slice"(%41) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %43 = stablehlo.reshape %42 : (tensor<1x2xui32>) -> tensor<2xui32>
    %44 = "stablehlo.slice"(%41) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %45 = stablehlo.reshape %44 : (tensor<1x2xui32>) -> tensor<2xui32>
    %46 = stablehlo.iota dim = 0 : tensor<10xui32>
    %47 = "stablehlo.slice"(%43) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %48 = stablehlo.reshape %47 : (tensor<1xui32>) -> tensor<ui32>
    %49 = "stablehlo.slice"(%43) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %50 = stablehlo.reshape %49 : (tensor<1xui32>) -> tensor<ui32>
    %51 = "stablehlo.slice"(%46) {limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<10xui32>) -> tensor<5xui32>
    %52 = "stablehlo.slice"(%46) {limit_indices = dense<10> : tensor<1xi64>, start_indices = dense<5> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<10xui32>) -> tensor<5xui32>
    %53 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %54 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %55 = stablehlo.xor %48, %50 : tensor<ui32>
    %56 = stablehlo.constant dense<466688986> : tensor<ui32>
    %57 = stablehlo.xor %55, %56 : tensor<ui32>
    %58 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<ui32>) -> tensor<5xui32>
    %59 = stablehlo.add %51, %58 : tensor<5xui32>
    %60 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<ui32>) -> tensor<5xui32>
    %61 = stablehlo.add %52, %60 : tensor<5xui32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63 = stablehlo.constant dense<0> : tensor<i32>
    %64:9 = stablehlo.while(%iterArg = %63, %iterArg_0 = %62, %iterArg_1 = %59, %iterArg_2 = %61, %iterArg_3 = %50, %iterArg_4 = %57, %iterArg_5 = %48, %iterArg_6 = %53, %iterArg_7 = %54) : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %147 = stablehlo.constant dense<5> : tensor<i32>
      %148 = stablehlo.compare  LT, %iterArg, %147,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %148 : tensor<i1>
    } do {
      %147 = stablehlo.constant dense<1> : tensor<i32>
      %148 = stablehlo.add %iterArg_0, %147 : tensor<i32>
      %149 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %150 = stablehlo.reshape %149 : (tensor<1xui32>) -> tensor<ui32>
      %151 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<5xui32>
      %152 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %153 = stablehlo.shift_left %iterArg_2, %152 : tensor<5xui32>
      %154 = stablehlo.constant dense<32> : tensor<ui32>
      %155 = stablehlo.subtract %154, %150 : tensor<ui32>
      %156 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %157 = stablehlo.shift_right_logical %iterArg_2, %156 : tensor<5xui32>
      %158 = stablehlo.or %153, %157 : tensor<5xui32>
      %159 = stablehlo.xor %151, %158 : tensor<5xui32>
      %160 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %161 = stablehlo.reshape %160 : (tensor<1xui32>) -> tensor<ui32>
      %162 = stablehlo.add %151, %159 : tensor<5xui32>
      %163 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %164 = stablehlo.shift_left %159, %163 : tensor<5xui32>
      %165 = stablehlo.constant dense<32> : tensor<ui32>
      %166 = stablehlo.subtract %165, %161 : tensor<ui32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %168 = stablehlo.shift_right_logical %159, %167 : tensor<5xui32>
      %169 = stablehlo.or %164, %168 : tensor<5xui32>
      %170 = stablehlo.xor %162, %169 : tensor<5xui32>
      %171 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %172 = stablehlo.reshape %171 : (tensor<1xui32>) -> tensor<ui32>
      %173 = stablehlo.add %162, %170 : tensor<5xui32>
      %174 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %175 = stablehlo.shift_left %170, %174 : tensor<5xui32>
      %176 = stablehlo.constant dense<32> : tensor<ui32>
      %177 = stablehlo.subtract %176, %172 : tensor<ui32>
      %178 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %179 = stablehlo.shift_right_logical %170, %178 : tensor<5xui32>
      %180 = stablehlo.or %175, %179 : tensor<5xui32>
      %181 = stablehlo.xor %173, %180 : tensor<5xui32>
      %182 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %183 = stablehlo.reshape %182 : (tensor<1xui32>) -> tensor<ui32>
      %184 = stablehlo.add %173, %181 : tensor<5xui32>
      %185 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %186 = stablehlo.shift_left %181, %185 : tensor<5xui32>
      %187 = stablehlo.constant dense<32> : tensor<ui32>
      %188 = stablehlo.subtract %187, %183 : tensor<ui32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %190 = stablehlo.shift_right_logical %181, %189 : tensor<5xui32>
      %191 = stablehlo.or %186, %190 : tensor<5xui32>
      %192 = stablehlo.xor %184, %191 : tensor<5xui32>
      %193 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %194 = stablehlo.add %184, %193 : tensor<5xui32>
      %195 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %196 = stablehlo.add %192, %195 : tensor<5xui32>
      %197 = stablehlo.constant dense<1> : tensor<i32>
      %198 = stablehlo.add %iterArg_0, %197 : tensor<i32>
      %199 = stablehlo.convert %198 : (tensor<i32>) -> tensor<ui32>
      %200 = stablehlo.broadcast_in_dim %199, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %201 = stablehlo.add %196, %200 : tensor<5xui32>
      %202 = stablehlo.constant dense<1> : tensor<i32>
      %203 = stablehlo.add %iterArg, %202 : tensor<i32>
      stablehlo.return %203, %148, %194, %201, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %65 = stablehlo.concatenate %64#2, %64#3, dim = 0 : (tensor<5xui32>, tensor<5xui32>) -> tensor<10xui32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [1] : (tensor<10xui32>) -> tensor<1x10xui32>
    %67 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %68 = stablehlo.constant dense<16> : tensor<ui32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %70 = stablehlo.multiply %69, %67 : tensor<2x1xui32>
    %71 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<1x10xui32>) -> tensor<2x10xui32>
    %72 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x10xui32>
    %73 = stablehlo.shift_right_logical %71, %72 : tensor<2x10xui32>
    %74 = stablehlo.constant dense<65535> : tensor<ui32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<ui32>) -> tensor<2x10xui32>
    %76 = stablehlo.and %75, %73 : tensor<2x10xui32>
    %77 = stablehlo.transpose %76, dims = [1, 0] : (tensor<2x10xui32>) -> tensor<10x2xui32>
    %78 = stablehlo.reshape %77 : (tensor<10x2xui32>) -> tensor<20xui32>
    %79 = stablehlo.convert %78 : (tensor<20xui32>) -> tensor<20xui16>
    %80 = stablehlo.reshape %79 : (tensor<20xui16>) -> tensor<5x4xui16>
    %81 = stablehlo.iota dim = 0 : tensor<10xui32>
    %82 = "stablehlo.slice"(%45) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %83 = stablehlo.reshape %82 : (tensor<1xui32>) -> tensor<ui32>
    %84 = "stablehlo.slice"(%45) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %85 = stablehlo.reshape %84 : (tensor<1xui32>) -> tensor<ui32>
    %86 = "stablehlo.slice"(%81) {limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<10xui32>) -> tensor<5xui32>
    %87 = "stablehlo.slice"(%81) {limit_indices = dense<10> : tensor<1xi64>, start_indices = dense<5> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<10xui32>) -> tensor<5xui32>
    %88 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %89 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %90 = stablehlo.xor %83, %85 : tensor<ui32>
    %91 = stablehlo.constant dense<466688986> : tensor<ui32>
    %92 = stablehlo.xor %90, %91 : tensor<ui32>
    %93 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<ui32>) -> tensor<5xui32>
    %94 = stablehlo.add %86, %93 : tensor<5xui32>
    %95 = stablehlo.broadcast_in_dim %85, dims = [] : (tensor<ui32>) -> tensor<5xui32>
    %96 = stablehlo.add %87, %95 : tensor<5xui32>
    %97 = stablehlo.constant dense<0> : tensor<i32>
    %98 = stablehlo.constant dense<0> : tensor<i32>
    %99:9 = stablehlo.while(%iterArg = %98, %iterArg_0 = %97, %iterArg_1 = %94, %iterArg_2 = %96, %iterArg_3 = %85, %iterArg_4 = %92, %iterArg_5 = %83, %iterArg_6 = %88, %iterArg_7 = %89) : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %147 = stablehlo.constant dense<5> : tensor<i32>
      %148 = stablehlo.compare  LT, %iterArg, %147,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %148 : tensor<i1>
    } do {
      %147 = stablehlo.constant dense<1> : tensor<i32>
      %148 = stablehlo.add %iterArg_0, %147 : tensor<i32>
      %149 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %150 = stablehlo.reshape %149 : (tensor<1xui32>) -> tensor<ui32>
      %151 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<5xui32>
      %152 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %153 = stablehlo.shift_left %iterArg_2, %152 : tensor<5xui32>
      %154 = stablehlo.constant dense<32> : tensor<ui32>
      %155 = stablehlo.subtract %154, %150 : tensor<ui32>
      %156 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %157 = stablehlo.shift_right_logical %iterArg_2, %156 : tensor<5xui32>
      %158 = stablehlo.or %153, %157 : tensor<5xui32>
      %159 = stablehlo.xor %151, %158 : tensor<5xui32>
      %160 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %161 = stablehlo.reshape %160 : (tensor<1xui32>) -> tensor<ui32>
      %162 = stablehlo.add %151, %159 : tensor<5xui32>
      %163 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %164 = stablehlo.shift_left %159, %163 : tensor<5xui32>
      %165 = stablehlo.constant dense<32> : tensor<ui32>
      %166 = stablehlo.subtract %165, %161 : tensor<ui32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %168 = stablehlo.shift_right_logical %159, %167 : tensor<5xui32>
      %169 = stablehlo.or %164, %168 : tensor<5xui32>
      %170 = stablehlo.xor %162, %169 : tensor<5xui32>
      %171 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %172 = stablehlo.reshape %171 : (tensor<1xui32>) -> tensor<ui32>
      %173 = stablehlo.add %162, %170 : tensor<5xui32>
      %174 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %175 = stablehlo.shift_left %170, %174 : tensor<5xui32>
      %176 = stablehlo.constant dense<32> : tensor<ui32>
      %177 = stablehlo.subtract %176, %172 : tensor<ui32>
      %178 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %179 = stablehlo.shift_right_logical %170, %178 : tensor<5xui32>
      %180 = stablehlo.or %175, %179 : tensor<5xui32>
      %181 = stablehlo.xor %173, %180 : tensor<5xui32>
      %182 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %183 = stablehlo.reshape %182 : (tensor<1xui32>) -> tensor<ui32>
      %184 = stablehlo.add %173, %181 : tensor<5xui32>
      %185 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %186 = stablehlo.shift_left %181, %185 : tensor<5xui32>
      %187 = stablehlo.constant dense<32> : tensor<ui32>
      %188 = stablehlo.subtract %187, %183 : tensor<ui32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %190 = stablehlo.shift_right_logical %181, %189 : tensor<5xui32>
      %191 = stablehlo.or %186, %190 : tensor<5xui32>
      %192 = stablehlo.xor %184, %191 : tensor<5xui32>
      %193 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %194 = stablehlo.add %184, %193 : tensor<5xui32>
      %195 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %196 = stablehlo.add %192, %195 : tensor<5xui32>
      %197 = stablehlo.constant dense<1> : tensor<i32>
      %198 = stablehlo.add %iterArg_0, %197 : tensor<i32>
      %199 = stablehlo.convert %198 : (tensor<i32>) -> tensor<ui32>
      %200 = stablehlo.broadcast_in_dim %199, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %201 = stablehlo.add %196, %200 : tensor<5xui32>
      %202 = stablehlo.constant dense<1> : tensor<i32>
      %203 = stablehlo.add %iterArg, %202 : tensor<i32>
      stablehlo.return %203, %148, %194, %201, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %100 = stablehlo.concatenate %99#2, %99#3, dim = 0 : (tensor<5xui32>, tensor<5xui32>) -> tensor<10xui32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [1] : (tensor<10xui32>) -> tensor<1x10xui32>
    %102 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %103 = stablehlo.constant dense<16> : tensor<ui32>
    %104 = stablehlo.broadcast_in_dim %103, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %105 = stablehlo.multiply %104, %102 : tensor<2x1xui32>
    %106 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<1x10xui32>) -> tensor<2x10xui32>
    %107 = stablehlo.broadcast_in_dim %105, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x10xui32>
    %108 = stablehlo.shift_right_logical %106, %107 : tensor<2x10xui32>
    %109 = stablehlo.constant dense<65535> : tensor<ui32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [] : (tensor<ui32>) -> tensor<2x10xui32>
    %111 = stablehlo.and %110, %108 : tensor<2x10xui32>
    %112 = stablehlo.transpose %111, dims = [1, 0] : (tensor<2x10xui32>) -> tensor<10x2xui32>
    %113 = stablehlo.reshape %112 : (tensor<10x2xui32>) -> tensor<20xui32>
    %114 = stablehlo.convert %113 : (tensor<20xui32>) -> tensor<20xui16>
    %115 = stablehlo.reshape %114 : (tensor<20xui16>) -> tensor<5x4xui16>
    %116 = stablehlo.subtract %20, %19 : tensor<1x1xi16>
    %117 = stablehlo.convert %116 : (tensor<1x1xi16>) -> tensor<1x1xui16>
    %118 = stablehlo.compare  LE, %20, %19,  SIGNED : (tensor<1x1xi16>, tensor<1x1xi16>) -> tensor<1x1xi1>
    %119 = stablehlo.constant dense<1> : tensor<ui16>
    %120 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<ui16>) -> tensor<1x1xui16>
    %121 = stablehlo.select %118, %120, %117 : tensor<1x1xi1>, tensor<1x1xui16>
    %122 = stablehlo.compare  GT, %20, %19,  SIGNED : (tensor<1x1xi16>, tensor<1x1xi16>) -> tensor<1x1xi1>
    %123 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %124 = stablehlo.and %123, %122 : tensor<1x1xi1>
    %125 = stablehlo.constant dense<1> : tensor<ui16>
    %126 = stablehlo.broadcast_in_dim %125, dims = [] : (tensor<ui16>) -> tensor<1x1xui16>
    %127 = stablehlo.add %121, %126 : tensor<1x1xui16>
    %128 = stablehlo.select %124, %127, %121 : tensor<1x1xi1>, tensor<1x1xui16>
    %129 = stablehlo.constant dense<256> : tensor<ui16>
    %130 = stablehlo.broadcast_in_dim %129, dims = [] : (tensor<ui16>) -> tensor<1x1xui16>
    %131 = stablehlo.remainder %130, %128 : tensor<1x1xui16>
    %132 = stablehlo.multiply %131, %131 : tensor<1x1xui16>
    %133 = stablehlo.remainder %132, %128 : tensor<1x1xui16>
    %134 = stablehlo.broadcast_in_dim %128, dims = [0, 1] : (tensor<1x1xui16>) -> tensor<5x4xui16>
    %135 = stablehlo.remainder %80, %134 : tensor<5x4xui16>
    %136 = stablehlo.broadcast_in_dim %133, dims = [0, 1] : (tensor<1x1xui16>) -> tensor<5x4xui16>
    %137 = stablehlo.multiply %135, %136 : tensor<5x4xui16>
    %138 = stablehlo.broadcast_in_dim %128, dims = [0, 1] : (tensor<1x1xui16>) -> tensor<5x4xui16>
    %139 = stablehlo.remainder %115, %138 : tensor<5x4xui16>
    %140 = stablehlo.add %137, %139 : tensor<5x4xui16>
    %141 = stablehlo.broadcast_in_dim %128, dims = [0, 1] : (tensor<1x1xui16>) -> tensor<5x4xui16>
    %142 = stablehlo.remainder %140, %141 : tensor<5x4xui16>
    %143 = stablehlo.convert %142 : (tensor<5x4xui16>) -> tensor<5x4xi16>
    %144 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x1xi16>) -> tensor<5x4xi16>
    %145 = stablehlo.add %144, %143 : tensor<5x4xi16>
    %146 = stablehlo.custom_call @check.eq(%145, %1) : (tensor<5x4xi16>, tensor<5x4xi16>) -> tensor<i1>
    return %146 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<5x4xi16> {
    %0 = stablehlo.constant dense<[[3, 3, -1, -5], [-3, 3, -1, 0], [-4, -5, 3, -1], [4, 3, 1, 2], [-1, 0, -4, 1]]> : tensor<5x4xi16>
    return %0 : tensor<5x4xi16>
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
