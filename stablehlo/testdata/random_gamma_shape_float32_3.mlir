// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<3xf32>)
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = call @gamma(%0#0, %0#1) : (tensor<2xui32>, tensor<3xf32>) -> tensor<3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<3xf32>) {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[-1.13422012, -5.13116312, -3.05222678]> : tensor<3xf32>
    return %0, %1 : tensor<2xui32>, tensor<3xf32>
  }
  func.func private @expected() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<0xFFC00000> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func private @gamma(%arg0: tensor<2xui32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<2xui32>) -> tensor<1x2xui32>
    %1 = stablehlo.iota dim = 0 : tensor<6xui32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x2xui32>) -> tensor<1x1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1x1xui32>) -> tensor<1xui32>
    %4 = "stablehlo.slice"(%0) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x2xui32>) -> tensor<1x1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1x1xui32>) -> tensor<1xui32>
    %6 = "stablehlo.slice"(%1) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
    %7 = "stablehlo.slice"(%1) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
    %8 = stablehlo.broadcast_in_dim %6, dims = [1] : (tensor<3xui32>) -> tensor<1x3xui32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<3xui32>) -> tensor<1x3xui32>
    %10 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %11 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %12 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %13 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %14 = stablehlo.xor %10, %11 : tensor<1x1xui32>
    %15 = stablehlo.constant dense<466688986> : tensor<ui32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %17 = stablehlo.xor %14, %16 : tensor<1x1xui32>
    %18 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<1x3xui32>
    %19 = stablehlo.add %8, %18 : tensor<1x3xui32>
    %20 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<1x3xui32>
    %21 = stablehlo.add %9, %20 : tensor<1x3xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %11, %iterArg_4 = %17, %iterArg_5 = %10, %iterArg_6 = %12, %iterArg_7 = %13) : tensor<i32>, tensor<i32>, tensor<1x3xui32>, tensor<1x3xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %32 = stablehlo.constant dense<5> : tensor<i32>
      %33 = stablehlo.compare  LT, %iterArg, %32,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %33 : tensor<i1>
    } do {
      %32 = stablehlo.constant dense<1> : tensor<i32>
      %33 = stablehlo.add %iterArg_0, %32 : tensor<i32>
      %34 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %35 = stablehlo.reshape %34 : (tensor<1xui32>) -> tensor<ui32>
      %36 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1x3xui32>
      %37 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %38 = stablehlo.shift_left %iterArg_2, %37 : tensor<1x3xui32>
      %39 = stablehlo.constant dense<32> : tensor<ui32>
      %40 = stablehlo.subtract %39, %35 : tensor<ui32>
      %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %42 = stablehlo.shift_right_logical %iterArg_2, %41 : tensor<1x3xui32>
      %43 = stablehlo.or %38, %42 : tensor<1x3xui32>
      %44 = stablehlo.xor %36, %43 : tensor<1x3xui32>
      %45 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %46 = stablehlo.reshape %45 : (tensor<1xui32>) -> tensor<ui32>
      %47 = stablehlo.add %36, %44 : tensor<1x3xui32>
      %48 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %49 = stablehlo.shift_left %44, %48 : tensor<1x3xui32>
      %50 = stablehlo.constant dense<32> : tensor<ui32>
      %51 = stablehlo.subtract %50, %46 : tensor<ui32>
      %52 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %53 = stablehlo.shift_right_logical %44, %52 : tensor<1x3xui32>
      %54 = stablehlo.or %49, %53 : tensor<1x3xui32>
      %55 = stablehlo.xor %47, %54 : tensor<1x3xui32>
      %56 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %57 = stablehlo.reshape %56 : (tensor<1xui32>) -> tensor<ui32>
      %58 = stablehlo.add %47, %55 : tensor<1x3xui32>
      %59 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %60 = stablehlo.shift_left %55, %59 : tensor<1x3xui32>
      %61 = stablehlo.constant dense<32> : tensor<ui32>
      %62 = stablehlo.subtract %61, %57 : tensor<ui32>
      %63 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %64 = stablehlo.shift_right_logical %55, %63 : tensor<1x3xui32>
      %65 = stablehlo.or %60, %64 : tensor<1x3xui32>
      %66 = stablehlo.xor %58, %65 : tensor<1x3xui32>
      %67 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %68 = stablehlo.reshape %67 : (tensor<1xui32>) -> tensor<ui32>
      %69 = stablehlo.add %58, %66 : tensor<1x3xui32>
      %70 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %71 = stablehlo.shift_left %66, %70 : tensor<1x3xui32>
      %72 = stablehlo.constant dense<32> : tensor<ui32>
      %73 = stablehlo.subtract %72, %68 : tensor<ui32>
      %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %75 = stablehlo.shift_right_logical %66, %74 : tensor<1x3xui32>
      %76 = stablehlo.or %71, %75 : tensor<1x3xui32>
      %77 = stablehlo.xor %69, %76 : tensor<1x3xui32>
      %78 = stablehlo.broadcast_in_dim %iterArg_3, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<1x3xui32>
      %79 = stablehlo.add %69, %78 : tensor<1x3xui32>
      %80 = stablehlo.broadcast_in_dim %iterArg_4, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<1x3xui32>
      %81 = stablehlo.add %77, %80 : tensor<1x3xui32>
      %82 = stablehlo.constant dense<1> : tensor<i32>
      %83 = stablehlo.add %iterArg_0, %82 : tensor<i32>
      %84 = stablehlo.convert %83 : (tensor<i32>) -> tensor<ui32>
      %85 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
      %86 = stablehlo.add %81, %85 : tensor<1x3xui32>
      %87 = stablehlo.constant dense<1> : tensor<i32>
      %88 = stablehlo.add %iterArg, %87 : tensor<i32>
      stablehlo.return %88, %33, %79, %86, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1x3xui32>, tensor<1x3xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 1 : (tensor<1x3xui32>, tensor<1x3xui32>) -> tensor<1x6xui32>
    %26 = stablehlo.reshape %25 : (tensor<1x6xui32>) -> tensor<1x3x2xui32>
    %27 = stablehlo.reshape %26 : (tensor<1x3x2xui32>) -> tensor<3x2xui32>
    %28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %30 = stablehlo.constant dense<0> : tensor<i32>
    %31:4 = stablehlo.while(%iterArg = %27, %iterArg_0 = %arg1, %iterArg_1 = %30, %iterArg_2 = %29) : tensor<3x2xui32>, tensor<3xf32>, tensor<i32>, tensor<3xf32>
     cond {
      %32 = stablehlo.constant dense<3> : tensor<i32>
      %33 = stablehlo.compare  LT, %iterArg_1, %32,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %33 : tensor<i1>
    } do {
      %32 = stablehlo.constant dense<0> : tensor<i32>
      %33 = stablehlo.compare  LT, %iterArg_1, %32,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %34 = stablehlo.convert %iterArg_1 : tensor<i32>
      %35 = stablehlo.constant dense<3> : tensor<i32>
      %36 = stablehlo.add %34, %35 : tensor<i32>
      %37 = stablehlo.select %33, %36, %iterArg_1 : tensor<i1>, tensor<i32>
      %38 = stablehlo.constant dense<0> : tensor<i32>
      %39 = stablehlo.dynamic_slice %iterArg, %37, %38, sizes = [1, 2] : (tensor<3x2xui32>, tensor<i32>, tensor<i32>) -> tensor<1x2xui32>
      %40 = stablehlo.reshape %39 : (tensor<1x2xui32>) -> tensor<2xui32>
      %41 = stablehlo.constant dense<0> : tensor<i32>
      %42 = stablehlo.compare  LT, %iterArg_1, %41,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %43 = stablehlo.convert %iterArg_1 : tensor<i32>
      %44 = stablehlo.constant dense<3> : tensor<i32>
      %45 = stablehlo.add %43, %44 : tensor<i32>
      %46 = stablehlo.select %42, %45, %iterArg_1 : tensor<i1>, tensor<i32>
      %47 = stablehlo.dynamic_slice %iterArg_0, %46, sizes = [1] : (tensor<3xf32>, tensor<i32>) -> tensor<1xf32>
      %48 = stablehlo.reshape %47 : (tensor<1xf32>) -> tensor<f32>
      %49 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %50 = stablehlo.compare  GE, %48, %49,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %51 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %52 = stablehlo.add %48, %51 : tensor<f32>
      %53 = stablehlo.select %50, %48, %52 : tensor<i1>, tensor<f32>
      %54 = stablehlo.constant dense<0.333333343> : tensor<f32>
      %55 = stablehlo.subtract %53, %54 : tensor<f32>
      %56 = stablehlo.sqrt %55 : tensor<f32>
      %57 = stablehlo.constant dense<0.333333343> : tensor<f32>
      %58 = stablehlo.divide %57, %56 : tensor<f32>
      %59 = stablehlo.iota dim = 0 : tensor<4xui32>
      %60 = "stablehlo.slice"(%40) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
      %61 = stablehlo.reshape %60 : (tensor<1xui32>) -> tensor<ui32>
      %62 = "stablehlo.slice"(%40) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
      %63 = stablehlo.reshape %62 : (tensor<1xui32>) -> tensor<ui32>
      %64 = "stablehlo.slice"(%59) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
      %65 = "stablehlo.slice"(%59) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
      %66 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
      %67 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
      %68 = stablehlo.xor %61, %63 : tensor<ui32>
      %69 = stablehlo.constant dense<466688986> : tensor<ui32>
      %70 = stablehlo.xor %68, %69 : tensor<ui32>
      %71 = stablehlo.broadcast_in_dim %61, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %72 = stablehlo.add %64, %71 : tensor<2xui32>
      %73 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %74 = stablehlo.add %65, %73 : tensor<2xui32>
      %75 = stablehlo.constant dense<0> : tensor<i32>
      %76 = stablehlo.constant dense<0> : tensor<i32>
      %77:9 = stablehlo.while(%iterArg_3 = %76, %iterArg_4 = %75, %iterArg_5 = %72, %iterArg_6 = %74, %iterArg_7 = %63, %iterArg_8 = %70, %iterArg_9 = %61, %iterArg_10 = %66, %iterArg_11 = %67) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
       cond {
        %151 = stablehlo.constant dense<5> : tensor<i32>
        %152 = stablehlo.compare  LT, %iterArg_3, %151,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %152 : tensor<i1>
      } do {
        %151 = stablehlo.constant dense<1> : tensor<i32>
        %152 = stablehlo.add %iterArg_4, %151 : tensor<i32>
        %153 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %154 = stablehlo.reshape %153 : (tensor<1xui32>) -> tensor<ui32>
        %155 = stablehlo.add %iterArg_5, %iterArg_6 : tensor<2xui32>
        %156 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %157 = stablehlo.shift_left %iterArg_6, %156 : tensor<2xui32>
        %158 = stablehlo.constant dense<32> : tensor<ui32>
        %159 = stablehlo.subtract %158, %154 : tensor<ui32>
        %160 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %161 = stablehlo.shift_right_logical %iterArg_6, %160 : tensor<2xui32>
        %162 = stablehlo.or %157, %161 : tensor<2xui32>
        %163 = stablehlo.xor %155, %162 : tensor<2xui32>
        %164 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %165 = stablehlo.reshape %164 : (tensor<1xui32>) -> tensor<ui32>
        %166 = stablehlo.add %155, %163 : tensor<2xui32>
        %167 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %168 = stablehlo.shift_left %163, %167 : tensor<2xui32>
        %169 = stablehlo.constant dense<32> : tensor<ui32>
        %170 = stablehlo.subtract %169, %165 : tensor<ui32>
        %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %172 = stablehlo.shift_right_logical %163, %171 : tensor<2xui32>
        %173 = stablehlo.or %168, %172 : tensor<2xui32>
        %174 = stablehlo.xor %166, %173 : tensor<2xui32>
        %175 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %176 = stablehlo.reshape %175 : (tensor<1xui32>) -> tensor<ui32>
        %177 = stablehlo.add %166, %174 : tensor<2xui32>
        %178 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %179 = stablehlo.shift_left %174, %178 : tensor<2xui32>
        %180 = stablehlo.constant dense<32> : tensor<ui32>
        %181 = stablehlo.subtract %180, %176 : tensor<ui32>
        %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %183 = stablehlo.shift_right_logical %174, %182 : tensor<2xui32>
        %184 = stablehlo.or %179, %183 : tensor<2xui32>
        %185 = stablehlo.xor %177, %184 : tensor<2xui32>
        %186 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %187 = stablehlo.reshape %186 : (tensor<1xui32>) -> tensor<ui32>
        %188 = stablehlo.add %177, %185 : tensor<2xui32>
        %189 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %190 = stablehlo.shift_left %185, %189 : tensor<2xui32>
        %191 = stablehlo.constant dense<32> : tensor<ui32>
        %192 = stablehlo.subtract %191, %187 : tensor<ui32>
        %193 = stablehlo.broadcast_in_dim %192, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %194 = stablehlo.shift_right_logical %185, %193 : tensor<2xui32>
        %195 = stablehlo.or %190, %194 : tensor<2xui32>
        %196 = stablehlo.xor %188, %195 : tensor<2xui32>
        %197 = stablehlo.broadcast_in_dim %iterArg_7, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %198 = stablehlo.add %188, %197 : tensor<2xui32>
        %199 = stablehlo.broadcast_in_dim %iterArg_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %200 = stablehlo.add %196, %199 : tensor<2xui32>
        %201 = stablehlo.constant dense<1> : tensor<i32>
        %202 = stablehlo.add %iterArg_4, %201 : tensor<i32>
        %203 = stablehlo.convert %202 : (tensor<i32>) -> tensor<ui32>
        %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<ui32>) -> tensor<2xui32>
        %205 = stablehlo.add %200, %204 : tensor<2xui32>
        %206 = stablehlo.constant dense<1> : tensor<i32>
        %207 = stablehlo.add %iterArg_3, %206 : tensor<i32>
        stablehlo.return %207, %152, %198, %205, %iterArg_8, %iterArg_9, %iterArg_7, %iterArg_11, %iterArg_10 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
      }
      %78 = stablehlo.concatenate %77#2, %77#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
      %79 = stablehlo.reshape %78 : (tensor<4xui32>) -> tensor<2x2xui32>
      %80 = "stablehlo.slice"(%79) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
      %81 = stablehlo.reshape %80 : (tensor<1x2xui32>) -> tensor<2xui32>
      %82 = "stablehlo.slice"(%79) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
      %83 = stablehlo.reshape %82 : (tensor<1x2xui32>) -> tensor<2xui32>
      %84 = stablehlo.constant dense<0> : tensor<1xui32>
      %85 = stablehlo.iota dim = 0 : tensor<1xui32>
      %86 = "stablehlo.slice"(%83) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
      %87 = stablehlo.reshape %86 : (tensor<1xui32>) -> tensor<ui32>
      %88 = "stablehlo.slice"(%83) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
      %89 = stablehlo.reshape %88 : (tensor<1xui32>) -> tensor<ui32>
      %90 = stablehlo.concatenate %85, %84, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
      %91 = "stablehlo.slice"(%90) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
      %92 = "stablehlo.slice"(%90) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
      %93 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
      %94 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
      %95 = stablehlo.xor %87, %89 : tensor<ui32>
      %96 = stablehlo.constant dense<466688986> : tensor<ui32>
      %97 = stablehlo.xor %95, %96 : tensor<ui32>
      %98 = stablehlo.broadcast_in_dim %87, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %99 = stablehlo.add %91, %98 : tensor<1xui32>
      %100 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %101 = stablehlo.add %92, %100 : tensor<1xui32>
      %102 = stablehlo.constant dense<0> : tensor<i32>
      %103 = stablehlo.constant dense<0> : tensor<i32>
      %104:9 = stablehlo.while(%iterArg_3 = %103, %iterArg_4 = %102, %iterArg_5 = %99, %iterArg_6 = %101, %iterArg_7 = %89, %iterArg_8 = %97, %iterArg_9 = %87, %iterArg_10 = %93, %iterArg_11 = %94) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
       cond {
        %151 = stablehlo.constant dense<5> : tensor<i32>
        %152 = stablehlo.compare  LT, %iterArg_3, %151,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %152 : tensor<i1>
      } do {
        %151 = stablehlo.constant dense<1> : tensor<i32>
        %152 = stablehlo.add %iterArg_4, %151 : tensor<i32>
        %153 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %154 = stablehlo.reshape %153 : (tensor<1xui32>) -> tensor<ui32>
        %155 = stablehlo.add %iterArg_5, %iterArg_6 : tensor<1xui32>
        %156 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %157 = stablehlo.shift_left %iterArg_6, %156 : tensor<1xui32>
        %158 = stablehlo.constant dense<32> : tensor<ui32>
        %159 = stablehlo.subtract %158, %154 : tensor<ui32>
        %160 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %161 = stablehlo.shift_right_logical %iterArg_6, %160 : tensor<1xui32>
        %162 = stablehlo.or %157, %161 : tensor<1xui32>
        %163 = stablehlo.xor %155, %162 : tensor<1xui32>
        %164 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %165 = stablehlo.reshape %164 : (tensor<1xui32>) -> tensor<ui32>
        %166 = stablehlo.add %155, %163 : tensor<1xui32>
        %167 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %168 = stablehlo.shift_left %163, %167 : tensor<1xui32>
        %169 = stablehlo.constant dense<32> : tensor<ui32>
        %170 = stablehlo.subtract %169, %165 : tensor<ui32>
        %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %172 = stablehlo.shift_right_logical %163, %171 : tensor<1xui32>
        %173 = stablehlo.or %168, %172 : tensor<1xui32>
        %174 = stablehlo.xor %166, %173 : tensor<1xui32>
        %175 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %176 = stablehlo.reshape %175 : (tensor<1xui32>) -> tensor<ui32>
        %177 = stablehlo.add %166, %174 : tensor<1xui32>
        %178 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %179 = stablehlo.shift_left %174, %178 : tensor<1xui32>
        %180 = stablehlo.constant dense<32> : tensor<ui32>
        %181 = stablehlo.subtract %180, %176 : tensor<ui32>
        %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %183 = stablehlo.shift_right_logical %174, %182 : tensor<1xui32>
        %184 = stablehlo.or %179, %183 : tensor<1xui32>
        %185 = stablehlo.xor %177, %184 : tensor<1xui32>
        %186 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
        %187 = stablehlo.reshape %186 : (tensor<1xui32>) -> tensor<ui32>
        %188 = stablehlo.add %177, %185 : tensor<1xui32>
        %189 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %190 = stablehlo.shift_left %185, %189 : tensor<1xui32>
        %191 = stablehlo.constant dense<32> : tensor<ui32>
        %192 = stablehlo.subtract %191, %187 : tensor<ui32>
        %193 = stablehlo.broadcast_in_dim %192, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %194 = stablehlo.shift_right_logical %185, %193 : tensor<1xui32>
        %195 = stablehlo.or %190, %194 : tensor<1xui32>
        %196 = stablehlo.xor %188, %195 : tensor<1xui32>
        %197 = stablehlo.broadcast_in_dim %iterArg_7, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %198 = stablehlo.add %188, %197 : tensor<1xui32>
        %199 = stablehlo.broadcast_in_dim %iterArg_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %200 = stablehlo.add %196, %199 : tensor<1xui32>
        %201 = stablehlo.constant dense<1> : tensor<i32>
        %202 = stablehlo.add %iterArg_4, %201 : tensor<i32>
        %203 = stablehlo.convert %202 : (tensor<i32>) -> tensor<ui32>
        %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %205 = stablehlo.add %200, %204 : tensor<1xui32>
        %206 = stablehlo.constant dense<1> : tensor<i32>
        %207 = stablehlo.add %iterArg_3, %206 : tensor<i32>
        stablehlo.return %207, %152, %198, %205, %iterArg_8, %iterArg_9, %iterArg_7, %iterArg_11, %iterArg_10 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
      }
      %105 = stablehlo.concatenate %104#2, %104#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
      %106 = stablehlo.constant dense<0> : tensor<i32>
      %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %108 = "stablehlo.gather"(%105, %107) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
      %109 = stablehlo.reshape %108 : (tensor<1xui32>) -> tensor<ui32>
      %110 = stablehlo.constant dense<9> : tensor<ui32>
      %111 = stablehlo.shift_right_logical %109, %110 : tensor<ui32>
      %112 = stablehlo.constant dense<1065353216> : tensor<ui32>
      %113 = stablehlo.or %111, %112 : tensor<ui32>
      %114 = stablehlo.bitcast_convert %113 : (tensor<ui32>) -> tensor<f32>
      %115 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %116 = stablehlo.subtract %114, %115 : tensor<f32>
      %117 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %119 = stablehlo.subtract %117, %118 : tensor<f32>
      %120 = stablehlo.multiply %116, %119 : tensor<f32>
      %121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %122 = stablehlo.add %120, %121 : tensor<f32>
      %123 = stablehlo.reshape %122 : (tensor<f32>) -> tensor<f32>
      %124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %125 = stablehlo.maximum %124, %123 : tensor<f32>
      %126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %127 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %128 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %129:6 = stablehlo.while(%iterArg_3 = %55, %iterArg_4 = %58, %iterArg_5 = %81, %iterArg_6 = %126, %iterArg_7 = %127, %iterArg_8 = %128) : tensor<f32>, tensor<f32>, tensor<2xui32>, tensor<f32>, tensor<f32>, tensor<f32>
       cond {
        %151 = stablehlo.multiply %iterArg_6, %iterArg_6 : tensor<f32>
        %152 = stablehlo.constant dense<3.310000e-02> : tensor<f32>
        %153 = stablehlo.multiply %152, %151 : tensor<f32>
        %154 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %155 = stablehlo.subtract %154, %153 : tensor<f32>
        %156 = stablehlo.compare  GE, %iterArg_8, %155,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %157 = stablehlo.log %iterArg_8 : tensor<f32>
        %158 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
        %159 = stablehlo.multiply %iterArg_6, %158 : tensor<f32>
        %160 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %161 = stablehlo.subtract %160, %iterArg_7 : tensor<f32>
        %162 = stablehlo.log %iterArg_7 : tensor<f32>
        %163 = stablehlo.add %161, %162 : tensor<f32>
        %164 = stablehlo.multiply %iterArg_3, %163 : tensor<f32>
        %165 = stablehlo.add %159, %164 : tensor<f32>
        %166 = stablehlo.compare  GE, %157, %165,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %167 = stablehlo.and %156, %166 : tensor<i1>
        stablehlo.return %167 : tensor<i1>
      } do {
        %151 = stablehlo.iota dim = 0 : tensor<6xui32>
        %152 = "stablehlo.slice"(%iterArg_5) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
        %153 = stablehlo.reshape %152 : (tensor<1xui32>) -> tensor<ui32>
        %154 = "stablehlo.slice"(%iterArg_5) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
        %155 = stablehlo.reshape %154 : (tensor<1xui32>) -> tensor<ui32>
        %156 = "stablehlo.slice"(%151) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
        %157 = "stablehlo.slice"(%151) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xui32>) -> tensor<3xui32>
        %158 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
        %159 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
        %160 = stablehlo.xor %153, %155 : tensor<ui32>
        %161 = stablehlo.constant dense<466688986> : tensor<ui32>
        %162 = stablehlo.xor %160, %161 : tensor<ui32>
        %163 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<ui32>) -> tensor<3xui32>
        %164 = stablehlo.add %156, %163 : tensor<3xui32>
        %165 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<ui32>) -> tensor<3xui32>
        %166 = stablehlo.add %157, %165 : tensor<3xui32>
        %167 = stablehlo.constant dense<0> : tensor<i32>
        %168 = stablehlo.constant dense<0> : tensor<i32>
        %169:9 = stablehlo.while(%iterArg_9 = %168, %iterArg_10 = %167, %iterArg_11 = %164, %iterArg_12 = %166, %iterArg_13 = %155, %iterArg_14 = %162, %iterArg_15 = %153, %iterArg_16 = %158, %iterArg_17 = %159) : tensor<i32>, tensor<i32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
         cond {
          %226 = stablehlo.constant dense<5> : tensor<i32>
          %227 = stablehlo.compare  LT, %iterArg_9, %226,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %227 : tensor<i1>
        } do {
          %226 = stablehlo.constant dense<1> : tensor<i32>
          %227 = stablehlo.add %iterArg_10, %226 : tensor<i32>
          %228 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %229 = stablehlo.reshape %228 : (tensor<1xui32>) -> tensor<ui32>
          %230 = stablehlo.add %iterArg_11, %iterArg_12 : tensor<3xui32>
          %231 = stablehlo.broadcast_in_dim %229, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %232 = stablehlo.shift_left %iterArg_12, %231 : tensor<3xui32>
          %233 = stablehlo.constant dense<32> : tensor<ui32>
          %234 = stablehlo.subtract %233, %229 : tensor<ui32>
          %235 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %236 = stablehlo.shift_right_logical %iterArg_12, %235 : tensor<3xui32>
          %237 = stablehlo.or %232, %236 : tensor<3xui32>
          %238 = stablehlo.xor %230, %237 : tensor<3xui32>
          %239 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %240 = stablehlo.reshape %239 : (tensor<1xui32>) -> tensor<ui32>
          %241 = stablehlo.add %230, %238 : tensor<3xui32>
          %242 = stablehlo.broadcast_in_dim %240, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %243 = stablehlo.shift_left %238, %242 : tensor<3xui32>
          %244 = stablehlo.constant dense<32> : tensor<ui32>
          %245 = stablehlo.subtract %244, %240 : tensor<ui32>
          %246 = stablehlo.broadcast_in_dim %245, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %247 = stablehlo.shift_right_logical %238, %246 : tensor<3xui32>
          %248 = stablehlo.or %243, %247 : tensor<3xui32>
          %249 = stablehlo.xor %241, %248 : tensor<3xui32>
          %250 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %251 = stablehlo.reshape %250 : (tensor<1xui32>) -> tensor<ui32>
          %252 = stablehlo.add %241, %249 : tensor<3xui32>
          %253 = stablehlo.broadcast_in_dim %251, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %254 = stablehlo.shift_left %249, %253 : tensor<3xui32>
          %255 = stablehlo.constant dense<32> : tensor<ui32>
          %256 = stablehlo.subtract %255, %251 : tensor<ui32>
          %257 = stablehlo.broadcast_in_dim %256, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %258 = stablehlo.shift_right_logical %249, %257 : tensor<3xui32>
          %259 = stablehlo.or %254, %258 : tensor<3xui32>
          %260 = stablehlo.xor %252, %259 : tensor<3xui32>
          %261 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %262 = stablehlo.reshape %261 : (tensor<1xui32>) -> tensor<ui32>
          %263 = stablehlo.add %252, %260 : tensor<3xui32>
          %264 = stablehlo.broadcast_in_dim %262, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %265 = stablehlo.shift_left %260, %264 : tensor<3xui32>
          %266 = stablehlo.constant dense<32> : tensor<ui32>
          %267 = stablehlo.subtract %266, %262 : tensor<ui32>
          %268 = stablehlo.broadcast_in_dim %267, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %269 = stablehlo.shift_right_logical %260, %268 : tensor<3xui32>
          %270 = stablehlo.or %265, %269 : tensor<3xui32>
          %271 = stablehlo.xor %263, %270 : tensor<3xui32>
          %272 = stablehlo.broadcast_in_dim %iterArg_13, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %273 = stablehlo.add %263, %272 : tensor<3xui32>
          %274 = stablehlo.broadcast_in_dim %iterArg_14, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %275 = stablehlo.add %271, %274 : tensor<3xui32>
          %276 = stablehlo.constant dense<1> : tensor<i32>
          %277 = stablehlo.add %iterArg_10, %276 : tensor<i32>
          %278 = stablehlo.convert %277 : (tensor<i32>) -> tensor<ui32>
          %279 = stablehlo.broadcast_in_dim %278, dims = [] : (tensor<ui32>) -> tensor<3xui32>
          %280 = stablehlo.add %275, %279 : tensor<3xui32>
          %281 = stablehlo.constant dense<1> : tensor<i32>
          %282 = stablehlo.add %iterArg_9, %281 : tensor<i32>
          stablehlo.return %282, %227, %273, %280, %iterArg_14, %iterArg_15, %iterArg_13, %iterArg_17, %iterArg_16 : tensor<i32>, tensor<i32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
        }
        %170 = stablehlo.concatenate %169#2, %169#3, dim = 0 : (tensor<3xui32>, tensor<3xui32>) -> tensor<6xui32>
        %171 = stablehlo.reshape %170 : (tensor<6xui32>) -> tensor<3x2xui32>
        %172 = "stablehlo.slice"(%171) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x2xui32>) -> tensor<1x2xui32>
        %173 = stablehlo.reshape %172 : (tensor<1x2xui32>) -> tensor<2xui32>
        %174 = "stablehlo.slice"(%171) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x2xui32>) -> tensor<1x2xui32>
        %175 = stablehlo.reshape %174 : (tensor<1x2xui32>) -> tensor<2xui32>
        %176 = "stablehlo.slice"(%171) {limit_indices = dense<[3, 2]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x2xui32>) -> tensor<1x2xui32>
        %177 = stablehlo.reshape %176 : (tensor<1x2xui32>) -> tensor<2xui32>
        %178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %179 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
        %180:4 = stablehlo.while(%iterArg_9 = %iterArg_4, %iterArg_10 = %175, %iterArg_11 = %178, %iterArg_12 = %179) : tensor<f32>, tensor<2xui32>, tensor<f32>, tensor<f32>
         cond {
          %226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %227 = stablehlo.compare  LE, %iterArg_12, %226,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
          stablehlo.return %227 : tensor<i1>
        } do {
          %226 = stablehlo.iota dim = 0 : tensor<4xui32>
          %227 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
          %228 = stablehlo.reshape %227 : (tensor<1xui32>) -> tensor<ui32>
          %229 = "stablehlo.slice"(%iterArg_10) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
          %230 = stablehlo.reshape %229 : (tensor<1xui32>) -> tensor<ui32>
          %231 = "stablehlo.slice"(%226) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
          %232 = "stablehlo.slice"(%226) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
          %233 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
          %234 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
          %235 = stablehlo.xor %228, %230 : tensor<ui32>
          %236 = stablehlo.constant dense<466688986> : tensor<ui32>
          %237 = stablehlo.xor %235, %236 : tensor<ui32>
          %238 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<ui32>) -> tensor<2xui32>
          %239 = stablehlo.add %231, %238 : tensor<2xui32>
          %240 = stablehlo.broadcast_in_dim %230, dims = [] : (tensor<ui32>) -> tensor<2xui32>
          %241 = stablehlo.add %232, %240 : tensor<2xui32>
          %242 = stablehlo.constant dense<0> : tensor<i32>
          %243 = stablehlo.constant dense<0> : tensor<i32>
          %244:9 = stablehlo.while(%iterArg_13 = %243, %iterArg_14 = %242, %iterArg_15 = %239, %iterArg_16 = %241, %iterArg_17 = %230, %iterArg_18 = %237, %iterArg_19 = %228, %iterArg_20 = %233, %iterArg_21 = %234) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
           cond {
            %299 = stablehlo.constant dense<5> : tensor<i32>
            %300 = stablehlo.compare  LT, %iterArg_13, %299,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
            stablehlo.return %300 : tensor<i1>
          } do {
            %299 = stablehlo.constant dense<1> : tensor<i32>
            %300 = stablehlo.add %iterArg_14, %299 : tensor<i32>
            %301 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %302 = stablehlo.reshape %301 : (tensor<1xui32>) -> tensor<ui32>
            %303 = stablehlo.add %iterArg_15, %iterArg_16 : tensor<2xui32>
            %304 = stablehlo.broadcast_in_dim %302, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %305 = stablehlo.shift_left %iterArg_16, %304 : tensor<2xui32>
            %306 = stablehlo.constant dense<32> : tensor<ui32>
            %307 = stablehlo.subtract %306, %302 : tensor<ui32>
            %308 = stablehlo.broadcast_in_dim %307, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %309 = stablehlo.shift_right_logical %iterArg_16, %308 : tensor<2xui32>
            %310 = stablehlo.or %305, %309 : tensor<2xui32>
            %311 = stablehlo.xor %303, %310 : tensor<2xui32>
            %312 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %313 = stablehlo.reshape %312 : (tensor<1xui32>) -> tensor<ui32>
            %314 = stablehlo.add %303, %311 : tensor<2xui32>
            %315 = stablehlo.broadcast_in_dim %313, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %316 = stablehlo.shift_left %311, %315 : tensor<2xui32>
            %317 = stablehlo.constant dense<32> : tensor<ui32>
            %318 = stablehlo.subtract %317, %313 : tensor<ui32>
            %319 = stablehlo.broadcast_in_dim %318, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %320 = stablehlo.shift_right_logical %311, %319 : tensor<2xui32>
            %321 = stablehlo.or %316, %320 : tensor<2xui32>
            %322 = stablehlo.xor %314, %321 : tensor<2xui32>
            %323 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %324 = stablehlo.reshape %323 : (tensor<1xui32>) -> tensor<ui32>
            %325 = stablehlo.add %314, %322 : tensor<2xui32>
            %326 = stablehlo.broadcast_in_dim %324, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %327 = stablehlo.shift_left %322, %326 : tensor<2xui32>
            %328 = stablehlo.constant dense<32> : tensor<ui32>
            %329 = stablehlo.subtract %328, %324 : tensor<ui32>
            %330 = stablehlo.broadcast_in_dim %329, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %331 = stablehlo.shift_right_logical %322, %330 : tensor<2xui32>
            %332 = stablehlo.or %327, %331 : tensor<2xui32>
            %333 = stablehlo.xor %325, %332 : tensor<2xui32>
            %334 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %335 = stablehlo.reshape %334 : (tensor<1xui32>) -> tensor<ui32>
            %336 = stablehlo.add %325, %333 : tensor<2xui32>
            %337 = stablehlo.broadcast_in_dim %335, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %338 = stablehlo.shift_left %333, %337 : tensor<2xui32>
            %339 = stablehlo.constant dense<32> : tensor<ui32>
            %340 = stablehlo.subtract %339, %335 : tensor<ui32>
            %341 = stablehlo.broadcast_in_dim %340, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %342 = stablehlo.shift_right_logical %333, %341 : tensor<2xui32>
            %343 = stablehlo.or %338, %342 : tensor<2xui32>
            %344 = stablehlo.xor %336, %343 : tensor<2xui32>
            %345 = stablehlo.broadcast_in_dim %iterArg_17, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %346 = stablehlo.add %336, %345 : tensor<2xui32>
            %347 = stablehlo.broadcast_in_dim %iterArg_18, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %348 = stablehlo.add %344, %347 : tensor<2xui32>
            %349 = stablehlo.constant dense<1> : tensor<i32>
            %350 = stablehlo.add %iterArg_14, %349 : tensor<i32>
            %351 = stablehlo.convert %350 : (tensor<i32>) -> tensor<ui32>
            %352 = stablehlo.broadcast_in_dim %351, dims = [] : (tensor<ui32>) -> tensor<2xui32>
            %353 = stablehlo.add %348, %352 : tensor<2xui32>
            %354 = stablehlo.constant dense<1> : tensor<i32>
            %355 = stablehlo.add %iterArg_13, %354 : tensor<i32>
            stablehlo.return %355, %300, %346, %353, %iterArg_18, %iterArg_19, %iterArg_17, %iterArg_21, %iterArg_20 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
          }
          %245 = stablehlo.concatenate %244#2, %244#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
          %246 = stablehlo.reshape %245 : (tensor<4xui32>) -> tensor<2x2xui32>
          %247 = "stablehlo.slice"(%246) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
          %248 = stablehlo.reshape %247 : (tensor<1x2xui32>) -> tensor<2xui32>
          %249 = "stablehlo.slice"(%246) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
          %250 = stablehlo.reshape %249 : (tensor<1x2xui32>) -> tensor<2xui32>
          %251 = stablehlo.constant dense<0> : tensor<1xui32>
          %252 = stablehlo.iota dim = 0 : tensor<1xui32>
          %253 = "stablehlo.slice"(%250) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
          %254 = stablehlo.reshape %253 : (tensor<1xui32>) -> tensor<ui32>
          %255 = "stablehlo.slice"(%250) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
          %256 = stablehlo.reshape %255 : (tensor<1xui32>) -> tensor<ui32>
          %257 = stablehlo.concatenate %252, %251, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
          %258 = "stablehlo.slice"(%257) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
          %259 = "stablehlo.slice"(%257) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
          %260 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
          %261 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
          %262 = stablehlo.xor %254, %256 : tensor<ui32>
          %263 = stablehlo.constant dense<466688986> : tensor<ui32>
          %264 = stablehlo.xor %262, %263 : tensor<ui32>
          %265 = stablehlo.broadcast_in_dim %254, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %266 = stablehlo.add %258, %265 : tensor<1xui32>
          %267 = stablehlo.broadcast_in_dim %256, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %268 = stablehlo.add %259, %267 : tensor<1xui32>
          %269 = stablehlo.constant dense<0> : tensor<i32>
          %270 = stablehlo.constant dense<0> : tensor<i32>
          %271:9 = stablehlo.while(%iterArg_13 = %270, %iterArg_14 = %269, %iterArg_15 = %266, %iterArg_16 = %268, %iterArg_17 = %256, %iterArg_18 = %264, %iterArg_19 = %254, %iterArg_20 = %260, %iterArg_21 = %261) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
           cond {
            %299 = stablehlo.constant dense<5> : tensor<i32>
            %300 = stablehlo.compare  LT, %iterArg_13, %299,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
            stablehlo.return %300 : tensor<i1>
          } do {
            %299 = stablehlo.constant dense<1> : tensor<i32>
            %300 = stablehlo.add %iterArg_14, %299 : tensor<i32>
            %301 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %302 = stablehlo.reshape %301 : (tensor<1xui32>) -> tensor<ui32>
            %303 = stablehlo.add %iterArg_15, %iterArg_16 : tensor<1xui32>
            %304 = stablehlo.broadcast_in_dim %302, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %305 = stablehlo.shift_left %iterArg_16, %304 : tensor<1xui32>
            %306 = stablehlo.constant dense<32> : tensor<ui32>
            %307 = stablehlo.subtract %306, %302 : tensor<ui32>
            %308 = stablehlo.broadcast_in_dim %307, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %309 = stablehlo.shift_right_logical %iterArg_16, %308 : tensor<1xui32>
            %310 = stablehlo.or %305, %309 : tensor<1xui32>
            %311 = stablehlo.xor %303, %310 : tensor<1xui32>
            %312 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %313 = stablehlo.reshape %312 : (tensor<1xui32>) -> tensor<ui32>
            %314 = stablehlo.add %303, %311 : tensor<1xui32>
            %315 = stablehlo.broadcast_in_dim %313, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %316 = stablehlo.shift_left %311, %315 : tensor<1xui32>
            %317 = stablehlo.constant dense<32> : tensor<ui32>
            %318 = stablehlo.subtract %317, %313 : tensor<ui32>
            %319 = stablehlo.broadcast_in_dim %318, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %320 = stablehlo.shift_right_logical %311, %319 : tensor<1xui32>
            %321 = stablehlo.or %316, %320 : tensor<1xui32>
            %322 = stablehlo.xor %314, %321 : tensor<1xui32>
            %323 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %324 = stablehlo.reshape %323 : (tensor<1xui32>) -> tensor<ui32>
            %325 = stablehlo.add %314, %322 : tensor<1xui32>
            %326 = stablehlo.broadcast_in_dim %324, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %327 = stablehlo.shift_left %322, %326 : tensor<1xui32>
            %328 = stablehlo.constant dense<32> : tensor<ui32>
            %329 = stablehlo.subtract %328, %324 : tensor<ui32>
            %330 = stablehlo.broadcast_in_dim %329, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %331 = stablehlo.shift_right_logical %322, %330 : tensor<1xui32>
            %332 = stablehlo.or %327, %331 : tensor<1xui32>
            %333 = stablehlo.xor %325, %332 : tensor<1xui32>
            %334 = "stablehlo.slice"(%iterArg_20) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
            %335 = stablehlo.reshape %334 : (tensor<1xui32>) -> tensor<ui32>
            %336 = stablehlo.add %325, %333 : tensor<1xui32>
            %337 = stablehlo.broadcast_in_dim %335, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %338 = stablehlo.shift_left %333, %337 : tensor<1xui32>
            %339 = stablehlo.constant dense<32> : tensor<ui32>
            %340 = stablehlo.subtract %339, %335 : tensor<ui32>
            %341 = stablehlo.broadcast_in_dim %340, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %342 = stablehlo.shift_right_logical %333, %341 : tensor<1xui32>
            %343 = stablehlo.or %338, %342 : tensor<1xui32>
            %344 = stablehlo.xor %336, %343 : tensor<1xui32>
            %345 = stablehlo.broadcast_in_dim %iterArg_17, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %346 = stablehlo.add %336, %345 : tensor<1xui32>
            %347 = stablehlo.broadcast_in_dim %iterArg_18, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %348 = stablehlo.add %344, %347 : tensor<1xui32>
            %349 = stablehlo.constant dense<1> : tensor<i32>
            %350 = stablehlo.add %iterArg_14, %349 : tensor<i32>
            %351 = stablehlo.convert %350 : (tensor<i32>) -> tensor<ui32>
            %352 = stablehlo.broadcast_in_dim %351, dims = [] : (tensor<ui32>) -> tensor<1xui32>
            %353 = stablehlo.add %348, %352 : tensor<1xui32>
            %354 = stablehlo.constant dense<1> : tensor<i32>
            %355 = stablehlo.add %iterArg_13, %354 : tensor<i32>
            stablehlo.return %355, %300, %346, %353, %iterArg_18, %iterArg_19, %iterArg_17, %iterArg_21, %iterArg_20 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
          }
          %272 = stablehlo.concatenate %271#2, %271#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
          %273 = stablehlo.constant dense<0> : tensor<i32>
          %274 = stablehlo.broadcast_in_dim %273, dims = [] : (tensor<i32>) -> tensor<1xi32>
          %275 = "stablehlo.gather"(%272, %274) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
          %276 = stablehlo.reshape %275 : (tensor<1xui32>) -> tensor<ui32>
          %277 = stablehlo.constant dense<9> : tensor<ui32>
          %278 = stablehlo.shift_right_logical %276, %277 : tensor<ui32>
          %279 = stablehlo.constant dense<1065353216> : tensor<ui32>
          %280 = stablehlo.or %278, %279 : tensor<ui32>
          %281 = stablehlo.bitcast_convert %280 : (tensor<ui32>) -> tensor<f32>
          %282 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
          %283 = stablehlo.subtract %281, %282 : tensor<f32>
          %284 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
          %285 = stablehlo.constant dense<-0.99999994> : tensor<f32>
          %286 = stablehlo.subtract %284, %285 : tensor<f32>
          %287 = stablehlo.multiply %283, %286 : tensor<f32>
          %288 = stablehlo.constant dense<-0.99999994> : tensor<f32>
          %289 = stablehlo.add %287, %288 : tensor<f32>
          %290 = stablehlo.reshape %289 : (tensor<f32>) -> tensor<f32>
          %291 = stablehlo.constant dense<-0.99999994> : tensor<f32>
          %292 = stablehlo.maximum %291, %290 : tensor<f32>
          %293 = func.call @erf_inv(%292) : (tensor<f32>) -> tensor<f32>
          %294 = stablehlo.constant dense<1.41421354> : tensor<f32>
          %295 = stablehlo.multiply %294, %293 : tensor<f32>
          %296 = stablehlo.multiply %295, %iterArg_9 : tensor<f32>
          %297 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
          %298 = stablehlo.add %297, %296 : tensor<f32>
          stablehlo.return %iterArg_9, %248, %295, %298 : tensor<f32>, tensor<2xui32>, tensor<f32>, tensor<f32>
        }
        %181 = stablehlo.multiply %180#2, %180#2 : tensor<f32>
        %182 = stablehlo.multiply %180#3, %180#3 : tensor<f32>
        %183 = stablehlo.multiply %182, %180#3 : tensor<f32>
        %184 = stablehlo.constant dense<0> : tensor<1xui32>
        %185 = stablehlo.iota dim = 0 : tensor<1xui32>
        %186 = "stablehlo.slice"(%177) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
        %187 = stablehlo.reshape %186 : (tensor<1xui32>) -> tensor<ui32>
        %188 = "stablehlo.slice"(%177) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
        %189 = stablehlo.reshape %188 : (tensor<1xui32>) -> tensor<ui32>
        %190 = stablehlo.concatenate %185, %184, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
        %191 = "stablehlo.slice"(%190) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
        %192 = "stablehlo.slice"(%190) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
        %193 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
        %194 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
        %195 = stablehlo.xor %187, %189 : tensor<ui32>
        %196 = stablehlo.constant dense<466688986> : tensor<ui32>
        %197 = stablehlo.xor %195, %196 : tensor<ui32>
        %198 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %199 = stablehlo.add %191, %198 : tensor<1xui32>
        %200 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<ui32>) -> tensor<1xui32>
        %201 = stablehlo.add %192, %200 : tensor<1xui32>
        %202 = stablehlo.constant dense<0> : tensor<i32>
        %203 = stablehlo.constant dense<0> : tensor<i32>
        %204:9 = stablehlo.while(%iterArg_9 = %203, %iterArg_10 = %202, %iterArg_11 = %199, %iterArg_12 = %201, %iterArg_13 = %189, %iterArg_14 = %197, %iterArg_15 = %187, %iterArg_16 = %193, %iterArg_17 = %194) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
         cond {
          %226 = stablehlo.constant dense<5> : tensor<i32>
          %227 = stablehlo.compare  LT, %iterArg_9, %226,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %227 : tensor<i1>
        } do {
          %226 = stablehlo.constant dense<1> : tensor<i32>
          %227 = stablehlo.add %iterArg_10, %226 : tensor<i32>
          %228 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %229 = stablehlo.reshape %228 : (tensor<1xui32>) -> tensor<ui32>
          %230 = stablehlo.add %iterArg_11, %iterArg_12 : tensor<1xui32>
          %231 = stablehlo.broadcast_in_dim %229, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %232 = stablehlo.shift_left %iterArg_12, %231 : tensor<1xui32>
          %233 = stablehlo.constant dense<32> : tensor<ui32>
          %234 = stablehlo.subtract %233, %229 : tensor<ui32>
          %235 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %236 = stablehlo.shift_right_logical %iterArg_12, %235 : tensor<1xui32>
          %237 = stablehlo.or %232, %236 : tensor<1xui32>
          %238 = stablehlo.xor %230, %237 : tensor<1xui32>
          %239 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %240 = stablehlo.reshape %239 : (tensor<1xui32>) -> tensor<ui32>
          %241 = stablehlo.add %230, %238 : tensor<1xui32>
          %242 = stablehlo.broadcast_in_dim %240, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %243 = stablehlo.shift_left %238, %242 : tensor<1xui32>
          %244 = stablehlo.constant dense<32> : tensor<ui32>
          %245 = stablehlo.subtract %244, %240 : tensor<ui32>
          %246 = stablehlo.broadcast_in_dim %245, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %247 = stablehlo.shift_right_logical %238, %246 : tensor<1xui32>
          %248 = stablehlo.or %243, %247 : tensor<1xui32>
          %249 = stablehlo.xor %241, %248 : tensor<1xui32>
          %250 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %251 = stablehlo.reshape %250 : (tensor<1xui32>) -> tensor<ui32>
          %252 = stablehlo.add %241, %249 : tensor<1xui32>
          %253 = stablehlo.broadcast_in_dim %251, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %254 = stablehlo.shift_left %249, %253 : tensor<1xui32>
          %255 = stablehlo.constant dense<32> : tensor<ui32>
          %256 = stablehlo.subtract %255, %251 : tensor<ui32>
          %257 = stablehlo.broadcast_in_dim %256, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %258 = stablehlo.shift_right_logical %249, %257 : tensor<1xui32>
          %259 = stablehlo.or %254, %258 : tensor<1xui32>
          %260 = stablehlo.xor %252, %259 : tensor<1xui32>
          %261 = "stablehlo.slice"(%iterArg_16) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
          %262 = stablehlo.reshape %261 : (tensor<1xui32>) -> tensor<ui32>
          %263 = stablehlo.add %252, %260 : tensor<1xui32>
          %264 = stablehlo.broadcast_in_dim %262, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %265 = stablehlo.shift_left %260, %264 : tensor<1xui32>
          %266 = stablehlo.constant dense<32> : tensor<ui32>
          %267 = stablehlo.subtract %266, %262 : tensor<ui32>
          %268 = stablehlo.broadcast_in_dim %267, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %269 = stablehlo.shift_right_logical %260, %268 : tensor<1xui32>
          %270 = stablehlo.or %265, %269 : tensor<1xui32>
          %271 = stablehlo.xor %263, %270 : tensor<1xui32>
          %272 = stablehlo.broadcast_in_dim %iterArg_13, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %273 = stablehlo.add %263, %272 : tensor<1xui32>
          %274 = stablehlo.broadcast_in_dim %iterArg_14, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %275 = stablehlo.add %271, %274 : tensor<1xui32>
          %276 = stablehlo.constant dense<1> : tensor<i32>
          %277 = stablehlo.add %iterArg_10, %276 : tensor<i32>
          %278 = stablehlo.convert %277 : (tensor<i32>) -> tensor<ui32>
          %279 = stablehlo.broadcast_in_dim %278, dims = [] : (tensor<ui32>) -> tensor<1xui32>
          %280 = stablehlo.add %275, %279 : tensor<1xui32>
          %281 = stablehlo.constant dense<1> : tensor<i32>
          %282 = stablehlo.add %iterArg_9, %281 : tensor<i32>
          stablehlo.return %282, %227, %273, %280, %iterArg_14, %iterArg_15, %iterArg_13, %iterArg_17, %iterArg_16 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
        }
        %205 = stablehlo.concatenate %204#2, %204#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
        %206 = stablehlo.constant dense<0> : tensor<i32>
        %207 = stablehlo.broadcast_in_dim %206, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %208 = "stablehlo.gather"(%205, %207) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
        %209 = stablehlo.reshape %208 : (tensor<1xui32>) -> tensor<ui32>
        %210 = stablehlo.constant dense<9> : tensor<ui32>
        %211 = stablehlo.shift_right_logical %209, %210 : tensor<ui32>
        %212 = stablehlo.constant dense<1065353216> : tensor<ui32>
        %213 = stablehlo.or %211, %212 : tensor<ui32>
        %214 = stablehlo.bitcast_convert %213 : (tensor<ui32>) -> tensor<f32>
        %215 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %216 = stablehlo.subtract %214, %215 : tensor<f32>
        %217 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %219 = stablehlo.subtract %217, %218 : tensor<f32>
        %220 = stablehlo.multiply %216, %219 : tensor<f32>
        %221 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %222 = stablehlo.add %220, %221 : tensor<f32>
        %223 = stablehlo.reshape %222 : (tensor<f32>) -> tensor<f32>
        %224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %225 = stablehlo.maximum %224, %223 : tensor<f32>
        stablehlo.return %iterArg_3, %iterArg_4, %173, %181, %183, %225 : tensor<f32>, tensor<f32>, tensor<2xui32>, tensor<f32>, tensor<f32>, tensor<f32>
      }
      %130 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %131 = stablehlo.divide %130, %48 : tensor<f32>
      %132 = stablehlo.power %125, %131 : tensor<f32>
      %133 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %134 = stablehlo.select %50, %133, %132 : tensor<i1>, tensor<f32>
      %135 = stablehlo.multiply %55, %129#4 : tensor<f32>
      %136 = stablehlo.multiply %135, %134 : tensor<f32>
      %137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %138 = stablehlo.compare  EQ, %136, %137,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %139 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
      %140 = stablehlo.select %138, %139, %136 : tensor<i1>, tensor<f32>
      %141 = stablehlo.broadcast_in_dim %140, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %142 = stablehlo.constant dense<0> : tensor<i32>
      %143 = stablehlo.compare  LT, %iterArg_1, %142,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %144 = stablehlo.convert %iterArg_1 : tensor<i32>
      %145 = stablehlo.constant dense<3> : tensor<i32>
      %146 = stablehlo.add %144, %145 : tensor<i32>
      %147 = stablehlo.select %143, %146, %iterArg_1 : tensor<i1>, tensor<i32>
      %148 = stablehlo.dynamic_update_slice %iterArg_2, %141, %147 : (tensor<3xf32>, tensor<1xf32>, tensor<i32>) -> tensor<3xf32>
      %149 = stablehlo.constant dense<1> : tensor<i32>
      %150 = stablehlo.add %iterArg_1, %149 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_0, %150, %148 : tensor<3x2xui32>, tensor<3xf32>, tensor<i32>, tensor<3xf32>
    }
    return %31#3 : tensor<3xf32>
  }
  func.func private @erf_inv(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = call @xla_fallback_erf_inv(%arg0) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @xla_fallback_erf_inv(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.abs %arg0 : tensor<f32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.compare  EQ, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<f32>
    %5 = stablehlo.negate %arg0 : tensor<f32>
    %6 = stablehlo.multiply %5, %arg0 : tensor<f32>
    %7 = stablehlo.log_plus_one %6 : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %10 = stablehlo.compare  LT, %8, %9 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = stablehlo.constant dense<1.50140941> : tensor<f32>
    %12 = stablehlo.constant dense<2.83297682> : tensor<f32>
    %13 = stablehlo.select %10, %11, %12 : tensor<i1>, tensor<f32>
    %14 = stablehlo.constant dense<0.246640727> : tensor<f32>
    %15 = stablehlo.constant dense<1.00167406> : tensor<f32>
    %16 = stablehlo.select %10, %14, %15 : tensor<i1>, tensor<f32>
    %17 = stablehlo.constant dense<-0.00417768164> : tensor<f32>
    %18 = stablehlo.constant dense<0.00943887047> : tensor<f32>
    %19 = stablehlo.select %10, %17, %18 : tensor<i1>, tensor<f32>
    %20 = stablehlo.constant dense<-0.00125372503> : tensor<f32>
    %21 = stablehlo.constant dense<-0.0076224613> : tensor<f32>
    %22 = stablehlo.select %10, %20, %21 : tensor<i1>, tensor<f32>
    %23 = stablehlo.constant dense<2.1858087E-4> : tensor<f32>
    %24 = stablehlo.constant dense<0.00573950773> : tensor<f32>
    %25 = stablehlo.select %10, %23, %24 : tensor<i1>, tensor<f32>
    %26 = stablehlo.constant dense<-4.39150654E-6> : tensor<f32>
    %27 = stablehlo.constant dense<-0.00367342844> : tensor<f32>
    %28 = stablehlo.select %10, %26, %27 : tensor<i1>, tensor<f32>
    %29 = stablehlo.constant dense<-3.5233877E-6> : tensor<f32>
    %30 = stablehlo.constant dense<0.00134934322> : tensor<f32>
    %31 = stablehlo.select %10, %29, %30 : tensor<i1>, tensor<f32>
    %32 = stablehlo.constant dense<3.43273939E-7> : tensor<f32>
    %33 = stablehlo.constant dense<1.00950558E-4> : tensor<f32>
    %34 = stablehlo.select %10, %32, %33 : tensor<i1>, tensor<f32>
    %35 = stablehlo.constant dense<2.81022636E-8> : tensor<f32>
    %36 = stablehlo.constant dense<-2.00214257E-4> : tensor<f32>
    %37 = stablehlo.select %10, %35, %36 : tensor<i1>, tensor<f32>
    %38 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %39 = stablehlo.subtract %8, %38 : tensor<f32>
    %40 = stablehlo.sqrt %8 : tensor<f32>
    %41 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %42 = stablehlo.subtract %40, %41 : tensor<f32>
    %43 = stablehlo.select %10, %39, %42 : tensor<i1>, tensor<f32>
    %44 = stablehlo.multiply %37, %43 : tensor<f32>
    %45 = stablehlo.add %34, %44 : tensor<f32>
    %46 = stablehlo.multiply %45, %43 : tensor<f32>
    %47 = stablehlo.add %31, %46 : tensor<f32>
    %48 = stablehlo.multiply %47, %43 : tensor<f32>
    %49 = stablehlo.add %28, %48 : tensor<f32>
    %50 = stablehlo.multiply %49, %43 : tensor<f32>
    %51 = stablehlo.add %25, %50 : tensor<f32>
    %52 = stablehlo.multiply %51, %43 : tensor<f32>
    %53 = stablehlo.add %22, %52 : tensor<f32>
    %54 = stablehlo.multiply %53, %43 : tensor<f32>
    %55 = stablehlo.add %19, %54 : tensor<f32>
    %56 = stablehlo.multiply %55, %43 : tensor<f32>
    %57 = stablehlo.add %16, %56 : tensor<f32>
    %58 = stablehlo.multiply %57, %43 : tensor<f32>
    %59 = stablehlo.add %13, %58 : tensor<f32>
    %60 = stablehlo.multiply %59, %arg0 : tensor<f32>
    %61 = stablehlo.select %2, %4, %60 : tensor<i1>, tensor<f32>
    return %61 : tensor<f32>
  }
}
