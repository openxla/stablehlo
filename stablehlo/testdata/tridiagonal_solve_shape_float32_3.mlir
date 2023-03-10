// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:4 = call @inputs() : () -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>)
    %1 = call @expected() : () -> tensor<3x1xf32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.dynamic_slice %0#2, %2, sizes = [1] : (tensor<3xf32>, tensor<i32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.dynamic_slice %0#1, %5, sizes = [1] : (tensor<3xf32>, tensor<i32>) -> tensor<1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
    %8 = stablehlo.divide %4, %7 : tensor<f32>
    %9 = "stablehlo.slice"(%0#1) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %10 = "stablehlo.slice"(%0#1) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %11 = "stablehlo.slice"(%0#2) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %12 = "stablehlo.slice"(%0#2) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %13 = "stablehlo.slice"(%0#0) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %14 = "stablehlo.slice"(%0#0) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %15 = stablehlo.reshape %9 : (tensor<0xf32>) -> tensor<0x32xf32>
    %16 = stablehlo.reshape %11 : (tensor<0xf32>) -> tensor<0x32xf32>
    %17 = stablehlo.reshape %13 : (tensor<0xf32>) -> tensor<0x32xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<0x32xf32>
    %20 = stablehlo.reshape %19 : (tensor<0x32xf32>) -> tensor<0xf32>
    %21 = "stablehlo.slice"(%10) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %22 = stablehlo.reshape %21 : (tensor<1xf32>) -> tensor<f32>
    %23 = "stablehlo.slice"(%12) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %24 = stablehlo.reshape %23 : (tensor<1xf32>) -> tensor<f32>
    %25 = "stablehlo.slice"(%14) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %26 = stablehlo.reshape %25 : (tensor<1xf32>) -> tensor<f32>
    %27 = stablehlo.multiply %26, %8 : tensor<f32>
    %28 = stablehlo.subtract %22, %27 : tensor<f32>
    %29 = stablehlo.divide %24, %28 : tensor<f32>
    %30 = stablehlo.multiply %26, %8 : tensor<f32>
    %31 = stablehlo.subtract %22, %30 : tensor<f32>
    %32 = stablehlo.divide %24, %31 : tensor<f32>
    %33 = "stablehlo.slice"(%10) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %34 = stablehlo.reshape %33 : (tensor<1xf32>) -> tensor<f32>
    %35 = "stablehlo.slice"(%12) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %36 = stablehlo.reshape %35 : (tensor<1xf32>) -> tensor<f32>
    %37 = "stablehlo.slice"(%14) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %38 = stablehlo.reshape %37 : (tensor<1xf32>) -> tensor<f32>
    %39 = stablehlo.multiply %38, %29 : tensor<f32>
    %40 = stablehlo.subtract %34, %39 : tensor<f32>
    %41 = stablehlo.divide %36, %40 : tensor<f32>
    %42 = stablehlo.multiply %38, %29 : tensor<f32>
    %43 = stablehlo.subtract %34, %42 : tensor<f32>
    %44 = stablehlo.divide %36, %43 : tensor<f32>
    %45 = "stablehlo.slice"(%10) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %46 = stablehlo.reshape %45 : (tensor<1xf32>) -> tensor<f32>
    %47 = "stablehlo.slice"(%12) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %48 = stablehlo.reshape %47 : (tensor<1xf32>) -> tensor<f32>
    %49 = "stablehlo.slice"(%14) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %50 = stablehlo.reshape %49 : (tensor<1xf32>) -> tensor<f32>
    %51 = stablehlo.multiply %50, %41 : tensor<f32>
    %52 = stablehlo.subtract %46, %51 : tensor<f32>
    %53 = stablehlo.divide %48, %52 : tensor<f32>
    %54 = stablehlo.multiply %50, %41 : tensor<f32>
    %55 = stablehlo.subtract %46, %54 : tensor<f32>
    %56 = stablehlo.divide %48, %55 : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %58 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %59 = stablehlo.broadcast_in_dim %56, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %60 = stablehlo.concatenate %57, %58, %59, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %61 = stablehlo.concatenate %20, %60, dim = 0 : (tensor<0xf32>, tensor<3xf32>) -> tensor<3xf32>
    %62 = stablehlo.constant dense<0> : tensor<i32>
    %63 = stablehlo.constant dense<0> : tensor<i32>
    %64 = stablehlo.dynamic_slice %0#3, %62, %63, sizes = [1, 1] : (tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %65 = stablehlo.reshape %64 : (tensor<1x1xf32>) -> tensor<1xf32>
    %66 = stablehlo.constant dense<0> : tensor<i32>
    %67 = stablehlo.dynamic_slice %0#1, %66, sizes = [1] : (tensor<3xf32>, tensor<i32>) -> tensor<1xf32>
    %68 = stablehlo.reshape %67 : (tensor<1xf32>) -> tensor<f32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %70 = stablehlo.divide %65, %69 : tensor<1xf32>
    %71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %73 = stablehlo.constant dense<0> : tensor<i32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %75 = "stablehlo.gather"(%61, %74) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<2> : tensor<1xi64>} : (tensor<3xf32>, tensor<1xi32>) -> tensor<2xf32>
    %76 = call @append(%72, %75) : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
    %77 = "stablehlo.slice"(%0#3) {limit_indices = dense<[0, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<0x1xf32>
    %78 = "stablehlo.slice"(%0#3) {limit_indices = dense<[3, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %79 = "stablehlo.slice"(%0#1) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %80 = "stablehlo.slice"(%0#1) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %81 = "stablehlo.slice"(%76) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %82 = "stablehlo.slice"(%76) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %83 = "stablehlo.slice"(%0#0) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %84 = "stablehlo.slice"(%0#0) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %85 = stablehlo.reshape %77 : (tensor<0x1xf32>) -> tensor<0x32x1xf32>
    %86 = stablehlo.reshape %79 : (tensor<0xf32>) -> tensor<0x32xf32>
    %87 = stablehlo.reshape %81 : (tensor<0xf32>) -> tensor<0x32xf32>
    %88 = stablehlo.reshape %83 : (tensor<0xf32>) -> tensor<0x32xf32>
    %89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<0x32x1xf32>
    %91 = stablehlo.reshape %90 : (tensor<0x32x1xf32>) -> tensor<0x1xf32>
    %92 = "stablehlo.slice"(%78) {limit_indices = dense<1> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %93 = stablehlo.reshape %92 : (tensor<1x1xf32>) -> tensor<1xf32>
    %94 = "stablehlo.slice"(%80) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %95 = stablehlo.reshape %94 : (tensor<1xf32>) -> tensor<f32>
    %96 = "stablehlo.slice"(%82) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %97 = stablehlo.reshape %96 : (tensor<1xf32>) -> tensor<f32>
    %98 = "stablehlo.slice"(%84) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %99 = stablehlo.reshape %98 : (tensor<1xf32>) -> tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %101 = stablehlo.multiply %100, %70 : tensor<1xf32>
    %102 = stablehlo.subtract %93, %101 : tensor<1xf32>
    %103 = stablehlo.multiply %99, %97 : tensor<f32>
    %104 = stablehlo.subtract %95, %103 : tensor<f32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %106 = stablehlo.divide %102, %105 : tensor<1xf32>
    %107 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %108 = stablehlo.multiply %107, %70 : tensor<1xf32>
    %109 = stablehlo.subtract %93, %108 : tensor<1xf32>
    %110 = stablehlo.multiply %99, %97 : tensor<f32>
    %111 = stablehlo.subtract %95, %110 : tensor<f32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %113 = stablehlo.divide %109, %112 : tensor<1xf32>
    %114 = "stablehlo.slice"(%78) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %115 = stablehlo.reshape %114 : (tensor<1x1xf32>) -> tensor<1xf32>
    %116 = "stablehlo.slice"(%80) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %117 = stablehlo.reshape %116 : (tensor<1xf32>) -> tensor<f32>
    %118 = "stablehlo.slice"(%82) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %119 = stablehlo.reshape %118 : (tensor<1xf32>) -> tensor<f32>
    %120 = "stablehlo.slice"(%84) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %121 = stablehlo.reshape %120 : (tensor<1xf32>) -> tensor<f32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %123 = stablehlo.multiply %122, %106 : tensor<1xf32>
    %124 = stablehlo.subtract %115, %123 : tensor<1xf32>
    %125 = stablehlo.multiply %121, %119 : tensor<f32>
    %126 = stablehlo.subtract %117, %125 : tensor<f32>
    %127 = stablehlo.broadcast_in_dim %126, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %128 = stablehlo.divide %124, %127 : tensor<1xf32>
    %129 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %130 = stablehlo.multiply %129, %106 : tensor<1xf32>
    %131 = stablehlo.subtract %115, %130 : tensor<1xf32>
    %132 = stablehlo.multiply %121, %119 : tensor<f32>
    %133 = stablehlo.subtract %117, %132 : tensor<f32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %135 = stablehlo.divide %131, %134 : tensor<1xf32>
    %136 = "stablehlo.slice"(%78) {limit_indices = dense<[3, 1]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %137 = stablehlo.reshape %136 : (tensor<1x1xf32>) -> tensor<1xf32>
    %138 = "stablehlo.slice"(%80) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %139 = stablehlo.reshape %138 : (tensor<1xf32>) -> tensor<f32>
    %140 = "stablehlo.slice"(%82) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %141 = stablehlo.reshape %140 : (tensor<1xf32>) -> tensor<f32>
    %142 = "stablehlo.slice"(%84) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %143 = stablehlo.reshape %142 : (tensor<1xf32>) -> tensor<f32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %145 = stablehlo.multiply %144, %128 : tensor<1xf32>
    %146 = stablehlo.subtract %137, %145 : tensor<1xf32>
    %147 = stablehlo.multiply %143, %141 : tensor<f32>
    %148 = stablehlo.subtract %139, %147 : tensor<f32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %150 = stablehlo.divide %146, %149 : tensor<1xf32>
    %151 = stablehlo.broadcast_in_dim %143, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %152 = stablehlo.multiply %151, %128 : tensor<1xf32>
    %153 = stablehlo.subtract %137, %152 : tensor<1xf32>
    %154 = stablehlo.multiply %143, %141 : tensor<f32>
    %155 = stablehlo.subtract %139, %154 : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %157 = stablehlo.divide %153, %156 : tensor<1xf32>
    %158 = stablehlo.broadcast_in_dim %113, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %159 = stablehlo.broadcast_in_dim %135, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %160 = stablehlo.broadcast_in_dim %157, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %161 = stablehlo.concatenate %158, %159, %160, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    %162 = stablehlo.concatenate %91, %161, dim = 0 : (tensor<0x1xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
    %163 = stablehlo.constant dense<-1> : tensor<i32>
    %164 = stablehlo.constant dense<3> : tensor<i32>
    %165 = stablehlo.add %163, %164 : tensor<i32>
    %166 = stablehlo.convert %165 : tensor<i32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %168 = "stablehlo.gather"(%162, %167) {dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>, tensor<1xi32>) -> tensor<1xf32>
    %169 = stablehlo.reverse %162, dims = [0] : tensor<3x1xf32>
    %170 = stablehlo.reverse %61, dims = [0] : tensor<3xf32>
    %171 = "stablehlo.slice"(%169) {limit_indices = dense<[0, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<0x1xf32>
    %172 = "stablehlo.slice"(%169) {limit_indices = dense<[3, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %173 = "stablehlo.slice"(%170) {limit_indices = dense<0> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<0xf32>
    %174 = "stablehlo.slice"(%170) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<3xf32>
    %175 = stablehlo.reshape %171 : (tensor<0x1xf32>) -> tensor<0x32x1xf32>
    %176 = stablehlo.reshape %173 : (tensor<0xf32>) -> tensor<0x32xf32>
    %177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<f32>) -> tensor<0x32x1xf32>
    %179 = stablehlo.reshape %178 : (tensor<0x32x1xf32>) -> tensor<0x1xf32>
    %180 = "stablehlo.slice"(%172) {limit_indices = dense<1> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %181 = stablehlo.reshape %180 : (tensor<1x1xf32>) -> tensor<1xf32>
    %182 = "stablehlo.slice"(%174) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %183 = stablehlo.reshape %182 : (tensor<1xf32>) -> tensor<f32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %185 = stablehlo.multiply %184, %168 : tensor<1xf32>
    %186 = stablehlo.subtract %181, %185 : tensor<1xf32>
    %187 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %188 = stablehlo.multiply %187, %168 : tensor<1xf32>
    %189 = stablehlo.subtract %181, %188 : tensor<1xf32>
    %190 = "stablehlo.slice"(%172) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %191 = stablehlo.reshape %190 : (tensor<1x1xf32>) -> tensor<1xf32>
    %192 = "stablehlo.slice"(%174) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %193 = stablehlo.reshape %192 : (tensor<1xf32>) -> tensor<f32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %195 = stablehlo.multiply %194, %186 : tensor<1xf32>
    %196 = stablehlo.subtract %191, %195 : tensor<1xf32>
    %197 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %198 = stablehlo.multiply %197, %186 : tensor<1xf32>
    %199 = stablehlo.subtract %191, %198 : tensor<1xf32>
    %200 = "stablehlo.slice"(%172) {limit_indices = dense<[3, 1]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %201 = stablehlo.reshape %200 : (tensor<1x1xf32>) -> tensor<1xf32>
    %202 = "stablehlo.slice"(%174) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<1xf32>
    %203 = stablehlo.reshape %202 : (tensor<1xf32>) -> tensor<f32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %205 = stablehlo.multiply %204, %196 : tensor<1xf32>
    %206 = stablehlo.subtract %201, %205 : tensor<1xf32>
    %207 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %208 = stablehlo.multiply %207, %196 : tensor<1xf32>
    %209 = stablehlo.subtract %201, %208 : tensor<1xf32>
    %210 = stablehlo.broadcast_in_dim %189, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %211 = stablehlo.broadcast_in_dim %199, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %212 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %213 = stablehlo.concatenate %210, %211, %212, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    %214 = stablehlo.concatenate %179, %213, dim = 0 : (tensor<0x1xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
    %215 = stablehlo.reverse %214, dims = [0] : tensor<3x1xf32>
    %216 = stablehlo.custom_call @check.eq(%215, %1) : (tensor<3x1xf32>, tensor<3x1xf32>) -> tensor<i1>
    return %216 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>) {
    %0 = stablehlo.constant dense<[0.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %2 = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 0.000000e+00]> : tensor<3xf32>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<3x1xf32>
    return %0, %1, %2, %3 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>
  }
  func.func private @expected() -> tensor<3x1xf32> {
    %0 = stablehlo.constant dense<[[0.571428597], [0.428571403], [-0.285714298]]> : tensor<3x1xf32>
    return %0 : tensor<3x1xf32>
  }
  func.func private @append(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<3xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
