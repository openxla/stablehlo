// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<9xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<9xbf16>, tensor<9xbf16>, tensor<9xbf16>)
    %1 = call @expected() : () -> tensor<9xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<9xbf16>) -> tensor<9xf32>
    %3 = stablehlo.convert %0#1 : (tensor<9xbf16>) -> tensor<9xf32>
    %4 = stablehlo.convert %0#2 : (tensor<9xbf16>) -> tensor<9xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %6 = stablehlo.compare  LE, %2, %5,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %8 = stablehlo.compare  LE, %3, %7,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %9 = stablehlo.or %6, %8 : tensor<9xi1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %11 = stablehlo.compare  LT, %4, %10,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %12 = stablehlo.or %9, %11 : tensor<9xi1>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %14 = stablehlo.compare  GT, %4, %13,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %15 = stablehlo.or %12, %14 : tensor<9xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %17 = stablehlo.add %2, %16 : tensor<9xf32>
    %18 = stablehlo.add %2, %3 : tensor<9xf32>
    %cst_4 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %20 = stablehlo.add %18, %19 : tensor<9xf32>
    %21 = stablehlo.divide %17, %20 : tensor<9xf32>
    %22 = stablehlo.compare  LT, %4, %21,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %23 = stablehlo.select %22, %2, %3 : tensor<9xi1>, tensor<9xf32>
    %24 = stablehlo.select %22, %3, %2 : tensor<9xi1>, tensor<9xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %26 = stablehlo.subtract %25, %4 : tensor<9xf32>
    %27 = stablehlo.select %22, %4, %26 : tensor<9xi1>, tensor<9xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %28 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %29 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %30 = stablehlo.compare  EQ, %28, %29,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %33 = stablehlo.select %30, %31, %32 : tensor<9xi1>, tensor<9xf32>
    %34 = stablehlo.abs %33 : tensor<9xf32>
    %cst_9 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %36 = stablehlo.compare  LT, %34, %35,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %cst_10 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %38 = stablehlo.select %36, %37, %33 : tensor<9xi1>, tensor<9xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %c_12 = stablehlo.constant dense<1> : tensor<i64>
    %c_13 = stablehlo.constant dense<true> : tensor<i1>
    %40:8 = stablehlo.while(%iterArg = %23, %iterArg_16 = %24, %iterArg_17 = %27, %iterArg_18 = %c_12, %iterArg_19 = %c_13, %iterArg_20 = %38, %iterArg_21 = %39, %iterArg_22 = %38) : tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>
     cond {
      %c_23 = stablehlo.constant dense<200> : tensor<i64>
      %63 = stablehlo.compare  LT, %iterArg_18, %c_23,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %64 = stablehlo.and %63, %iterArg_19 : tensor<i1>
      stablehlo.return %64 : tensor<i1>
    } do {
      %63 = stablehlo.broadcast_in_dim %iterArg_18, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %c_23 = stablehlo.constant dense<2> : tensor<i64>
      %64 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %65 = func.call @remainder(%63, %64) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %c_24 = stablehlo.constant dense<0> : tensor<i64>
      %66 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %67 = stablehlo.compare  EQ, %65, %66,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %c_25 = stablehlo.constant dense<1> : tensor<i64>
      %68 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %69 = stablehlo.compare  EQ, %63, %68,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %c_26 = stablehlo.constant dense<1> : tensor<i64>
      %70 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %71 = stablehlo.subtract %63, %70 : tensor<9xi64>
      %c_27 = stablehlo.constant dense<2> : tensor<i64>
      %72 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %73 = func.call @floor_divide(%71, %72) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %74 = stablehlo.convert %73 : (tensor<9xi64>) -> tensor<9xf32>
      %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %75 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %cst_29 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %76 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %77 = stablehlo.add %iterArg, %74 : tensor<9xf32>
      %78 = stablehlo.negate %77 : tensor<9xf32>
      %79 = stablehlo.add %iterArg, %iterArg_16 : tensor<9xf32>
      %80 = stablehlo.add %79, %74 : tensor<9xf32>
      %81 = stablehlo.multiply %78, %80 : tensor<9xf32>
      %82 = stablehlo.multiply %81, %iterArg_17 : tensor<9xf32>
      %83 = stablehlo.multiply %76, %74 : tensor<9xf32>
      %84 = stablehlo.add %iterArg, %83 : tensor<9xf32>
      %85 = stablehlo.multiply %76, %74 : tensor<9xf32>
      %86 = stablehlo.add %iterArg, %85 : tensor<9xf32>
      %87 = stablehlo.add %86, %75 : tensor<9xf32>
      %88 = stablehlo.multiply %84, %87 : tensor<9xf32>
      %89 = stablehlo.divide %82, %88 : tensor<9xf32>
      %90 = stablehlo.subtract %iterArg_16, %74 : tensor<9xf32>
      %91 = stablehlo.multiply %74, %90 : tensor<9xf32>
      %92 = stablehlo.multiply %91, %iterArg_17 : tensor<9xf32>
      %93 = stablehlo.multiply %76, %74 : tensor<9xf32>
      %94 = stablehlo.add %iterArg, %93 : tensor<9xf32>
      %95 = stablehlo.subtract %94, %75 : tensor<9xf32>
      %96 = stablehlo.multiply %76, %74 : tensor<9xf32>
      %97 = stablehlo.add %iterArg, %96 : tensor<9xf32>
      %98 = stablehlo.multiply %95, %97 : tensor<9xf32>
      %99 = stablehlo.divide %92, %98 : tensor<9xf32>
      %cst_30 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %100 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %101 = stablehlo.select %67, %89, %99 : tensor<9xi1>, tensor<9xf32>
      %102 = stablehlo.select %69, %100, %101 : tensor<9xi1>, tensor<9xf32>
      %103 = stablehlo.broadcast_in_dim %iterArg_18, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %c_31 = stablehlo.constant dense<0> : tensor<i64>
      %104 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %105 = stablehlo.compare  EQ, %103, %104,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %106 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %cst_33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %107 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %108 = stablehlo.select %105, %106, %107 : tensor<9xi1>, tensor<9xf32>
      %109 = stablehlo.divide %102, %iterArg_20 : tensor<9xf32>
      %110 = stablehlo.add %108, %109 : tensor<9xf32>
      %cst_34 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
      %111 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %112 = stablehlo.abs %110 : tensor<9xf32>
      %113 = stablehlo.compare  LT, %112, %111,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %114 = stablehlo.select %113, %111, %110 : tensor<9xi1>, tensor<9xf32>
      %115 = stablehlo.multiply %102, %iterArg_21 : tensor<9xf32>
      %116 = stablehlo.add %108, %115 : tensor<9xf32>
      %117 = stablehlo.abs %116 : tensor<9xf32>
      %118 = stablehlo.compare  LT, %117, %111,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %119 = stablehlo.select %118, %111, %116 : tensor<9xi1>, tensor<9xf32>
      %120 = func.call @integer_pow(%119) : (tensor<9xf32>) -> tensor<9xf32>
      %121 = stablehlo.multiply %114, %120 : tensor<9xf32>
      %122 = stablehlo.multiply %iterArg_22, %121 : tensor<9xf32>
      %c_35 = stablehlo.constant dense<1> : tensor<i64>
      %123 = stablehlo.add %iterArg_18, %c_35 : tensor<i64>
      %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %124 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %125 = stablehlo.subtract %121, %124 : tensor<9xf32>
      %126 = stablehlo.abs %125 : tensor<9xf32>
      %cst_37 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
      %127 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %128 = stablehlo.compare  GE, %126, %127,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %c_38 = stablehlo.constant dense<false> : tensor<i1>
      %129 = stablehlo.reduce(%128 init: %c_38) applies stablehlo.or across dimensions = [0] : (tensor<9xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %iterArg, %iterArg_16, %iterArg_17, %123, %129, %114, %120, %122 : tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>
    }
    %41 = chlo.lgamma %23 : tensor<9xf32> -> tensor<9xf32>
    %42 = chlo.lgamma %24 : tensor<9xf32> -> tensor<9xf32>
    %43 = stablehlo.add %41, %42 : tensor<9xf32>
    %44 = stablehlo.add %23, %24 : tensor<9xf32>
    %45 = chlo.lgamma %44 : tensor<9xf32> -> tensor<9xf32>
    %46 = stablehlo.subtract %43, %45 : tensor<9xf32>
    %47 = stablehlo.log %27 : tensor<9xf32>
    %48 = stablehlo.multiply %47, %23 : tensor<9xf32>
    %49 = stablehlo.negate %27 : tensor<9xf32>
    %50 = stablehlo.log_plus_one %49 : tensor<9xf32>
    %51 = stablehlo.multiply %50, %24 : tensor<9xf32>
    %52 = stablehlo.add %48, %51 : tensor<9xf32>
    %53 = stablehlo.subtract %52, %46 : tensor<9xf32>
    %54 = stablehlo.exponential %53 : tensor<9xf32>
    %55 = stablehlo.multiply %40#7, %54 : tensor<9xf32>
    %56 = stablehlo.divide %55, %23 : tensor<9xf32>
    %cst_14 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %58 = stablehlo.select %15, %57, %56 : tensor<9xi1>, tensor<9xf32>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %60 = stablehlo.subtract %59, %58 : tensor<9xf32>
    %61 = stablehlo.select %22, %58, %60 : tensor<9xi1>, tensor<9xf32>
    %62 = stablehlo.convert %61 : (tensor<9xf32>) -> tensor<9xbf16>
    stablehlo.custom_call @check.expect_close(%62, %1) {has_side_effect = true} : (tensor<9xbf16>, tensor<9xbf16>) -> ()
    return %62 : tensor<9xbf16>
  }
  func.func private @inputs() -> (tensor<9xbf16> {mhlo.layout_mode = "default"}, tensor<9xbf16> {mhlo.layout_mode = "default"}, tensor<9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-1.601560e+00, -1.398440e+00, -1.000000e+00, 0.000000e+00, 1.000980e-01, 3.007810e-01, 1.000000e+00, 1.398440e+00, 1.601560e+00]> : tensor<9xbf16>
    %cst_0 = stablehlo.constant dense<[-1.601560e+00, 1.398440e+00, 1.000000e+00, 0.000000e+00, 2.001950e-01, 1.000980e-01, 1.000000e+00, 1.398440e+00, -1.601560e+00]> : tensor<9xbf16>
    %cst_1 = stablehlo.constant dense<[1.000000e+00, -1.000000e+00, 2.000000e+00, 1.000000e+00, 3.007810e-01, 3.007810e-01, -1.000000e+00, 2.406250e+00, 1.601560e+00]> : tensor<9xbf16>
    return %cst, %cst_0, %cst_1 : tensor<9xbf16>, tensor<9xbf16>, tensor<9xbf16>
  }
  func.func private @expected() -> (tensor<9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0x7FC0, 0x7FC0, 0x7FC0, 0x7FC0, 6.210940e-01, 1.943360e-01, 0x7FC0, 0x7FC0, 0x7FC0]> : tensor<9xbf16>
    return %cst : tensor<9xbf16>
  }
  func.func private @remainder(%arg0: tensor<9xi64> {mhlo.layout_mode = "default"}, %arg1: tensor<9xi64> {mhlo.layout_mode = "default"}) -> (tensor<9xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %1 = stablehlo.compare  EQ, %arg1, %0,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %3 = call @_where(%1, %2, %arg1) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %4 = stablehlo.remainder %arg0, %3 : tensor<9xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %6 = stablehlo.compare  NE, %4, %5,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %8 = stablehlo.compare  LT, %4, %7,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %10 = stablehlo.compare  LT, %3, %9,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %11 = stablehlo.compare  NE, %8, %10,  UNSIGNED : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %12 = stablehlo.and %11, %6 : tensor<9xi1>
    %13 = stablehlo.add %4, %3 : tensor<9xi64>
    %14 = stablehlo.select %12, %13, %4 : tensor<9xi1>, tensor<9xi64>
    return %14 : tensor<9xi64>
  }
  func.func private @_where(%arg0: tensor<9xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<9xi64> {mhlo.layout_mode = "default"}, %arg2: tensor<9xi64> {mhlo.layout_mode = "default"}) -> (tensor<9xi64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<9xi1>, tensor<9xi64>
    return %0 : tensor<9xi64>
  }
  func.func private @floor_divide(%arg0: tensor<9xi64> {mhlo.layout_mode = "default"}, %arg1: tensor<9xi64> {mhlo.layout_mode = "default"}) -> (tensor<9xi64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<9xi64>
    %1 = stablehlo.sign %arg0 : tensor<9xi64>
    %2 = stablehlo.sign %arg1 : tensor<9xi64>
    %3 = stablehlo.compare  NE, %1, %2,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %4 = stablehlo.remainder %arg0, %arg1 : tensor<9xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %6 = stablehlo.compare  NE, %4, %5,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %c_0 = stablehlo.constant dense<false> : tensor<i1>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i1>) -> tensor<9xi1>
    %8 = stablehlo.compare  NE, %3, %7,  UNSIGNED : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %9 = stablehlo.convert %8 : tensor<9xi1>
    %c_1 = stablehlo.constant dense<false> : tensor<i1>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i1>) -> tensor<9xi1>
    %11 = stablehlo.compare  NE, %6, %10,  UNSIGNED : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %12 = stablehlo.convert %11 : tensor<9xi1>
    %13 = stablehlo.and %9, %12 : tensor<9xi1>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %14 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %15 = stablehlo.subtract %0, %14 : tensor<9xi64>
    %16 = call @_where_0(%13, %15, %0) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    return %16 : tensor<9xi64>
  }
  func.func private @_where_0(%arg0: tensor<9xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<9xi64> {mhlo.layout_mode = "default"}, %arg2: tensor<9xi64> {mhlo.layout_mode = "default"}) -> (tensor<9xi64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<9xi1>, tensor<9xi64>
    return %0 : tensor<9xi64>
  }
  func.func private @integer_pow(%arg0: tensor<9xf32>) -> tensor<9xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %1 = stablehlo.divide %0, %arg0 : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
}
