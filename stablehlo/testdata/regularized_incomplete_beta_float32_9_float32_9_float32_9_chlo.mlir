// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<9xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<9xf32>, tensor<9xf32>, tensor<9xf32>)
    %1 = call @expected() : () -> tensor<9xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %3 = stablehlo.compare  LE, %0#0, %2,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %5 = stablehlo.compare  LE, %0#1, %4,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %6 = stablehlo.or %3, %5 : tensor<9xi1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %8 = stablehlo.compare  LT, %0#2, %7,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %9 = stablehlo.or %6, %8 : tensor<9xi1>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %11 = stablehlo.compare  GT, %0#2, %10,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %12 = stablehlo.or %9, %11 : tensor<9xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %14 = stablehlo.add %0#0, %13 : tensor<9xf32>
    %15 = stablehlo.add %0#0, %0#1 : tensor<9xf32>
    %cst_4 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %17 = stablehlo.add %15, %16 : tensor<9xf32>
    %18 = stablehlo.divide %14, %17 : tensor<9xf32>
    %19 = stablehlo.compare  LT, %0#2, %18,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %20 = stablehlo.select %19, %0#0, %0#1 : tensor<9xi1>, tensor<9xf32>
    %21 = stablehlo.select %19, %0#1, %0#0 : tensor<9xi1>, tensor<9xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %23 = stablehlo.subtract %22, %0#2 : tensor<9xf32>
    %24 = stablehlo.select %19, %0#2, %23 : tensor<9xi1>, tensor<9xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %25 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %26 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<9xi64>
    %27 = stablehlo.compare  EQ, %25, %26,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %30 = stablehlo.select %27, %28, %29 : tensor<9xi1>, tensor<9xf32>
    %31 = stablehlo.abs %30 : tensor<9xf32>
    %cst_9 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %33 = stablehlo.compare  LT, %31, %32,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %cst_10 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %35 = stablehlo.select %33, %34, %30 : tensor<9xi1>, tensor<9xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %c_12 = stablehlo.constant dense<1> : tensor<i64>
    %c_13 = stablehlo.constant dense<true> : tensor<i1>
    %37:8 = stablehlo.while(%iterArg = %20, %iterArg_16 = %21, %iterArg_17 = %24, %iterArg_18 = %c_12, %iterArg_19 = %c_13, %iterArg_20 = %35, %iterArg_21 = %36, %iterArg_22 = %35) : tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>
     cond {
      %c_23 = stablehlo.constant dense<200> : tensor<i64>
      %59 = stablehlo.compare  LT, %iterArg_18, %c_23,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %60 = stablehlo.and %59, %iterArg_19 : tensor<i1>
      stablehlo.return %60 : tensor<i1>
    } do {
      %59 = stablehlo.broadcast_in_dim %iterArg_18, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %c_23 = stablehlo.constant dense<2> : tensor<i64>
      %60 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %61 = func.call @remainder(%59, %60) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %c_24 = stablehlo.constant dense<0> : tensor<i64>
      %62 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %63 = stablehlo.compare  EQ, %61, %62,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %c_25 = stablehlo.constant dense<1> : tensor<i64>
      %64 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %65 = stablehlo.compare  EQ, %59, %64,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %c_26 = stablehlo.constant dense<1> : tensor<i64>
      %66 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %67 = stablehlo.subtract %59, %66 : tensor<9xi64>
      %c_27 = stablehlo.constant dense<2> : tensor<i64>
      %68 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %69 = func.call @floor_divide(%67, %68) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %70 = stablehlo.convert %69 : (tensor<9xi64>) -> tensor<9xf32>
      %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %71 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %cst_29 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %72 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %73 = stablehlo.add %iterArg, %70 : tensor<9xf32>
      %74 = stablehlo.negate %73 : tensor<9xf32>
      %75 = stablehlo.add %iterArg, %iterArg_16 : tensor<9xf32>
      %76 = stablehlo.add %75, %70 : tensor<9xf32>
      %77 = stablehlo.multiply %74, %76 : tensor<9xf32>
      %78 = stablehlo.multiply %77, %iterArg_17 : tensor<9xf32>
      %79 = stablehlo.multiply %72, %70 : tensor<9xf32>
      %80 = stablehlo.add %iterArg, %79 : tensor<9xf32>
      %81 = stablehlo.multiply %72, %70 : tensor<9xf32>
      %82 = stablehlo.add %iterArg, %81 : tensor<9xf32>
      %83 = stablehlo.add %82, %71 : tensor<9xf32>
      %84 = stablehlo.multiply %80, %83 : tensor<9xf32>
      %85 = stablehlo.divide %78, %84 : tensor<9xf32>
      %86 = stablehlo.subtract %iterArg_16, %70 : tensor<9xf32>
      %87 = stablehlo.multiply %70, %86 : tensor<9xf32>
      %88 = stablehlo.multiply %87, %iterArg_17 : tensor<9xf32>
      %89 = stablehlo.multiply %72, %70 : tensor<9xf32>
      %90 = stablehlo.add %iterArg, %89 : tensor<9xf32>
      %91 = stablehlo.subtract %90, %71 : tensor<9xf32>
      %92 = stablehlo.multiply %72, %70 : tensor<9xf32>
      %93 = stablehlo.add %iterArg, %92 : tensor<9xf32>
      %94 = stablehlo.multiply %91, %93 : tensor<9xf32>
      %95 = stablehlo.divide %88, %94 : tensor<9xf32>
      %cst_30 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %96 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %97 = stablehlo.select %63, %85, %95 : tensor<9xi1>, tensor<9xf32>
      %98 = stablehlo.select %65, %96, %97 : tensor<9xi1>, tensor<9xf32>
      %99 = stablehlo.broadcast_in_dim %iterArg_18, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %c_31 = stablehlo.constant dense<0> : tensor<i64>
      %100 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i64>) -> tensor<9xi64>
      %101 = stablehlo.compare  EQ, %99, %100,  SIGNED : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %102 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %cst_33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %103 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %104 = stablehlo.select %101, %102, %103 : tensor<9xi1>, tensor<9xf32>
      %105 = stablehlo.divide %98, %iterArg_20 : tensor<9xf32>
      %106 = stablehlo.add %104, %105 : tensor<9xf32>
      %cst_34 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
      %107 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %108 = stablehlo.abs %106 : tensor<9xf32>
      %109 = stablehlo.compare  LT, %108, %107,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %110 = stablehlo.select %109, %107, %106 : tensor<9xi1>, tensor<9xf32>
      %111 = stablehlo.multiply %98, %iterArg_21 : tensor<9xf32>
      %112 = stablehlo.add %104, %111 : tensor<9xf32>
      %113 = stablehlo.abs %112 : tensor<9xf32>
      %114 = stablehlo.compare  LT, %113, %107,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %115 = stablehlo.select %114, %107, %112 : tensor<9xi1>, tensor<9xf32>
      %116 = func.call @integer_pow(%115) : (tensor<9xf32>) -> tensor<9xf32>
      %117 = stablehlo.multiply %110, %116 : tensor<9xf32>
      %118 = stablehlo.multiply %iterArg_22, %117 : tensor<9xf32>
      %c_35 = stablehlo.constant dense<1> : tensor<i64>
      %119 = stablehlo.add %iterArg_18, %c_35 : tensor<i64>
      %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %120 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %121 = stablehlo.subtract %117, %120 : tensor<9xf32>
      %122 = stablehlo.abs %121 : tensor<9xf32>
      %cst_37 = stablehlo.constant dense<5.96046448E-8> : tensor<f32>
      %123 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<9xf32>
      %124 = stablehlo.compare  GE, %122, %123,  FLOAT : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %c_38 = stablehlo.constant dense<false> : tensor<i1>
      %125 = stablehlo.reduce(%124 init: %c_38) applies stablehlo.or across dimensions = [0] : (tensor<9xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %iterArg, %iterArg_16, %iterArg_17, %119, %125, %110, %116, %118 : tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>
    }
    %38 = chlo.lgamma %20 : tensor<9xf32> -> tensor<9xf32>
    %39 = chlo.lgamma %21 : tensor<9xf32> -> tensor<9xf32>
    %40 = stablehlo.add %38, %39 : tensor<9xf32>
    %41 = stablehlo.add %20, %21 : tensor<9xf32>
    %42 = chlo.lgamma %41 : tensor<9xf32> -> tensor<9xf32>
    %43 = stablehlo.subtract %40, %42 : tensor<9xf32>
    %44 = stablehlo.log %24 : tensor<9xf32>
    %45 = stablehlo.multiply %44, %20 : tensor<9xf32>
    %46 = stablehlo.negate %24 : tensor<9xf32>
    %47 = stablehlo.log_plus_one %46 : tensor<9xf32>
    %48 = stablehlo.multiply %47, %21 : tensor<9xf32>
    %49 = stablehlo.add %45, %48 : tensor<9xf32>
    %50 = stablehlo.subtract %49, %43 : tensor<9xf32>
    %51 = stablehlo.exponential %50 : tensor<9xf32>
    %52 = stablehlo.multiply %37#7, %51 : tensor<9xf32>
    %53 = stablehlo.divide %52, %20 : tensor<9xf32>
    %cst_14 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %55 = stablehlo.select %12, %54, %53 : tensor<9xi1>, tensor<9xf32>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<9xf32>
    %57 = stablehlo.subtract %56, %55 : tensor<9xf32>
    %58 = stablehlo.select %19, %55, %57 : tensor<9xi1>, tensor<9xf32>
    stablehlo.custom_call @check.expect_almost_eq(%58, %1) {has_side_effect = true} : (tensor<9xf32>, tensor<9xf32>) -> ()
    return %58 : tensor<9xf32>
  }
  func.func private @inputs() -> (tensor<9xf32> {mhlo.layout_mode = "default"}, tensor<9xf32> {mhlo.layout_mode = "default"}, tensor<9xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-1.600000e+00, -1.400000e+00, -1.000000e+00, 0.000000e+00, 1.000000e-01, 3.000000e-01, 1.000000e+00, 1.400000e+00, 1.600000e+00]> : tensor<9xf32>
    %cst_0 = stablehlo.constant dense<[-1.600000e+00, 1.400000e+00, 1.000000e+00, 0.000000e+00, 2.000000e-01, 1.000000e-01, 1.000000e+00, 1.400000e+00, -1.600000e+00]> : tensor<9xf32>
    %cst_1 = stablehlo.constant dense<[1.000000e+00, -1.000000e+00, 2.000000e+00, 1.000000e+00, 3.000000e-01, 3.000000e-01, -1.000000e+00, 2.400000e+00, 1.600000e+00]> : tensor<9xf32>
    return %cst, %cst_0, %cst_1 : tensor<9xf32>, tensor<9xf32>, tensor<9xf32>
  }
  func.func private @expected() -> (tensor<9xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0.622842908, 0.194610357, 0x7FC00000, 0x7FC00000, 0x7FC00000]> : tensor<9xf32>
    return %cst : tensor<9xf32>
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
