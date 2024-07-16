// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1] : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %4 = stablehlo.compare  LE, %2, %3,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %6 = stablehlo.compare  LE, %0#0, %5,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %7 = stablehlo.or %4, %6 : tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %9 = stablehlo.compare  LT, %2, %8,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = stablehlo.compare  LT, %2, %0#0,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = stablehlo.or %9, %10 : tensor<20x20xi1>
    %12 = stablehlo.log %2 : tensor<20x20xf32>
    %13 = stablehlo.multiply %0#0, %12 : tensor<20x20xf32>
    %14 = stablehlo.subtract %13, %2 : tensor<20x20xf32>
    %15 = chlo.lgamma %0#0 : tensor<20x20xf32> -> tensor<20x20xf32>
    %16 = stablehlo.subtract %14, %15 : tensor<20x20xf32>
    %cst_2 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %17 = stablehlo.log %cst_2 : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %20 = stablehlo.compare  LT, %16, %19,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = stablehlo.or %7, %20 : tensor<20x20xi1>
    %22 = stablehlo.not %21 : tensor<20x20xi1>
    %23 = stablehlo.exponential %16 : tensor<20x20xf32>
    %24 = stablehlo.and %22, %11 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %29:7 = stablehlo.while(%iterArg = %24, %iterArg_20 = %0#0, %iterArg_21 = %25, %iterArg_22 = %26, %iterArg_23 = %2, %iterArg_24 = %27, %iterArg_25 = %28) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %c = stablehlo.constant dense<false> : tensor<i1>
      %63 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %63 : tensor<i1>
    } do {
      %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %63 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %64 = stablehlo.add %iterArg_20, %63 : tensor<20x20xf32>
      %65 = stablehlo.divide %iterArg_23, %64 : tensor<20x20xf32>
      %66 = stablehlo.multiply %iterArg_24, %65 : tensor<20x20xf32>
      %67 = stablehlo.multiply %iterArg_21, %iterArg_23 : tensor<20x20xf32>
      %68 = stablehlo.multiply %64, %64 : tensor<20x20xf32>
      %69 = stablehlo.divide %67, %68 : tensor<20x20xf32>
      %70 = stablehlo.subtract %66, %69 : tensor<20x20xf32>
      %71 = stablehlo.add %iterArg_25, %70 : tensor<20x20xf32>
      %72 = stablehlo.divide %iterArg_23, %64 : tensor<20x20xf32>
      %73 = stablehlo.multiply %iterArg_21, %72 : tensor<20x20xf32>
      %74 = stablehlo.add %iterArg_22, %73 : tensor<20x20xf32>
      %75 = stablehlo.divide %73, %74 : tensor<20x20xf32>
      %cst_27 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %76 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %77 = stablehlo.compare  GT, %75, %76,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %78 = stablehlo.and %iterArg, %77 : tensor<20x20xi1>
      %79 = stablehlo.select %iterArg, %64, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %80 = stablehlo.select %iterArg, %73, %iterArg_21 : tensor<20x20xi1>, tensor<20x20xf32>
      %81 = stablehlo.select %iterArg, %74, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %82 = stablehlo.select %iterArg, %iterArg_23, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %83 = stablehlo.select %iterArg, %70, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %84 = stablehlo.select %iterArg, %71, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %78, %79, %80, %81, %82, %83, %84 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %30 = stablehlo.multiply %29#3, %23 : tensor<20x20xf32>
    %31 = stablehlo.divide %30, %0#0 : tensor<20x20xf32>
    %32 = stablehlo.not %11 : tensor<20x20xi1>
    %33 = stablehlo.and %22, %32 : tensor<20x20xi1>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %35 = stablehlo.subtract %34, %0#0 : tensor<20x20xf32>
    %36 = stablehlo.add %2, %35 : tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %38 = stablehlo.add %36, %37 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %41 = stablehlo.add %2, %40 : tensor<20x20xf32>
    %42 = stablehlo.multiply %38, %2 : tensor<20x20xf32>
    %43 = stablehlo.divide %41, %42 : tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %48 = stablehlo.negate %2 : tensor<20x20xf32>
    %49 = stablehlo.multiply %43, %48 : tensor<20x20xf32>
    %50 = stablehlo.subtract %47, %49 : tensor<20x20xf32>
    %51 = stablehlo.divide %50, %42 : tensor<20x20xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52:15 = stablehlo.while(%iterArg = %33, %iterArg_20 = %43, %iterArg_21 = %44, %iterArg_22 = %35, %iterArg_23 = %38, %iterArg_24 = %cst_15, %iterArg_25 = %41, %iterArg_26 = %42, %iterArg_27 = %39, %iterArg_28 = %2, %iterArg_29 = %45, %iterArg_30 = %46, %iterArg_31 = %47, %iterArg_32 = %48, %iterArg_33 = %51) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %cst_34 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %63 = stablehlo.compare  LT, %iterArg_24, %cst_34,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %64 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %65 = stablehlo.and %63, %64 : tensor<i1>
      stablehlo.return %65 : tensor<i1>
    } do {
      %cst_34 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %63 = stablehlo.add %iterArg_24, %cst_34 : tensor<f32>
      %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %65 = stablehlo.add %iterArg_22, %64 : tensor<20x20xf32>
      %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %66 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %67 = stablehlo.add %iterArg_23, %66 : tensor<20x20xf32>
      %68 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %69 = stablehlo.multiply %65, %68 : tensor<20x20xf32>
      %70 = stablehlo.multiply %iterArg_25, %67 : tensor<20x20xf32>
      %71 = stablehlo.multiply %iterArg_27, %69 : tensor<20x20xf32>
      %72 = stablehlo.subtract %70, %71 : tensor<20x20xf32>
      %73 = stablehlo.multiply %iterArg_26, %67 : tensor<20x20xf32>
      %74 = stablehlo.multiply %iterArg_28, %69 : tensor<20x20xf32>
      %75 = stablehlo.subtract %73, %74 : tensor<20x20xf32>
      %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %76 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %77 = stablehlo.compare  NE, %75, %76,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %78 = stablehlo.divide %72, %75 : tensor<20x20xf32>
      %79 = stablehlo.subtract %iterArg_20, %78 : tensor<20x20xf32>
      %80 = stablehlo.divide %79, %78 : tensor<20x20xf32>
      %81 = stablehlo.abs %80 : tensor<20x20xf32>
      %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %82 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %83 = stablehlo.select %77, %81, %82 : tensor<20x20xi1>, tensor<20x20xf32>
      %84 = stablehlo.select %77, %78, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %85 = stablehlo.multiply %iterArg_31, %67 : tensor<20x20xf32>
      %86 = stablehlo.subtract %85, %iterArg_25 : tensor<20x20xf32>
      %87 = stablehlo.multiply %iterArg_29, %69 : tensor<20x20xf32>
      %88 = stablehlo.subtract %86, %87 : tensor<20x20xf32>
      %89 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %90 = stablehlo.multiply %iterArg_27, %89 : tensor<20x20xf32>
      %91 = stablehlo.add %88, %90 : tensor<20x20xf32>
      %92 = stablehlo.multiply %iterArg_32, %67 : tensor<20x20xf32>
      %93 = stablehlo.subtract %92, %iterArg_26 : tensor<20x20xf32>
      %94 = stablehlo.multiply %iterArg_30, %69 : tensor<20x20xf32>
      %95 = stablehlo.subtract %93, %94 : tensor<20x20xf32>
      %96 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %97 = stablehlo.multiply %iterArg_28, %96 : tensor<20x20xf32>
      %98 = stablehlo.add %95, %97 : tensor<20x20xf32>
      %99 = stablehlo.multiply %84, %98 : tensor<20x20xf32>
      %100 = stablehlo.subtract %91, %99 : tensor<20x20xf32>
      %101 = stablehlo.divide %100, %75 : tensor<20x20xf32>
      %102 = stablehlo.select %77, %101, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %103 = stablehlo.subtract %102, %iterArg_33 : tensor<20x20xf32>
      %104 = stablehlo.abs %103 : tensor<20x20xf32>
      %cst_39 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %105 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %106 = stablehlo.select %77, %104, %105 : tensor<20x20xi1>, tensor<20x20xf32>
      %107 = stablehlo.abs %72 : tensor<20x20xf32>
      %cst_40 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %108 = func.call @integer_pow(%cst_40) : (tensor<f32>) -> tensor<f32>
      %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %110 = stablehlo.compare  GT, %107, %109,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %cst_41 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %111 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %112 = stablehlo.multiply %iterArg_25, %111 : tensor<20x20xf32>
      %113 = stablehlo.select %110, %112, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_42 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %114 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %115 = stablehlo.multiply %72, %114 : tensor<20x20xf32>
      %116 = stablehlo.select %110, %115, %72 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_43 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %117 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %118 = stablehlo.multiply %iterArg_26, %117 : tensor<20x20xf32>
      %119 = stablehlo.select %110, %118, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_44 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %120 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %121 = stablehlo.multiply %75, %120 : tensor<20x20xf32>
      %122 = stablehlo.select %110, %121, %75 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_45 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %123 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %124 = stablehlo.multiply %iterArg_31, %123 : tensor<20x20xf32>
      %125 = stablehlo.select %110, %124, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_46 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %127 = stablehlo.multiply %iterArg_32, %126 : tensor<20x20xf32>
      %128 = stablehlo.select %110, %127, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_47 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %129 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %130 = stablehlo.multiply %91, %129 : tensor<20x20xf32>
      %131 = stablehlo.select %110, %130, %91 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_48 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %132 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %133 = stablehlo.multiply %98, %132 : tensor<20x20xf32>
      %134 = stablehlo.select %110, %133, %98 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_49 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %135 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %136 = stablehlo.compare  GT, %83, %135,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %137 = stablehlo.and %iterArg, %136 : tensor<20x20xi1>
      %138 = stablehlo.select %iterArg, %84, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %139 = stablehlo.select %iterArg, %83, %iterArg_21 : tensor<20x20xi1>, tensor<20x20xf32>
      %140 = stablehlo.select %iterArg, %65, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %141 = stablehlo.select %iterArg, %67, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %142 = stablehlo.select %iterArg, %116, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %143 = stablehlo.select %iterArg, %122, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %144 = stablehlo.select %iterArg, %113, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %145 = stablehlo.select %iterArg, %119, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %146 = stablehlo.select %iterArg, %125, %iterArg_29 : tensor<20x20xi1>, tensor<20x20xf32>
      %147 = stablehlo.select %iterArg, %128, %iterArg_30 : tensor<20x20xi1>, tensor<20x20xf32>
      %148 = stablehlo.select %iterArg, %131, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %149 = stablehlo.select %iterArg, %134, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %150 = stablehlo.select %iterArg, %102, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %137, %138, %139, %140, %141, %63, %142, %143, %144, %145, %146, %147, %148, %149, %150 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %53 = stablehlo.multiply %52#1, %23 : tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %55 = stablehlo.subtract %54, %31 : tensor<20x20xf32>
    %56 = stablehlo.select %11, %55, %53 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %58 = stablehlo.compare  EQ, %2, %57,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %60 = stablehlo.select %58, %59, %56 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %62 = stablehlo.select %7, %61, %60 : tensor<20x20xi1>, tensor<20x20xf32>
    stablehlo.custom_call @check.expect_almost_eq(%62, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %62 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}, tensor<1x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xBA1452BF375D99C0F4B5E0BF6759743FBA894A403359C83EEC8A24402D46E23FB9B10A402E67493F74CE2EBF17316240AA7E5240173733BF46459F3F7434B1BE15F9B8C09A52DB3F96E05640375A29C09B2C1AC045286F401A638CBFC5B4D1BE0A0B94C01F345E3F2CB0F63D7F889F405FF3B9BE4D7A1540C94BD9BF29DF083FA4FE003F1322E0BE9560FD3F409482C01105FCBF03C781BF47F15D407F27C640E2C7D4BF4953993D57A18EBF6FD3F93F4624084025C352C025113A3DDB3DC2BFCA5FA740C9B613C049A017C03557FB3F34E68840233A093FF2D414C0A46A534045F8DFBFB83461BF7B3FB9C0065898C0F95AACBFD3EDAF3F8D99E43FE884FBBDA538C73E5C3726BF0802B43F7343B1BFE9FB1140116099C056419FC0FC05A43FAFDFFABF0340FFBFE9E6A73F6E4718C0B6C978BF4A675F40A3628E3E90F3E23FD0A842BECB5754C0792446C06F175440E02C51402A8624BFB4B4F0BE963732C08A008F40F6F86EBF88856DC0634F98C04B2F99C06F33273F9ECFA13F6323403F2E0DEC3F8437CEBFAECCC43D5034B8BF169C4E3FDAD741BF82CB45C077792E40AD4CC7C08C0A8DBFFB8509409C04B73F7B31623E13B523C02A48A1C06E49C4C0805E93406FC91940BF7418C06C1300C0700009C0BBFAE63F66B0B4BF31E032BF261BD83F16490740D300D0408FFFC63F3A30A9409382A1BFDC2786401D7D913F7AF4B3C026591BC017618FBEC02A1B40BDD7274062CD2C408ADD4440DAC60F40A2320E40881189C07DD656BFA0E135BF8ABC3FC01E04563FB45B7FC0842B133F67E8933F111A1840E2AADB40D1F88DBED7B601C0C37AB1403F1EB2C096DAD8BF756C064086D2E3BF75E05E40FE930F403EBB59BFD7F9D9BE23C88E3F2BA0A5C05E4A8EC00A2D3F40FD7CFD3FDB1AC7BF1D30AF400B361EC0C51DAA3FBD53444031C44740B2DBC8BFD2E57F40858A24BF99EF54BF1385CEBECA22A43F5986404010DF8EC01A200D403DA539C0A59861C0480B41C06D6C154066BCB23F4F7F27BF308C4ABF37A11740985C24BEFCA74A3FE94E6B3FCE5FCDC03C6481405AC85740ADE838C0C192B9BF5CB2844012621BBF18DA6A40E73CAE3FA4A997BF35DC553F977D9FC05DDF693F66CAFEBF3FCCEEBF8E8ED4BF42D0A04079A0E73F37D2713F0B31483FCD39B73F8FD51B4009B240C0C34093C09DDDAD3FABDD79C03A5B9DC0ED443140B30B83BF92740340EEEA73BF445576BF7FA099C0DFF3644049BF63BFE15CBABC11DB0B4054417940C36050C0411953C0CBB015BE36B9B7BF3DEFAC40BC83B8BFE12DC1BEB14D863F0E51E24060A90D409AB8143F68CB864029195740AB4628BFF930BA3F74C8E0BFC8FD8E3FFD7AB340DD13ADBE3097603F708EA33FA149BC3FCA5AC4BC3202C4C0552C574088B32140FB8DBB3F51CEC1C09A3B1140E3D0FBBF96AA88BEFB7E614053221940D6F2C0C0E45347403752443F25458A407FB6D03F2110EB40DD93843F6F3F234011E3AD4054A2A53E938EBEBF3618D53F42328840E8AF83C002C11FC0EDD27BBE26E112C0012C3F3FB501853F9B4D17BFEB4FD140594C9ABF614F3340C2E4F6BF4BF323405CD728C025167440309B4740674ECC3FAEE725C06750E2BFB1FFF53F05814D3FE0E526C0ADAE92405FD2ADBF52DAA83E4A3B7CC0870519C01222D53FFA7C91BF854F14404E21DD3F23968A3FD17928BF616286BE63EF303EE621B0C08D5F9EBF329E00C068F23BBF3256333F4DFA3AC08F3F51C099F25E407687773FBEE4BC3F1F72554035B60C3F0BFA98C0CB2492BF79918CC0F5B9B43FAB009F407B40DABD3832F1BFE16A95C08A09B03FC48596C027AC87C051F0ACC06A66C7BF09EDE8BF1C41C140147C89405932E6BE5F9C763F90B07B3E6070E03FC33B9DBFBE5C8D40F44880C0DED114405761C0BEF022C33F788D3C4055E6013F658A15C0D07FDDBF3E8004C0329682406D1C693FF8BC1840FB417CC061F7C4BF73D41940E2313440E77653BF7D2A71C08C63434013398BC0DE77E9C062BB39C012CA383F3F93284052EB5B3F6DEE8AC04D7BFEBF18398A40A7D797401F3FF9BFDFD34040D88FB43FF2FDC6BF37F09240BA9B0E406E5B423F48D2723E543FAEC03BAA00C0EAC1FB3F97319DBFB121593FF0B0FA3FC25DE73E87FD22BFEAD804BFDA681F40FBB91BC0C6BCB7BF797AD53F6DF060404E58143F1826A93F74CE893F523B893F62F09C408B44A74043BE48C09FECDA3D"> : tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<[[3.33433104, -3.0612781, 1.59221566, 1.76121521, -3.06256843, 1.50310075, -3.63711357, -1.06411946, 2.40649104, 1.22511339, -0.406261683, -5.50833511, 0.86351943, 2.61897612, -0.425570458, -1.28150737, 0.478907317, -1.39718246, 3.93083906, -1.35018599]]> : tensor<1x20xf32>
    return %cst, %cst_0 : tensor<20x20xf32>, tensor<1x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x0000803F0000803F0000803F70C5233E0000803F3E64733D0000803F0000803F888DB33E7E565B3E0000803F0000803FB089763F0000803F0000803F0000803F0000803F0000803FAA13A23E0000803F0000803F0000803F0000803F0000803F0000803FF0D7393E0000803F0000803F0000803F9CE73E3F0000803F0000803F381C433E0000803F0000803F0000803F0000803F0000803FCB4FAD3E0000803F0000803F0000803F0000803F5A4BEB3E0000803F0000803F0000803F0000803F8D066C3F0000803F0000803F0000803F44237E3FBD7ECA3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F76E8EA3E0000803F0000803F0000803F0000803F0000803F2601C33E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F880D003B0000803F0000803F0000803F0000803FF8524D3F0000803F0000803F0000803F0000803F11AB583F0000803F0000803F0000803F0000803FAC860C3D0000803F0000803FA912643F0000803FC338FC390000803F8933BF3C0000803F0000803F00A02D3F0000803F0000803F0000803F0000803F8CD30F3C0000803F0000803F0000803F87F57E3FA5D0B93E0000803F0000803F0000803F0000803F0000803F0000803FC8DADB3D0000803F71477F3F287AAB3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FAD22673FF677E03E0000803F0000803FB272713F0000803F0000803F0000803F0000803F0000803F0000803F7BF6973D0000803F7EB62A3F0000803F0000803F0000803F09187F3F0000803F0000803FF2564F3F0000803F0000803F0000803F0000803F0000803F3C24CC3C0000803F0000803F0000803F3F73053F0000803F0000803F0000803F0000803F0000803F5AF4183F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F33C6A83E0000803F0000803F38342A3F0000803F0000803F1E259F3D0000803F0000803F0000803F0000803F0000803F0000803F0000803FFF327F3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F645E7B3F0000803F0000803FB0A3703D4AF0EA3E0000803F0000803F0000803F47EB033E0000803F0000803F02CD7A3F0000803F822FD33D0000803F0000803F0000803F89A6603F0000803F0000803F80EF1C3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F683A713F0000803FEC23FB3E0000803F0000803F0000803F0000803F08AA503E0000803F0000803F0000803F0000803F62F5373E0000803F0000803F0000803F74B3633FE813183E0000803F0000803F0000803F0000803FB305B33E0000803F0000803F0000803FC3EF0C3E6E486D3F0000803F5DDD7F3F0000803F0000803F4C946F3F89B7893D0000803F0000803FA5107E3F0000803F0000803F0000803F0000803F0000803FD7B7AE3C0000803FF1CE6B3F0000803FC57F3E3F0000803F0000803F0000803F0000803F0000803FF8B5543E0000803F0000803F0000803F88C3A83E0000803F0000803F0000803FAC975A3E0000803F0000803F0000803F0000803F0000803FDC18E13EE1DD463E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F42ED4D3F0000803F6FFFC33B0000803F0000803F0000803F1669AB3EAE0D773F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC792733F0000803F0000803F1AEE193F0000803F2DE6903D0000803FCEFA273F0000803F9DD41F3F0000803F0000803F42474C3F0000803F0000803F0000803F0000803F0000803F0000803F0FFB5D3F0000803F0000803F0000803F743F7B3F0000803F0000803F0000803F0000803F0000803F0000803F588ED53D0000803F5B19373E0000803F0000803F378F533F32D37C3F0000803F0000803F6CDD183F0000803F0000803F0000803F62EBFA3E0000803F0000803F0000803F57A0183E0000803F5D74233E1A5EEC3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F7061663E9E91FA3D0000803F0000803F4FF47F3F0000803F0000803F0000803F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
