// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1] : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %4 = stablehlo.compare  LE, %0#1, %3,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %6 = stablehlo.compare  LE, %2, %5,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %7 = stablehlo.or %4, %6 : tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %9 = stablehlo.compare  LT, %0#1, %8,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = stablehlo.compare  LT, %0#1, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = stablehlo.or %9, %10 : tensor<20x20xi1>
    %12 = stablehlo.log %0#1 : tensor<20x20xf32>
    %13 = stablehlo.multiply %2, %12 : tensor<20x20xf32>
    %14 = stablehlo.subtract %13, %0#1 : tensor<20x20xf32>
    %15 = chlo.lgamma %2 : tensor<20x20xf32> -> tensor<20x20xf32>
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
    %29:7 = stablehlo.while(%iterArg = %24, %iterArg_20 = %2, %iterArg_21 = %25, %iterArg_22 = %26, %iterArg_23 = %0#1, %iterArg_24 = %27, %iterArg_25 = %28) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %31 = stablehlo.divide %30, %2 : tensor<20x20xf32>
    %32 = stablehlo.not %11 : tensor<20x20xi1>
    %33 = stablehlo.and %22, %32 : tensor<20x20xi1>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %35 = stablehlo.subtract %34, %2 : tensor<20x20xf32>
    %36 = stablehlo.add %0#1, %35 : tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %38 = stablehlo.add %36, %37 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %41 = stablehlo.add %0#1, %40 : tensor<20x20xf32>
    %42 = stablehlo.multiply %38, %0#1 : tensor<20x20xf32>
    %43 = stablehlo.divide %41, %42 : tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %48 = stablehlo.negate %0#1 : tensor<20x20xf32>
    %49 = stablehlo.multiply %43, %48 : tensor<20x20xf32>
    %50 = stablehlo.subtract %47, %49 : tensor<20x20xf32>
    %51 = stablehlo.divide %50, %42 : tensor<20x20xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52:15 = stablehlo.while(%iterArg = %33, %iterArg_20 = %43, %iterArg_21 = %44, %iterArg_22 = %35, %iterArg_23 = %38, %iterArg_24 = %cst_15, %iterArg_25 = %41, %iterArg_26 = %42, %iterArg_27 = %39, %iterArg_28 = %0#1, %iterArg_29 = %45, %iterArg_30 = %46, %iterArg_31 = %47, %iterArg_32 = %48, %iterArg_33 = %51) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %58 = stablehlo.compare  EQ, %0#1, %57,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %60 = stablehlo.select %58, %59, %56 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %62 = stablehlo.select %7, %61, %60 : tensor<20x20xi1>, tensor<20x20xf32>
    stablehlo.custom_call @check.expect_almost_eq(%62, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %62 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<1x20xf32> {mhlo.layout_mode = "default"}, tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.36930537, -1.06475282, -1.10180056, 0.632253647, -1.85677385, 0.1495893, 0.756189227, 3.17537475, -3.76900983, 1.28371584, -5.5223279, -1.53344905, 0.191576734, -1.65628433, 3.71646309, 1.38363183, -1.56151712, -0.758981466, -0.340194583, -2.37838793]]> : tensor<1x20xf32>
    %cst_0 = stablehlo.constant dense<"0xEEB2AF3F0A0DB2BFB690A2C05E059040D8D61EC014FC1440BC2BB84009402F404E3404C024451CC0FB92CA4055A08CC028ADD4400C69494046A25C40EA09BEC06CD43B3EE6C02540552B7CC0B4E41DC041B9B0C085AC2CBFE32AD5C0CDE85B4077EDDA3E04173DBF6C4817BE6C6D78C09428FC3E4C6A8D3F2A200241EB8D354014551B3FA425F0BFC48F86409D49D240F1D0D3C09F80BCBF81D3E53F9E14E9C054EB7E40BB91A3C03D8C2040ED5BF4BF5C4C65407A3BA440663371C03FE882C08F7217C0D8A3ED40DC3B70C0D7D2C840E291E4BEE38E2D40FF2F7C404CDA74C00FB78B40CE54C0C03A0369400492A5C0BF9E4440743C893F5DFB8B3F2D674B40260DE73E0BAD2D4045318040C12677409F113BBF4A761EC0D780C940299E9C3FA829BEBF30F666C01C0692BECDAE61BF6AF0853FA7500F3F51B1A03E834C0DC0DD653440A473FBBFB10C10400D71EFBEF4A5DBBFC89964C01B4D523F0A258A3FD38B363ECB325940064EBBBF9608F63F19F4B2402B19593E72DD5B3FCC0C85C07232483F5E9A1B3F8B8103C0BA7FC03FDEE09CC08B6D2B404CC58DBFE082264059CFFC3FE983F8BFDED4B34011758CBFA232044088861940F37E6FBF071B2C40CBDBACC0C151EB3FF775CB3EFEF6E3BE017736C0F278B9BF1AD101C0FDD52940FBF75A3E8EECB9BFBBA027C0F9C5CA3F1295ED3EAFA872C04E076C408B64A83F5313CABF761F80BF4F3E674012E8B23F5F1E92C0AF34F0BF6AF21BC077373640485F3FC0C4658EBF79F20AC167B007404DCB093F43D5CB3B58DCAEC051411B40EC8AC2BFF24755C0D0A70EBF65A5953FF2060540E30197BE6B808DBFB3E31340E535C93F2EE1E73E23870CBF3E1E863F6295733FCB03A34006692BC0CB2C7940FE7182C083C8023FD5A52740215B273F2B2011C1B50DC1BE5A67A2BF011D77BFA05CE8BF0ADDAF3FE31234C08FA08340DB6DDABE55B830404BE73FBEEB8481C0286D19C0AA838F3FE0BB4DBFB9358ABE923586400FBA7CC0212FA4BFE1DB1CC0C7F076C01F889D3B39E6B44078963D3FF0F96ABF8F7BAB4098B459C047AC04BE2161B03F7F9A0CC0C6B5073F4F83CEC0C5D48D40D098A6C00203944058AFCBBF14619C40FB7E1D40DFDFBB401442D0BF902986BF5B04A13ED82DB5BFD4089440CC4D7640E95B8640DC38773F0CB45B3F2D48F73F314C45BE691C15C032D35440021D6BC0A92744407D51BC3E258E493F7D9C594056A8D03EE2E08D401AFAB6C0BEFCB84082CD0540236CB1C0B4FF56405D67D93FA40E7140DB81853FCCFB51406F5801C0CD885640C8CC17403F7B16C0F21689C04F5CEAC0812E24C08873B83ECDBAB03F518A57C024DF41C09C5440C07C2ACB3F20C72740E2A85A3D380B42C067B6F5BFCC4AA2BF5766AB40824492C0692AA3C0DD11A4BF7AA92040698BD340741C90BEE316923F79A168C0F3148340DAFDC440C91BA03FC29D184038B4FB3FFB0C0D3F6466E63F0C688CBF8F2C19BF99AE27BF37309540980D82C079AC3CC0A73E58BE95E9CA3F4F9A8D3FA9AD4CC0F9C61040387F3A40E120A63F0818C1BFC77F31C0FB414FC069E116BE66A4C13E8086C43F5030C03F0D6744C06CFD2EBFA7C10340C1C3C9BF22B5863F70A3ACBF620D4140233D003FBC7EFDBD1435A240EC5179C058033C4033DB0F4025BE94C0740A7F40BF9C453E639DB7C0EEA0E1BF96B4244076FCFC3FB3030FC03D2EB5405C256D4018088C4001647ABFD667DF40B970A93FF3EB83BFE68316C064749840DDD2C03DB3A0224094708F3F69784ABEC8E4223EF9F79BBF47E109C01C3EAF403E811F40B1F7EA3ECBD7B240901F2C3C3EF3E4BF60C154401E2B32BFC9D185BFB07963BF36EE7EC0BBA780BFF1FC71BEF6B9E83F43729ABFD655CF3F216B39C063A31AC0510F2F40AB5288C05E0A09C09D2BB9BF388A76BEB3D1D93E986563BE4A8CA9C0E52021C06C1ACB3F28A3A4C00EE8D23EBA53F23F783F50407113E83FEB7C63C0E6ECF73FDBC19140DB7BB1BE6B5504C0AC31B23F438C93BF0F8C84C000A14E3F735C364065B0F23FF320C93E47B576401EEF2440A7E3B5C062779B40B910F13F1E33C93F938D86BEE4086CC0097304C0241FEABF23E409402F7FD14054FFB1BCD3A5F3BF016C44C05B2D61407BA88A40BF7F8ABFFBF12DC08791873F6A11B93E8C952D40A2FA4ABF4294C53F6C951940BB5747C07372873F5E0892BF21079DBF63F6B7BD7F6BB23F4E8850BE"> : tensor<20x20xf32>
    return %cst, %cst_0 : tensor<1x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xDD6AC53E0000803F0000803FD9D6893B0000803FFC3CC23BC891D63AF4D9063F0000803F0000803F0000803F0000803FCF4C5D380000803F4080F93E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F3407593C0000803F0000803F0000803F0000803F0000803F8689E43E0000803F0000803FD0D6C23D0000803F998FAD3E841A603B0000803F0000803F0000803F0000803FB49E1A3D0000803F0000803F0000803F0000803F6CE057390000803F0000803F0000803F1B879E3A0000803F0000803F0000803F0000803F662DC63E0000803F0000803F0000803F0000803F0000803F8417B23D0000803F0000803FBBBE8F3C0000803F54426E3B6F4A263C60D5953E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F2DEADF3D0000803F0000803F0000803F0000803F0000803F1075A53E1A786C3F0000803F2A8D673D0000803F0000803FE88533390000803FE0447B3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8796073D0000803F0000803FEBF9F63A0000803F0000803FE925113E0000803F0000803F0000803F0000803F889D7F3F0000803F0000803F0000803F0000803F0000803F507D693F0000803F0000803F63BFD63D0000803F0000803FCD66683C0210613F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF0C2DE3D0000803F0000803F0000803F0000803F62593D3F0000803F0000803F60E9243D0000803F0000803F0000803FE764683F0000803F0000803F0000803F0000803F05C0B23C0000803F0000803F5317023F0000803F0000803F0000803F0000803F0000803F0000803F0000803F1825A73E0000803F0000803F0000803F0000803F0000803FDC5DB53E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF05FFE3C0000803F0000803F0000803F0000803FBC9E043F5C92EE3A4472783F0000803F107A103C0000803F0000803F4A43EA3C0000803FADFC7E3F0000803F0000803F0000803F0000803F0000803F7A5B853C0000803F0000803F0000803F0000803F04D9073E0000803F75263D3E0000803F9B6FD93C0000803F0000803FA7715C3C0000803F0000803FEE9F903D0000803F0000803F0000803F0000803F7C1F843D0000803F0000803F0000803F0000803F6725043C0000803FD3D5C53E0000803FA053233D0000803F0000803F0000803F0000803F64FD3B3F0000803F0000803F0000803F0000803F0000803FB819C43E0000803F0000803F0000803F0000803F4BBA853B745A623F0000803F0000803F0000803F0000803F0000803F0000803F0000803F52CB333F32E7573B0000803F0000803F0000803F0000803F524AA13B0000803F0000803F385C8A3D0000803FC09C423C0000803F0000803F0000803FDC378C3C0000803F0000803F0000803F0000803FC3F2753F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FB4DCF33E0000803F58E8923C0000803F0000803F0000803F0000803F0000803F0000803F6BC5603B0000803F0000803F1104673C0000803F0000803F0000803F0000803F40591A3D0000803F0000803F0000803F0000803F82A3193C0000803FF7B7BF3D0000803F5AE4B73C0000803F0000803F4B0BFD3C0000803F0000803FD98D993C0000803F0000803F0000803F0000803F1D84703F0000803F0000803F035CC33A0000803FD06AC63DC605FF3AFFFF7F3F0000803F6D0B773D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FE6B1053F0000803F0000803F0000803F0000803F0000803F90260D3E0000803FAEA7073FBE238A3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8235493B1DB3C53D09B27E3F0000803FC59CF63D0000803F0000803F16086B3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FD676463C0000803F0000803F0000803F85556D3F0000803FBDCFD93D0000803F0000803FCD4DF13B0000803F132A773F0000803F0000803F0000803F0000803F0000803F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
