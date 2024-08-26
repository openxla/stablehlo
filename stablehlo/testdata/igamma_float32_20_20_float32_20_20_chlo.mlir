// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.compare  NE, %0#0, %0#0,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %3 = stablehlo.compare  NE, %0#1, %0#1,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.or %2, %3 : tensor<20x20xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %6 = stablehlo.compare  EQ, %0#1, %5,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %8 = stablehlo.compare  EQ, %0#1, %7,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %10 = stablehlo.compare  LT, %0#1, %9,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %12 = stablehlo.compare  LE, %0#0, %11,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %13 = stablehlo.or %10, %12 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %15 = stablehlo.compare  GT, %0#1, %14,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %16 = stablehlo.compare  GT, %0#1, %0#0,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %17 = stablehlo.and %15, %16 : tensor<20x20xi1>
    %18 = stablehlo.log %0#1 : tensor<20x20xf32>
    %19 = stablehlo.multiply %0#0, %18 : tensor<20x20xf32>
    %20 = stablehlo.subtract %19, %0#1 : tensor<20x20xf32>
    %21 = chlo.lgamma %0#0 : tensor<20x20xf32> -> tensor<20x20xf32>
    %22 = stablehlo.subtract %20, %21 : tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %23 = stablehlo.log %cst_4 : tensor<f32>
    %24 = stablehlo.negate %23 : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %26 = stablehlo.compare  LT, %22, %25,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %27 = stablehlo.exponential %22 : tensor<20x20xf32>
    %28 = stablehlo.or %6, %13 : tensor<20x20xi1>
    %29 = stablehlo.or %28, %26 : tensor<20x20xi1>
    %30 = stablehlo.or %29, %4 : tensor<20x20xi1>
    %31 = stablehlo.not %30 : tensor<20x20xi1>
    %32 = stablehlo.and %31, %17 : tensor<20x20xi1>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %34 = stablehlo.subtract %33, %0#0 : tensor<20x20xf32>
    %35 = stablehlo.add %0#1, %34 : tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %37 = stablehlo.add %35, %36 : tensor<20x20xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %40 = stablehlo.add %0#1, %39 : tensor<20x20xf32>
    %41 = stablehlo.multiply %37, %0#1 : tensor<20x20xf32>
    %42 = stablehlo.divide %40, %41 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %43 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %47 = stablehlo.negate %0#1 : tensor<20x20xf32>
    %48 = stablehlo.multiply %42, %47 : tensor<20x20xf32>
    %49 = stablehlo.subtract %46, %48 : tensor<20x20xf32>
    %50 = stablehlo.divide %49, %41 : tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51:15 = stablehlo.while(%iterArg = %32, %iterArg_22 = %42, %iterArg_23 = %43, %iterArg_24 = %34, %iterArg_25 = %37, %iterArg_26 = %cst_13, %iterArg_27 = %40, %iterArg_28 = %41, %iterArg_29 = %38, %iterArg_30 = %0#1, %iterArg_31 = %44, %iterArg_32 = %45, %iterArg_33 = %46, %iterArg_34 = %47, %iterArg_35 = %50) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %cst_36 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %72 = stablehlo.compare  LT, %iterArg_26, %cst_36,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %73 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %74 = stablehlo.and %72, %73 : tensor<i1>
      stablehlo.return %74 : tensor<i1>
    } do {
      %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %72 = stablehlo.add %iterArg_26, %cst_36 : tensor<f32>
      %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %73 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %74 = stablehlo.add %iterArg_24, %73 : tensor<20x20xf32>
      %cst_38 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %75 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %76 = stablehlo.add %iterArg_25, %75 : tensor<20x20xf32>
      %77 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %78 = stablehlo.multiply %74, %77 : tensor<20x20xf32>
      %79 = stablehlo.multiply %iterArg_27, %76 : tensor<20x20xf32>
      %80 = stablehlo.multiply %iterArg_29, %78 : tensor<20x20xf32>
      %81 = stablehlo.subtract %79, %80 : tensor<20x20xf32>
      %82 = stablehlo.multiply %iterArg_28, %76 : tensor<20x20xf32>
      %83 = stablehlo.multiply %iterArg_30, %78 : tensor<20x20xf32>
      %84 = stablehlo.subtract %82, %83 : tensor<20x20xf32>
      %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %85 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %86 = stablehlo.compare  NE, %84, %85,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %87 = stablehlo.divide %81, %84 : tensor<20x20xf32>
      %88 = stablehlo.subtract %iterArg_22, %87 : tensor<20x20xf32>
      %89 = stablehlo.divide %88, %87 : tensor<20x20xf32>
      %90 = stablehlo.abs %89 : tensor<20x20xf32>
      %cst_40 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %91 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %92 = stablehlo.select %86, %90, %91 : tensor<20x20xi1>, tensor<20x20xf32>
      %93 = stablehlo.select %86, %87, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %94 = stablehlo.multiply %iterArg_33, %76 : tensor<20x20xf32>
      %95 = stablehlo.subtract %94, %iterArg_27 : tensor<20x20xf32>
      %96 = stablehlo.multiply %iterArg_31, %78 : tensor<20x20xf32>
      %97 = stablehlo.subtract %95, %96 : tensor<20x20xf32>
      %98 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %99 = stablehlo.multiply %iterArg_29, %98 : tensor<20x20xf32>
      %100 = stablehlo.add %97, %99 : tensor<20x20xf32>
      %101 = stablehlo.multiply %iterArg_34, %76 : tensor<20x20xf32>
      %102 = stablehlo.subtract %101, %iterArg_28 : tensor<20x20xf32>
      %103 = stablehlo.multiply %iterArg_32, %78 : tensor<20x20xf32>
      %104 = stablehlo.subtract %102, %103 : tensor<20x20xf32>
      %105 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %106 = stablehlo.multiply %iterArg_30, %105 : tensor<20x20xf32>
      %107 = stablehlo.add %104, %106 : tensor<20x20xf32>
      %108 = stablehlo.multiply %93, %107 : tensor<20x20xf32>
      %109 = stablehlo.subtract %100, %108 : tensor<20x20xf32>
      %110 = stablehlo.divide %109, %84 : tensor<20x20xf32>
      %111 = stablehlo.select %86, %110, %iterArg_35 : tensor<20x20xi1>, tensor<20x20xf32>
      %112 = stablehlo.subtract %111, %iterArg_35 : tensor<20x20xf32>
      %113 = stablehlo.abs %112 : tensor<20x20xf32>
      %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %114 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %115 = stablehlo.select %86, %113, %114 : tensor<20x20xi1>, tensor<20x20xf32>
      %116 = stablehlo.abs %81 : tensor<20x20xf32>
      %cst_42 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %117 = func.call @integer_pow(%cst_42) : (tensor<f32>) -> tensor<f32>
      %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %119 = stablehlo.compare  GT, %116, %118,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %cst_43 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %120 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %121 = stablehlo.multiply %iterArg_27, %120 : tensor<20x20xf32>
      %122 = stablehlo.select %119, %121, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_44 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %123 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %124 = stablehlo.multiply %81, %123 : tensor<20x20xf32>
      %125 = stablehlo.select %119, %124, %81 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_45 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %127 = stablehlo.multiply %iterArg_28, %126 : tensor<20x20xf32>
      %128 = stablehlo.select %119, %127, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_46 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %129 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %130 = stablehlo.multiply %84, %129 : tensor<20x20xf32>
      %131 = stablehlo.select %119, %130, %84 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_47 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %132 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %133 = stablehlo.multiply %iterArg_33, %132 : tensor<20x20xf32>
      %134 = stablehlo.select %119, %133, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_48 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %135 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %136 = stablehlo.multiply %iterArg_34, %135 : tensor<20x20xf32>
      %137 = stablehlo.select %119, %136, %iterArg_34 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_49 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %138 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %139 = stablehlo.multiply %100, %138 : tensor<20x20xf32>
      %140 = stablehlo.select %119, %139, %100 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_50 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %141 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %142 = stablehlo.multiply %107, %141 : tensor<20x20xf32>
      %143 = stablehlo.select %119, %142, %107 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_51 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %144 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %145 = stablehlo.compare  GT, %92, %144,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %146 = stablehlo.and %iterArg, %145 : tensor<20x20xi1>
      %147 = stablehlo.select %iterArg, %93, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %148 = stablehlo.select %iterArg, %92, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %149 = stablehlo.select %iterArg, %74, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %150 = stablehlo.select %iterArg, %76, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %151 = stablehlo.select %iterArg, %125, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %152 = stablehlo.select %iterArg, %131, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %153 = stablehlo.select %iterArg, %122, %iterArg_29 : tensor<20x20xi1>, tensor<20x20xf32>
      %154 = stablehlo.select %iterArg, %128, %iterArg_30 : tensor<20x20xi1>, tensor<20x20xf32>
      %155 = stablehlo.select %iterArg, %134, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %156 = stablehlo.select %iterArg, %137, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %157 = stablehlo.select %iterArg, %140, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %158 = stablehlo.select %iterArg, %143, %iterArg_34 : tensor<20x20xi1>, tensor<20x20xf32>
      %159 = stablehlo.select %iterArg, %111, %iterArg_35 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %146, %147, %148, %149, %150, %72, %151, %152, %153, %154, %155, %156, %157, %158, %159 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %52 = stablehlo.multiply %51#1, %27 : tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %54 = stablehlo.subtract %53, %52 : tensor<20x20xf32>
    %55 = stablehlo.not %17 : tensor<20x20xi1>
    %56 = stablehlo.and %31, %55 : tensor<20x20xi1>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %61:7 = stablehlo.while(%iterArg = %56, %iterArg_22 = %0#0, %iterArg_23 = %57, %iterArg_24 = %58, %iterArg_25 = %0#1, %iterArg_26 = %59, %iterArg_27 = %60) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %c = stablehlo.constant dense<false> : tensor<i1>
      %72 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %72 : tensor<i1>
    } do {
      %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %72 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %73 = stablehlo.add %iterArg_22, %72 : tensor<20x20xf32>
      %74 = stablehlo.divide %iterArg_25, %73 : tensor<20x20xf32>
      %75 = stablehlo.multiply %iterArg_26, %74 : tensor<20x20xf32>
      %76 = stablehlo.multiply %iterArg_23, %iterArg_25 : tensor<20x20xf32>
      %77 = stablehlo.multiply %73, %73 : tensor<20x20xf32>
      %78 = stablehlo.divide %76, %77 : tensor<20x20xf32>
      %79 = stablehlo.subtract %75, %78 : tensor<20x20xf32>
      %80 = stablehlo.add %iterArg_27, %79 : tensor<20x20xf32>
      %81 = stablehlo.divide %iterArg_25, %73 : tensor<20x20xf32>
      %82 = stablehlo.multiply %iterArg_23, %81 : tensor<20x20xf32>
      %83 = stablehlo.add %iterArg_24, %82 : tensor<20x20xf32>
      %84 = stablehlo.divide %82, %83 : tensor<20x20xf32>
      %cst_29 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %85 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %86 = stablehlo.compare  GT, %84, %85,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %87 = stablehlo.and %iterArg, %86 : tensor<20x20xi1>
      %88 = stablehlo.select %iterArg, %73, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %89 = stablehlo.select %iterArg, %82, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %90 = stablehlo.select %iterArg, %83, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %91 = stablehlo.select %iterArg, %iterArg_25, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %92 = stablehlo.select %iterArg, %79, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %93 = stablehlo.select %iterArg, %80, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %87, %88, %89, %90, %91, %92, %93 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %62 = stablehlo.multiply %61#3, %27 : tensor<20x20xf32>
    %63 = stablehlo.divide %62, %0#0 : tensor<20x20xf32>
    %64 = stablehlo.select %17, %54, %63 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %66 = stablehlo.select %6, %65, %64 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %68 = stablehlo.select %8, %67, %66 : tensor<20x20xi1>, tensor<20x20xf32>
    %69 = stablehlo.or %13, %4 : tensor<20x20xi1>
    %cst_21 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %71 = stablehlo.select %69, %70, %68 : tensor<20x20xi1>, tensor<20x20xf32>
    stablehlo.custom_call @check.expect_almost_eq(%71, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %71 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}, tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x3000773F967E22C0C0FDF4C0AAC3673F6A6C57BEBCB2B7BFA2B9EFBF572C00404C383BBD11D2FFBE2CCF73C07F80CEC010F650C0D2A84CC090CD2FC0532D463FEBB48EC0BD42DFBFF1B9AE40B511593F661C34C02B2A86BF44DBB83FA497873F40D4B73FF3526B4009B6A43FDA973A40ADC846C09665B5BF4A2B4B3FFC555040EF5CE13FCCB30CC0D1FDD23F4CAEC23FBA94223F398F55C00362BFBFA26C4440176B993F5B506E408B1F8C3F6A7976BF801B1240122FAD40283234408548E3409C2B9FC0878A974025B82BC0807A65C0A16C31BF49D4DA3F2A93993F3E9AE03FEB52B84074062EC01B8D2B4049351ABEA614043FCB0D664047547BC0D2CB13BF4D9D604033F936BFFA0A64C0E25036C0AE452A406E0F33401A63D73FBBEA5B40CDF1E4BFDE976FC07C387C407DAEA3408D3CBD3F74AF133FD4E5FB3FFC9B9F405E5492BE07C3BCBFF04DB43F6B6DBB3F5BA782BF1AF643BF9F2BB53FCA73A83F0427B940FAD77D3FD224C6BF553F8BBE715C67C03199CE3F36D65A3FF71C83BF41E0974087E1F6BFABC89E40952731404AF73EC0C7A2A2C0D80B5EBF5AEA42C0A2F2933F74390E40C4A41AC19F0999407A62AFC0CC67CC3FD6D0123FABFD164032C24CC0E4BAEC4051766A3EC6FFE53F1D3C853F5B1889BFCC4FB5BEEAA3673FA9E9554023966C3F818D8FBF656F113E6075193F20071ABF74C3923FAEE7EBBECC9C3840C08A3CC014856F4054BA1BBF32EDE43F87894240050B4ABFF392C840698CB9405EF8D8BF586F47405AC60BC057978E40264782C0D104584029DDE5C055A23C4070EF1AC030C3FF3FA6F4A2BF0DA394400FBDD63F4B0295C074CFF03E54FEBDC0BB6F2EBFEA7C4840B79B6F3EFDB508C0B7206BC060C71B3E48A5BD3F8C3B6DBFF607E4BE1758EB405FBF64C0CF794440774A95C0C2AE19404D33E340356489400D67BE3E7C33314091FB63C03AEC62C0E4EA2AC031E077403920624099211CC0DDF355C0E0E3F7BFF4272BC00532B23E48635F402F3B51403C499E40674B243FE3FC854018D59940FC17D7BF670B1140B7A3A53F7863203E44A68CBE1F2B3BC0685488BFA8010F40CAC3E4BFE012FB3F98F180BF0E3D3BC0BF2E893FD658173E1618BC3FE46BA1BFC7CEAB4044557E405112D440FDD1F7BF064A4ABFFEE007404EBCBD3E84E484BFB80C0BBF9104283F27A21CC0B520FB3F097E8540BD7E0640C4E87440101606C0FCBB7AC0152C29C061AB4FBFEE45AFBE92FFAA3F6957D2C0AE797EBFC7FE53C0143DADBE6F3E1EC09203B0BFDDC19AC0E4F08C40B655ABBF8E132840F30E8B40044664403E8113BF933230BF96DB544011D23CC0985ADDC0964B0140B2932EC01E1C913FBE203FC0062CFF3FB85D6F40384373401331903FC5CDABC047DDA5C052A8C13FC503A4BFF580624085C08D40EC787B3FFC11AEBFCCB56C40C8CA31C0AE3F00BD04AC0CBF8B0E0B40704CCB408062034158253FC003573F3F08309CBFFDC5A53FA86F7CC0DD8871BE7C9B0E41AD63C040DC516F403C809A40A5B7CBBEB4B65740DA27B73FDCA1FFC068341A40D23E453F21AF7C3E5F864BBEFC293D3FC94C09405F87F0BF455991BFC293F63E8D100740A126F1BEDA0C4F40C4268ABF718546403B1B1B4056D883BFBEC2C53F5F20AABFDAD94B3FBADB9D3F4323913F647E50401DF0583D56C24040559EA23E4441BBBF45855CC0AF409CBEF91F0FBEDA63A1C0AFEA2D4073C19340A1AAD7405E85B1C0746A02BF54AF8EBED0E46FBF07E76B403FD758BF057FAAC0076281C0B2029D40EC932CBF426E1140C74DD7BF49D0D33E7C95CA40B38536C084C1D73F28D7C7BFA51D3DBEA7302B40D1CC36BF76497EBF3626D63FE5079B3DBF0F8B4007AD45C004D90D401E04F9BE8E5CEE3FA9698840294F99C0B3A867C02A400840A1699B3FF3B622C036D2CD4000C51D3F4EC90E402FFC543FF5D5A6C0751FD440C5B1D03F643FE240926E0CC01AE68640E9220E3B54B098C0AA468DC029C23A403B799BC0453EBAC0D8999FC0B307B5405363D43FE7E026C0BFCE67C001D0C5BFB6F75340AF5B9540AD7B9E3F1BBB033ED216B4C0B31EC43EDFC3064024DA6040DE690DC09524FBC059C34DBFD11AA74065529EBF210124401E5C93C06409B03FC4A591BF6E0384BE55648CC0D3B787C0B15B2EC02150BD3F0172AB3F552880C0B7A661C05546323DC63E0D40C1265CBF5851CBBFFDF1DDBF16E3C03E131194C0D4310740"> : tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<"0xB22DA8BEC7A123C0F9733C40CBAEF8C0FACB653E59EBB3BF497D08C075A02640642D78C0BB4802C003819B402ABC2940141C82BFA3C61A4037DD293E2412C9BF525685C0E503C04081BB5DBF07A90AC0C4432440240698C07FE4724057B3F13F869BA13E83E201406B849FC050AEE13F8BBBD0BFB6A9504010EB14BF09C16E3FF128F4BF116CC33FBE1D5B40501F2A40583A19C01C31C2C07E670C40774D60C0A1DFEDBFF69306C05B8C8DC040FA6B405926DCC07E3680BF368E88C038F733BFD30C173FA999D8BF86AEFCC0ECBEE5BF3DD321406C761DBF5361913F1C2D16400B6281BFDCAC3FBFB662564029A830402EF4533F5A440F41CE08D13FA84C0C40413434BFBDC140C06BB4463FC41D9DC0120BB94019B1E3BF0C3117C057433C4079C6E23F1A3281C0A68BF23E4B0C62C0614BD9C09F9B8E4011D2A8C042E92EC022F3DEBFB34302407C862B403C985540D28F41C00470A03F117350BFE8C2A1C0A417613F6F613EC00E7E303F91C6D6BF23A0CE3FBF5CBCBF46870C4063CF4DBFBAF725BF6A51B4400F2A893FED5A554082C7E8BF35AF3140415E5E4089A8F3BF2C1FCF3F06B6B240631FDCBF597483BFDE33AEBFBA0BED3FE055EDBF4BB4F93DE1FBF4405D3F08400506114030D08F40E6E4B9C04186094080F87CBE12A73940FBBBA440EB2328C0833E94C09B0E88BF3AF01EC0F34BE8BFE330B03F944E6BC0FB5E5D3FBA66DF4052A7B6C0A49609C0442D133FB8D7EABFF594D43FCD0676C0A11C953F02967940B5BC2FC08EDBB2BFE7E86140A059E9BFAC39DFBF34486AC01CB9E5BF0EBFE0BE7EF42A40F95D7CC0B9D9AA3FCD57F33EC30712414436614038C79DBEE1AB92BFBE8C81C0EC024340BE0F48405025EBBFB25CBF3FED33043FCDD5DBBFF04E1F4091E8693F050E76BD782E083F6C2984C0B5FBF13FDBE5843FA05A17C151683B3FE583C03EED7A2BC053A00940D8219E402192E73EB1614AC0A76011C022E0DEC015ACDC3FFBDC214084E4C340682046BFB7A669C03F502B3FCC007D403413A13F17A0C83F324037C0ED6F66BFEC125ABFFDE794C0323146C0D09719BEA5D2CD40533B8FC0758B4EC0AC2518408126C8BFDFB108C093AF40C0D3560BBF51B690C0BBB201C029109A3E976A9E3DCDECA140B8328AC00B93D7BE393D98C01311703F625895403C9196BF294B68C072BD863E3F77C83F8E6D8C3EF67F62C0BAB7483F480F773FE9688440AFD1B9C0B38441C00EAA833F7ABAB2BFBC1C2BC013A0FD3F3470643F3045273F2E59D9BFF273CABF01FDA3C0E9946CBFCC10CE3F034B403FA5F89C3FB298394077FD77BF8325F33F7193DE3ECDEBAEC02E9C07BFB01B633F560F8F3DF20E2B3F00611DBF36D79A3F99CA3A408D584F40962995409F9510C06D8783BFBD5F323F51EAB4BF52E6F1BF3EE95E40F34200C0853F0EBF21C472407BF2CF3F8D368D3FEED269C0777E32406480CCC0726513BFB58A98BFCC6960401825A2C0C33483BE3978923F86AB8F3E6FF284C08225913ED6570140241FE4BF52A354C0ED8397BFC6D3AABEB6451E3F4738FEBE9503F9BF35274A40CE2E19C0959B5D4019878740611FE7BF8AF659C0AC4978C0381E45403FAF38C0FB31603FCEAAA2401505E8BFE3B8B0BEC86B54C0F1AD8D40D604B63E60B54C3EE27E4C4050FF923F8686AABEAC832D3F4BE585C0E364A03F785301C0CB378B40D477B7402ED59BC090AB67BF1C5FB1C0ECB61CBF17C015BEB5E788BFC00C08403A72CCC0D8D6ABC0620C79C004EDC9C0A6A32A40BFB4814007D4F140CEE92FC0E2482BC0C15287C063F07840ACE467C0A6621BC0875B403E377CB9C0B8F9B440E019B53F2C2B81BFD3D7E3BECFABA73F902F6CC0E4DD853E8BBAA3C08D27AF3EB40EDC3EBFC324400D2965C01DA349C03E399F3F9E6FAFBFE1223CBE74DB933E4028AC3EEBFBD9BF9418F6BFF4542E40C10D743E10CBE8C020E57CC0F2E247C0093F45C03EB70EBF77F60540B47C9CBEF6B40E402FC1C240CA6E5EC0AA9CFA3FB35715C07CD9F73E2FD83DC06D052F3F50F2D1C0E1DB67C03E851840F828B240E21580C0189714BF5BA1F13F99E401C0D5CCA03FBC0046C0149501C02A15673F42E5B33FE86C6940C3DB823F245352400BCA103F17968FBF5A5597C0155DFEBF56C7ABBFA1E7CE40D4E635C07B2739BF00A71C4010BA4DC07A7FE93F5CDD243F29A4EEBFBB4569BFEECA453F7C51B9BEAE81463F100D9E3FD8F308C0"> : tensor<20x20xf32>
    return %cst, %cst_0 : tensor<20x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F338F3B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7E06733F2F8A553F72FEFF3DA3FF483E0000C07FCB518E3E0000C07F0000C07F0000C07F8DB6423D0000C07F0000C07FE0CE673F8B91583F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEE3A193F79363E3F0000C07F0000C07F3E1A383F0000C07F345D4B3F3E727C3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0034733F0000C07F0000C07F4D11EC3E0000C07F0000C07F44E9D73A0000C07F0000C07FE10D7F3F0000C07F0000C07F0000C07F0000C07F428C5E3F65D16B3F0000C07F0000C07F0000C07F0000C07FB0B8F4390000C07F0000C07F0000C07F0000C07F0000C07FA1236A3F0000C07F0000C07F0000C07FED5BAA3BD49F323F0000C07F0000C07F0000C07F0000C07F8826413FFA58773F0000C07F0000C07F0000C07F17F32C3F0000C07FFF25133B0000C07FAA97733B5C307D3F9114743F0000C07F0000C07F0000C07FEB5A743FA10E593F0000C07F0000C07F0000C07F0000C07F0000C07FC323323F0000C07FCD5D8A3D0000C07F0000C07F0000C07FA21E203E0000C07F0000C07F0000C07F984FF03A0000C07F0000C07F0000C07F614DC13E0000C07F0000C07F0000C07F0000C07F0000C07FFB193F3F0000C07F7707A13C05E3103E0000C07FB5287E3F0000C07F0000C07F0000C07FC4E47E3F0000C07F0000C07F6E467B3FFD5B593E0000C07F0000C07FB608BB370000C07FD349723C0000C07F695BE43EFED1B3380000C07FFDA8563F99B9373C0000C07F0000C07F0000C07FDDD7D53A0000C07F0000C07F0000C07F0000C07F0000C07F3EF07F3F0000C07F0000C07F61EF3C3A9B017E3FD36EF83C5C6AE63C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F3554323F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F040DB536683BD1355E3E983E0000C07F0000C07F0000C07F4B91613F0000C07F0000C07F0000C07F0000C07F28BDF33EEC90F4380000C07F00B23C3C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9543943DF7D2B63CC9EAD73E0000C07F0000C07F5EBDA03B0000C07F0000C07F41E65E3E0000C07FCF6FD93E0000C07FA886AF3E11C3C63EF3A7E53E84CD7C3F0000C07F0000C07F03B5933E0000C07F0000C07F78DCBE3E0000C07F0000C07FB88F163F0000C07F0000C07F0000C07FE96B3A3F0000C07F0000C07F0000C07F18B37B3F0000C07F0000C07F0000C07F0000C07F0000C07FF12C1235EEDB3C3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F57F57E3F0000C07FB0977B3F9AE1683F0000C07F0000C07F0000C07F0D854A3F0000C07F677C2A3D0000C07F0000C07F0000C07F0000C07F5A61773F0000C07F77008C3E53F86F3F641C213F0000C07F45837A3F0000C07FE8CF6F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F6D58613F0000C07F0000C07F0000C07F7E057F3F0000C07F0000C07FF5760E3D0000C07F0000C07FDB076F3E0000C07F0000C07FEDFEF73E0000C07FA6CA73380000C07FD071F43C0000C07F83A1423F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEE1CDD340000C07F0000C07FE703743F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FDDF97F3F0000C07F0000C07FF85A723F0000C07F0000C07F0000C07F9665F0370000C07F0000C07F0000C07F0000C07FA763B43EB30A353F0000C07F0000C07F0000C07F0000C07FE00CA73E0000C07F0000C07F0000C07F0000C07F2771883E0000C07F91903B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE6855A3F0000C07F0000C07F28397B3F0000C07F0000C07F0000C07F0000C07FB3AF583F0000C07F0000C07F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
