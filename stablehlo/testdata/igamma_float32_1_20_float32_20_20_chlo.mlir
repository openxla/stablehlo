// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1] : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.compare  NE, %2, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.compare  NE, %0#1, %0#1,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.or %3, %4 : tensor<20x20xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %7 = stablehlo.compare  EQ, %0#1, %6,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %9 = stablehlo.compare  EQ, %0#1, %8,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %11 = stablehlo.compare  LT, %0#1, %10,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %13 = stablehlo.compare  LE, %2, %12,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = stablehlo.or %11, %13 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %16 = stablehlo.compare  GT, %0#1, %15,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %17 = stablehlo.compare  GT, %0#1, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = stablehlo.and %16, %17 : tensor<20x20xi1>
    %19 = stablehlo.log %0#1 : tensor<20x20xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<20x20xf32>
    %21 = stablehlo.subtract %20, %0#1 : tensor<20x20xf32>
    %22 = chlo.lgamma %2 : tensor<20x20xf32> -> tensor<20x20xf32>
    %23 = stablehlo.subtract %21, %22 : tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %24 = stablehlo.log %cst_4 : tensor<f32>
    %25 = stablehlo.negate %24 : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %27 = stablehlo.compare  LT, %23, %26,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %28 = stablehlo.exponential %23 : tensor<20x20xf32>
    %29 = stablehlo.or %7, %14 : tensor<20x20xi1>
    %30 = stablehlo.or %29, %27 : tensor<20x20xi1>
    %31 = stablehlo.or %30, %5 : tensor<20x20xi1>
    %32 = stablehlo.not %31 : tensor<20x20xi1>
    %33 = stablehlo.and %32, %18 : tensor<20x20xi1>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %35 = stablehlo.subtract %34, %2 : tensor<20x20xf32>
    %36 = stablehlo.add %0#1, %35 : tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %38 = stablehlo.add %36, %37 : tensor<20x20xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %41 = stablehlo.add %0#1, %40 : tensor<20x20xf32>
    %42 = stablehlo.multiply %38, %0#1 : tensor<20x20xf32>
    %43 = stablehlo.divide %41, %42 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %48 = stablehlo.negate %0#1 : tensor<20x20xf32>
    %49 = stablehlo.multiply %43, %48 : tensor<20x20xf32>
    %50 = stablehlo.subtract %47, %49 : tensor<20x20xf32>
    %51 = stablehlo.divide %50, %42 : tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52:15 = stablehlo.while(%iterArg = %33, %iterArg_22 = %43, %iterArg_23 = %44, %iterArg_24 = %35, %iterArg_25 = %38, %iterArg_26 = %cst_13, %iterArg_27 = %41, %iterArg_28 = %42, %iterArg_29 = %39, %iterArg_30 = %0#1, %iterArg_31 = %45, %iterArg_32 = %46, %iterArg_33 = %47, %iterArg_34 = %48, %iterArg_35 = %51) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %cst_36 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %73 = stablehlo.compare  LT, %iterArg_26, %cst_36,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %74 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %75 = stablehlo.and %73, %74 : tensor<i1>
      stablehlo.return %75 : tensor<i1>
    } do {
      %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %73 = stablehlo.add %iterArg_26, %cst_36 : tensor<f32>
      %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %74 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %75 = stablehlo.add %iterArg_24, %74 : tensor<20x20xf32>
      %cst_38 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %76 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %77 = stablehlo.add %iterArg_25, %76 : tensor<20x20xf32>
      %78 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %79 = stablehlo.multiply %75, %78 : tensor<20x20xf32>
      %80 = stablehlo.multiply %iterArg_27, %77 : tensor<20x20xf32>
      %81 = stablehlo.multiply %iterArg_29, %79 : tensor<20x20xf32>
      %82 = stablehlo.subtract %80, %81 : tensor<20x20xf32>
      %83 = stablehlo.multiply %iterArg_28, %77 : tensor<20x20xf32>
      %84 = stablehlo.multiply %iterArg_30, %79 : tensor<20x20xf32>
      %85 = stablehlo.subtract %83, %84 : tensor<20x20xf32>
      %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %86 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %87 = stablehlo.compare  NE, %85, %86,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %88 = stablehlo.divide %82, %85 : tensor<20x20xf32>
      %89 = stablehlo.subtract %iterArg_22, %88 : tensor<20x20xf32>
      %90 = stablehlo.divide %89, %88 : tensor<20x20xf32>
      %91 = stablehlo.abs %90 : tensor<20x20xf32>
      %cst_40 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %92 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %93 = stablehlo.select %87, %91, %92 : tensor<20x20xi1>, tensor<20x20xf32>
      %94 = stablehlo.select %87, %88, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %95 = stablehlo.multiply %iterArg_33, %77 : tensor<20x20xf32>
      %96 = stablehlo.subtract %95, %iterArg_27 : tensor<20x20xf32>
      %97 = stablehlo.multiply %iterArg_31, %79 : tensor<20x20xf32>
      %98 = stablehlo.subtract %96, %97 : tensor<20x20xf32>
      %99 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %100 = stablehlo.multiply %iterArg_29, %99 : tensor<20x20xf32>
      %101 = stablehlo.add %98, %100 : tensor<20x20xf32>
      %102 = stablehlo.multiply %iterArg_34, %77 : tensor<20x20xf32>
      %103 = stablehlo.subtract %102, %iterArg_28 : tensor<20x20xf32>
      %104 = stablehlo.multiply %iterArg_32, %79 : tensor<20x20xf32>
      %105 = stablehlo.subtract %103, %104 : tensor<20x20xf32>
      %106 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %107 = stablehlo.multiply %iterArg_30, %106 : tensor<20x20xf32>
      %108 = stablehlo.add %105, %107 : tensor<20x20xf32>
      %109 = stablehlo.multiply %94, %108 : tensor<20x20xf32>
      %110 = stablehlo.subtract %101, %109 : tensor<20x20xf32>
      %111 = stablehlo.divide %110, %85 : tensor<20x20xf32>
      %112 = stablehlo.select %87, %111, %iterArg_35 : tensor<20x20xi1>, tensor<20x20xf32>
      %113 = stablehlo.subtract %112, %iterArg_35 : tensor<20x20xf32>
      %114 = stablehlo.abs %113 : tensor<20x20xf32>
      %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %115 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %116 = stablehlo.select %87, %114, %115 : tensor<20x20xi1>, tensor<20x20xf32>
      %117 = stablehlo.abs %82 : tensor<20x20xf32>
      %cst_42 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %118 = func.call @integer_pow(%cst_42) : (tensor<f32>) -> tensor<f32>
      %119 = stablehlo.broadcast_in_dim %118, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %120 = stablehlo.compare  GT, %117, %119,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %cst_43 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %121 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %122 = stablehlo.multiply %iterArg_27, %121 : tensor<20x20xf32>
      %123 = stablehlo.select %120, %122, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_44 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %124 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %125 = stablehlo.multiply %82, %124 : tensor<20x20xf32>
      %126 = stablehlo.select %120, %125, %82 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_45 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %127 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %128 = stablehlo.multiply %iterArg_28, %127 : tensor<20x20xf32>
      %129 = stablehlo.select %120, %128, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_46 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %130 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %131 = stablehlo.multiply %85, %130 : tensor<20x20xf32>
      %132 = stablehlo.select %120, %131, %85 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_47 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %133 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %134 = stablehlo.multiply %iterArg_33, %133 : tensor<20x20xf32>
      %135 = stablehlo.select %120, %134, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_48 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %136 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %137 = stablehlo.multiply %iterArg_34, %136 : tensor<20x20xf32>
      %138 = stablehlo.select %120, %137, %iterArg_34 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_49 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %139 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %140 = stablehlo.multiply %101, %139 : tensor<20x20xf32>
      %141 = stablehlo.select %120, %140, %101 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_50 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %142 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %143 = stablehlo.multiply %108, %142 : tensor<20x20xf32>
      %144 = stablehlo.select %120, %143, %108 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_51 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %145 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %146 = stablehlo.compare  GT, %93, %145,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %147 = stablehlo.and %iterArg, %146 : tensor<20x20xi1>
      %148 = stablehlo.select %iterArg, %94, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %149 = stablehlo.select %iterArg, %93, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %150 = stablehlo.select %iterArg, %75, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %151 = stablehlo.select %iterArg, %77, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %152 = stablehlo.select %iterArg, %126, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %153 = stablehlo.select %iterArg, %132, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %154 = stablehlo.select %iterArg, %123, %iterArg_29 : tensor<20x20xi1>, tensor<20x20xf32>
      %155 = stablehlo.select %iterArg, %129, %iterArg_30 : tensor<20x20xi1>, tensor<20x20xf32>
      %156 = stablehlo.select %iterArg, %135, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %157 = stablehlo.select %iterArg, %138, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %158 = stablehlo.select %iterArg, %141, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %159 = stablehlo.select %iterArg, %144, %iterArg_34 : tensor<20x20xi1>, tensor<20x20xf32>
      %160 = stablehlo.select %iterArg, %112, %iterArg_35 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %147, %148, %149, %150, %151, %73, %152, %153, %154, %155, %156, %157, %158, %159, %160 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %53 = stablehlo.multiply %52#1, %28 : tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %55 = stablehlo.subtract %54, %53 : tensor<20x20xf32>
    %56 = stablehlo.not %18 : tensor<20x20xi1>
    %57 = stablehlo.and %32, %56 : tensor<20x20xi1>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %62:7 = stablehlo.while(%iterArg = %57, %iterArg_22 = %2, %iterArg_23 = %58, %iterArg_24 = %59, %iterArg_25 = %0#1, %iterArg_26 = %60, %iterArg_27 = %61) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %c = stablehlo.constant dense<false> : tensor<i1>
      %73 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %73 : tensor<i1>
    } do {
      %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %73 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %74 = stablehlo.add %iterArg_22, %73 : tensor<20x20xf32>
      %75 = stablehlo.divide %iterArg_25, %74 : tensor<20x20xf32>
      %76 = stablehlo.multiply %iterArg_26, %75 : tensor<20x20xf32>
      %77 = stablehlo.multiply %iterArg_23, %iterArg_25 : tensor<20x20xf32>
      %78 = stablehlo.multiply %74, %74 : tensor<20x20xf32>
      %79 = stablehlo.divide %77, %78 : tensor<20x20xf32>
      %80 = stablehlo.subtract %76, %79 : tensor<20x20xf32>
      %81 = stablehlo.add %iterArg_27, %80 : tensor<20x20xf32>
      %82 = stablehlo.divide %iterArg_25, %74 : tensor<20x20xf32>
      %83 = stablehlo.multiply %iterArg_23, %82 : tensor<20x20xf32>
      %84 = stablehlo.add %iterArg_24, %83 : tensor<20x20xf32>
      %85 = stablehlo.divide %83, %84 : tensor<20x20xf32>
      %cst_29 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %86 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %87 = stablehlo.compare  GT, %85, %86,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %88 = stablehlo.and %iterArg, %87 : tensor<20x20xi1>
      %89 = stablehlo.select %iterArg, %74, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %90 = stablehlo.select %iterArg, %83, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %91 = stablehlo.select %iterArg, %84, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %92 = stablehlo.select %iterArg, %iterArg_25, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %93 = stablehlo.select %iterArg, %80, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %94 = stablehlo.select %iterArg, %81, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %88, %89, %90, %91, %92, %93, %94 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %63 = stablehlo.multiply %62#3, %28 : tensor<20x20xf32>
    %64 = stablehlo.divide %63, %2 : tensor<20x20xf32>
    %65 = stablehlo.select %18, %55, %64 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %67 = stablehlo.select %7, %66, %65 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %69 = stablehlo.select %9, %68, %67 : tensor<20x20xi1>, tensor<20x20xf32>
    %70 = stablehlo.or %14, %5 : tensor<20x20xi1>
    %cst_21 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %72 = stablehlo.select %70, %71, %69 : tensor<20x20xi1>, tensor<20x20xf32>
    stablehlo.custom_call @check.expect_almost_eq(%72, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %72 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<1x20xf32> {mhlo.layout_mode = "default"}, tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.6881727, 2.78302264, 1.72012115, 6.96233272, -0.654121696, 5.66036797, 0.782327175, 1.67816162, -2.48140669, -4.60913324, 7.00846099, 2.21609473, -3.05883622, -0.913837909, -1.77270198, 2.91084933, 1.2070893, 1.45224094, -1.30566072, -6.07208776]]> : tensor<1x20xf32>
    %cst_0 = stablehlo.constant dense<"0x653003C045D2BABFF0E40F40D01FF03FE946B43E7120153E94E9593F680600C02DA4DD3EBED3223F76E731C0978F26407051134045088DC07D5848BF964C813F79B3F03F8D793F40E1ADD93CDE463BBE0A0C7A40EE8362C0B67E0D4014F2B7BE7D0DE7BFBF5BE5BFB5072A3F943405C0024C29C0648CE7C0F8E735408023144069EB4340013D12C09F380140C124FC3F467C53C0BE7DBA3F47E1E440A95001405BA6B4C08BC7A53EB19098C044C797C00D003E3F84E062C00C4D4840E6994C40A72308C051E32D4079BBDDBF4CA21F402D438C3F3B4C5F40848BC7BF80BC74C015A300C0305E1BC1A2C961BE83A2C33F8F96E4C0955F91408292D6BF1E2C16BE338FEFBF39AB8640CFD3E7BFC7905BC077C973C0271CCCBFF5B98CC0F80C72BE47476DC089D9F540724775BF845A4740873074BF4A41CCBF07C8573E4E7838C00B7BB3C0C2D3853E3AD0923FBB187FC0BEC1123F1CB52F3F3E3CA63FA4B706C0FE9BD43F0DCF493E1DAD2FBEAAD343BF40113F40EA15D63F6B880440E39DA5BDEC9CB6BED4589D4004DB883E58B19FBF16902640768E50C0E0342F40F9801B3F55271B4038340EBFEE2303402A877B3F2722B640A06BF0BF12C435407283C73EFEB74140B5FD93BF16FA10C0D21D15C08BDC92BE5EEF294056D23DC0C34924BFA5F633402F620E409D924ABFC1922E3F220FDFBF4A77F5BD992A12C0188CBCBFD97498C0879EE5BE050334404E3C22C00BE10740B26F983F6D5F14C08CE614C02562C73FCDF271BF72588AC075BCD6BF1BDEF5BE55C523404D44FA3D66B2014006DECA40E91392BE606D30BF6D2582C085BC88C0078616BFCB0DA9408C157F3FAFB80A40D29D7E40F5738CBFD8AE8E403C7F0CC098CCE7BEB887653FA07B7F40835A36403DA0713FABF3F7BE868D16C055238340D86685C05F370B4062DFBABF72FB4DC0668385BF5407D0BF30D01341B665404088E7284065444F409476E6BF712BDFC01ED45A3FC0E088C0C30E864027905E40D20DAABF69D29FBEB320E6BE3F773E4031F06D3F524C3A3E1FE2DDBFC698D63E5CA5973ECA12713EEEF72740F0CF38C0DDFA1E40EA198F3F379FF83F8F4A6CC06D5C48403DFBC53F52B80AC0914F1D3E61FAC63D6571A7BF8D2E824070D3523FC9B5BC406353D33D71BFF5BED5F6FFC0DA327AC03747EBC0891F474080345A3FFAD751C0AE3902C0851D42BE395DBFBF8C18203EF6A14B3E984335BE972919413D9682C052E82540A2CC09C062E1FD3F41D051C0611D5740BD1666C0105A03C1CF78B9BFC7665840E067AF3F7E43F43F355E13402593A43F2738B1C0AA9AEBBFB227C3BF0B7FB4BFB21080C0E136683F41CEB4409013C43F597C60C03E2B87BF3F9875BFC4F830BF155DCB405B2CA940328219C0F22492BEE43443C0CE87AEBFA2B6A140434BA5BF0F2F0CC00D271CC07FF71CC0B285373FA0061BC0BAA253C068878BC00412BEBE194B933FD590BD3F1D7BD54032E610C02283263F285D6CBF08B8BC3F27A45F4028946AC0285D09C1E258693FEE2238C00D188F4091AA284082D50C40101632BF31AECB3F4C04AE4066D6F1BF314F5F401EDCF93F81194A3FADEF6040809894C068310A4013E34FBE9036893FDBC2C23F4D78CC3F98F511406E3BFB3DE74DAA3E19F89A4079EF98BF281BDE3E99E30DC08A78C23FBCB4D1BFBDC17240DBC9A73F910019C0F97DD3BD5AD1D43FEB91994095485ABE01DB8B402F157DBF3451DFBE44296C408D4515BF3FDFA2C07D4887C0E12A783F39B73BC0B31FD6C01A3090BEE65DACBFEA4B31C0C93AE03F8BFA27BF36E498C09B5F173F6901CD3E1AD723408BF7F94066835740135EFA40C8968AC01FDCA5BF361D1941A98FBBBF8A6B803F811D9FC0A4D1B13F42DC09403B3F8E407AF750C03F3AFE3F6D6E6840ADD400408CBBC7C0237766C095D533C0A70173C09A6F8C4028E881C0F7B9F9BFDF5C063F06A295BF7E457B40B0A095BF229501C0F6FBA4C07F2C18C03C88E7C090D1733E40590B3F447969404A696CBFE4B45AC07A537C3FF16858BF95C180401F7FCA4024E289C08433733F240BA7C01D973F405C925DBF9BBB30C0360E7BBF8D13B9BFF911BD3F05C59E402F0866BF490F2DBF51AC14C0F4E8D4409555393F822C4340B7E01DC0D88D3A3FE67853BDC8725BBF3D795E40A148ECBF40F63C3DA5F62ABFBD9B8EC08017E03EC6788A3F83AA00C048B087C0D7721FC03957C3BF9FBD9BC06B180740"> : tensor<20x20xf32>
    return %cst, %cst_0 : tensor<1x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x0000C07F0000C07F30623B3FB06C5E3B0000C07F8E21353303EB2C3F0000C07F0000C07F0000C07F0000C07FDB2C2E3F0000C07F0000C07F0000C07FCC11BD3DDA5F4B3F1E0B653F0000C07F0000C07F0000C07F0000C07F475E393F0000C07F0000C07F0000C07FA6BB183F0000C07F0000C07F0000C07FC28DD43C563C1D3F0000C07F0000C07F0000C07FE210AC3E0000C07F35911C3F0000C07F0000C07F0000C07F98EAF33B0000C07F0000C07F0000C07F0000C07F3104793F318F613F0000C07F0000C07F0000C07F1618283F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F254C5C3F0000C07F0000C07F0000C07FABED983E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB9681F3F0000C07F0000C07F0000C07F0000C07F0000C07F4DA08C3B44A1D03E0000C07F0000C07F8D7C3539C9DA4D3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F5B497B3F0000C07F0000C07F0000C07F0000C07FEB5E513F2D9983360000C07F0000C07F31E9693F5595B63E0000C07F0000C07F36CBD33C01151E3D0000C07F0000C07F0000C07F0000C07F0000C07F6EC65B3F0000C07F0000C07F0000C07FBF5FE03E0000C07F11070A370000C07F0000C07F0000C07F0000C07F0000C07F0000C07FCD71CA3C0000C07F0000C07F0000C07F0000C07F0000C07F963C393F0000C07F0000C07F0000C07F0000C07F2493063F8DF8813C8C92A73B0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE2E58E3EEA1A553E0000C07F0000C07F0000C07FD6AA553F0000C07F0000C07F0000C07F0000C07F0000C07FB0F3C03D0000C07F0000C07F0000C07F0000C07F6FB76C3F0000C07F0000C07F0000C07F0000C07FDA9E7F3F0000C07F0000C07F0000C07F0000C07F0000C07FAF1AC43E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F85044E3AD6A8863E0000C07F0000C07F0000C07F6907D8317A5E2F3F0000C07F0000C07F0000C07FB82CA83E0000C07F9B29683F0000C07F0000C07F0000C07F406EA1390000C07FE0EDF73D0000C07F921D183F41B8323E0000C07F0000C07F0000C07F0000C07F6459463F0000C07F0000C07F0000C07F0000C07F0000C07F1443443D0000C07F0000C07F0000C07F0000C07F4D9F4B3F0000C07F0000C07F0000C07F33877A3F0000C07F0000C07F0000C07FF8D0653DC4F4AA3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F63C86F3F3E2F0B3F0000C07F0000C07F0000C07F0000C07FB9197E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F19F85B390000C07F4AA4343F0000C07F42095D3E0000C07F0000C07FCCE2833D0000C07F0000C07F0000C07F0000C07F6206563FCEF7653F8E5E4A3F0000C07F0000C07F0000C07F0000C07F74DC663F0083893B0000C07F39263B3E0000C07F6568393F0000C07F0000C07F2AE2813ACA3CD13E0000C07F0000C07F0000C07F26485F3F0000C07FD652383E0000C07F0000C07F0000C07F5042453F40D3EF3E0000C07F0000C07FC7033D3C26C87E3F0000C07F0000C07F0000C07F0000C07F55205A3F0000C07F0000C07F0000C07F7A96AC3D0000C07F0000C07F0000C07F0000C07F0000C07FC4559C3E0000C07F0000C07F0000C07F17C82E373C39733FD17F7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FA48A2D3F7834483F0000C07F0000C07F0000C07F77EE3E3F25DB2D3F0000C07F0000C07F0000C07F0000C07F7306753F0000C07F0000C07F96C6B4350000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1785AC380000C07F0858883ED6BF7F3F0000C07F0000C07F0000C07F47CF063D0000C07F0000C07F0000C07F0000C07F90EA4E3EA32C7D3F0000C07F0000C07F0000C07F0000C07FE321573DAACE5B3F0000C07F0000C07F0000C07F0000C07FE6EE673F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
