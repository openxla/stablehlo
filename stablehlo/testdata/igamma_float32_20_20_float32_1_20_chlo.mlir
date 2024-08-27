// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1] : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.compare  NE, %0#0, %0#0,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.compare  NE, %2, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.or %3, %4 : tensor<20x20xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %7 = stablehlo.compare  EQ, %2, %6,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %9 = stablehlo.compare  EQ, %2, %8,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %11 = stablehlo.compare  LT, %2, %10,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %13 = stablehlo.compare  LE, %0#0, %12,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = stablehlo.or %11, %13 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %16 = stablehlo.compare  GT, %2, %15,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %17 = stablehlo.compare  GT, %2, %0#0,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = stablehlo.and %16, %17 : tensor<20x20xi1>
    %19 = stablehlo.log %2 : tensor<20x20xf32>
    %20 = stablehlo.multiply %0#0, %19 : tensor<20x20xf32>
    %21 = stablehlo.subtract %20, %2 : tensor<20x20xf32>
    %22 = chlo.lgamma %0#0 : tensor<20x20xf32> -> tensor<20x20xf32>
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
    %35 = stablehlo.subtract %34, %0#0 : tensor<20x20xf32>
    %36 = stablehlo.add %2, %35 : tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %38 = stablehlo.add %36, %37 : tensor<20x20xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %41 = stablehlo.add %2, %40 : tensor<20x20xf32>
    %42 = stablehlo.multiply %38, %2 : tensor<20x20xf32>
    %43 = stablehlo.divide %41, %42 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %48 = stablehlo.negate %2 : tensor<20x20xf32>
    %49 = stablehlo.multiply %43, %48 : tensor<20x20xf32>
    %50 = stablehlo.subtract %47, %49 : tensor<20x20xf32>
    %51 = stablehlo.divide %50, %42 : tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52:15 = stablehlo.while(%iterArg = %33, %iterArg_22 = %43, %iterArg_23 = %44, %iterArg_24 = %35, %iterArg_25 = %38, %iterArg_26 = %cst_13, %iterArg_27 = %41, %iterArg_28 = %42, %iterArg_29 = %39, %iterArg_30 = %2, %iterArg_31 = %45, %iterArg_32 = %46, %iterArg_33 = %47, %iterArg_34 = %48, %iterArg_35 = %51) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %62:7 = stablehlo.while(%iterArg = %57, %iterArg_22 = %0#0, %iterArg_23 = %58, %iterArg_24 = %59, %iterArg_25 = %2, %iterArg_26 = %60, %iterArg_27 = %61) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %64 = stablehlo.divide %63, %0#0 : tensor<20x20xf32>
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
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}, tensor<1x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x1C75ED3D5200F33F3624F53FCBAAC03D813E78BF1F2616BF4F0F3EBFA01ED4BF82E58B403F93374039DE2CC0453949C02E8F88BE43B9C7C032F1CCBE39AF1840B8DE6A40F82FF0BF0D94E63F9A34B1BFE23273C00757DA3ED2A7DABF52A18240D935DC3F10787640427D53C074295B3F752A0C40BC2CFA3F5F0576C02226893FAC579340A6735AC0416D54BFD96406BD4925C83F5C6520C05CED9FBF9C5F10407F9608C0CF85793E0B64FA3F82A78DC027602540B8F2844007377340EADA59BF353F1BC06E7BB740925A973EEEB147C04C36FA4019345CBF258DE4403D2DF4BE2036B3BD62E8FDBFF65A88C05900C5BFD39D384023A8AA3FC224CABE4C8104C03270F23F2EBAB93DBFEE7BC03EF823BF3EA0EEBFF6A859BF53EAED3FD4E95F4035FC7E3F40F2FFBFA912314071006F3F6D8963C00D4104403CF3014148832940EDD999BFC28C3BC060D63BC0A23615405ED74C40AA41034162FD8B408D1CB93E60A487C07C1889C0F43CFCBE78F327C0941068402FEB1F40AE9150BF975C2E404B5A893D79AD69C08019983F9169CD3F66B35340B7B0E4BFBE4C874093E2C63E26B7DABF06F7ACBF3A710D3C348E523F6CC438409F7932C0C3D9813E7C5F81BF358E90C0503E3EBF20D041C0195CD63F53533640B1F60FBFDFBF4A405AB666400AF5043E729A58BF562356C03BE03ABFBCE2F13EDDB49BBFF52DA640FE9E443F385EFFBF45C91540884F593F58EC4ABF76FB67C0E466E43F57030FC0B4685ABE2AE4274035A58240E83AAEBFBFE0DA4063F5C33F11C2C6BFF4584640608DE13E5C7201C0D33940404CA782C03F4EC7BF874175C06C4DDBBF8E764BBFCBE04C3E73B017BF76B1A8BF992D3FC0153C3F403DAC133ECA11A83EEE2E54BF86AA813F0E9E5FC09A33E1BF874E78403E967B3F4068E34058690FC02443BA3FA1A8D33E28438E400C24F940980D0C3E6C5FC840B0DEBBBFB25550408513C2407E913E40AB473C3F088534C017C5F7BEF6002E409A007DC0D4CFB7C0CDD904C0811528415D9FA93F80CBC5BE34BF5D404DC841BFFE59E73FBC6F3FC0066B953E4E48E3BF4A1AEEBE3F0916406BB149C051B025C07D06D1BF32263F400B1E21C097EDA8C0CCEBB4407B92A9BD1B9E6140496492C0F892DD4070F7CBC0B6216D40E95250408F544D40E214AC3F78EB16C020DEBDBFEFEF283E0A9BAB4018763FBFF0637D3E367957C0C032D4BF6A4907406ECE59408A0C67C0EFBB973FB06B863F561291C0A9C489C0DE39FD4069184BC0632C86C0FFDC09BF91A61D4017DC924037C7CE3FEF42053F6232623ECC8F15C0192C4A403BBD743F252AAEBF25990B403840083F1990394064DE91BDFEA100C07827F03EAE89BA3F01CE0A40F7431EC088725A3F11C919C0F3BA833FD90593BED3CF4EC032E51E4071281540A2B1963E65D8CB3EA858CF40665472BF0C18A140C181F73FF3DA9DC08657B83E8E4A47C009E442C076F4183F0F98643E5F7A1D40B604303F80D21140231DDE40EE20FEC073FFA03F7C4D903FC02173400687DA3F7EDD854045A635C087124ABFE60D9A3FCCA2CEBFBD3AA1BFA3ED74C01D48F73F84A9CB40F11300409FABD43ED476203F38DA263F48625FC06B3504408D4E64C011DEFEBDA10083BFBDB709C031D08ABFBA2C023F2DF4B73DC81890C0DED45B40AAF834C0F9D727409EAAD33FC01E64C03E03FEBFFA51F2BFED90B7BCF499273CF9877CC05B26C4BE7E6EE5BF00D24D40881CFF3FA2FC14C0D465ACC0D77CB8BFBFD91C40E1AE30BE76B588C04E2C20C027D2A0BF2C28EA3F9D0C9EBFB11ACE3EB8EBE13FC00018C0F0050CC052064B40B8D7393E85B7B63F8930CC3F295B81BFFA193D3FDC0294BE18A04340ED954E3F176785BE89B02FC017BC05C091B46FC0068170BFF10E8EC06DCA113FC3F1A23F1587B4BFD4CDCE3F4E278E40F78019403E33DCBFB69FA24013F78FC0847BAD40E702D7BF993DC6C047EED5BE6BDD413FFC7C9F40C051FCBDA292F2BFE80A9D3FD9788E4099E537C0676642C07BEE9540C8C8B740636169C0285DC6C09C817C3F3EC559C0EA6CA7C0709320400A91ABBF2E85CEBF539111C0200080BF96637340EA0A86BF74915940E252573E234BABBE6E606D3E4BC9B83FD1B36FC06BC6B0BEDEBCA2BF55E51ABF578E00C05DE48FBFCBDF9FC01FB782C0D0AD9740585112403334BFBFF2EAD2405F425640F97A36C0A1F8364045F0E53FFD352E40657B223F62B507BE"> : tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<[[4.39382219, 0.189328432, 1.12942672, 8.07508659, -0.368767828, -0.476076275, -2.56633258, -1.27846646, -2.41863656, 2.14068699, -0.300600946, 5.38453531, -2.61912084, -0.976270675, -0.512060165, -0.295296222, -7.70805025, 6.2874341, -4.14961195, 1.55763531]]> : tensor<1x20xf32>
    return %cst, %cst_0 : tensor<20x20xf32>, tensor<1x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x19E97F3F4D7FA83C59ADAC3EBAFF7F3F0000C07F0000C07F0000C07F0000C07F0000C07F9280C93E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9075063F0000C07F75D4743F0000C07F0000C07F0000C07F0000C07F0000C07F64C2243F0000C07F0A9C7E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FD1C2C43E0000C07F9460353F8946A63E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F5DB1F63C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF8CA543F4C13A83D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FD8E4593F0000C07F0000C07F0000C07F0000C07F0000C07F6E277C3F0000C07F01E08F3E0000C07F0000C07F0000C07FE4BB7E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FA42B173F1CCD433F0000C07FF14DA43CACFD7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF4F2E63D9DE57F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FABD7083F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FC13F5D3F0000C07FCB0BC23AC250773F0000C07F1E78BF3D09FD7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEAF07F3F0000C07F0000C07F0000C07F0000C07F0000C07F1BF47F3F0000C07F8C0D493F0000C07F0000C07FEDF0073D90EC7F3F0000C07F0000C07F0000C07F0000C07F0000C07F76F6133B0000C07F00D1D03E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB681873E0000C07F0000C07F0000C07F4508743E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FC54E733F0000C07F0000C07FBB74AA3E0000C07FC63C5D3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F06694E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB4570E3E0000C07F306CEB3DB412283F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F611B003F0000C07FE7E77B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB8536A3F473E543F0000C07F0000C07FA3FC7F3F0000C07F0000C07F0000C07F0000C07F0000C07F1069603F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1FEBF63E0000C07F2C73163F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7212E53B0000C07F4FE97D3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7C0CAA3E8C63463F0000C07F0000C07F0000C07F0000C07F0000C07F87B31C3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FBBF25D3F26181C3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE293783F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F4B69723F0000C07F152E663FA5887F3F0000C07F0000C07F0000C07F0000C07F0000C07FD7113E3F0000C07F3D6A7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9266D03E4F2E113F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F3EA92F3D0000C07F0000C07FAEC4233C5976573F0000C07F0000C07F0000C07F0000C07F0000C07FDBCBF93E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1AFA7F3F0000C07F0ACD783FB17C783F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F65DCAF3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FCCA2763F0000C07F0000C07F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
