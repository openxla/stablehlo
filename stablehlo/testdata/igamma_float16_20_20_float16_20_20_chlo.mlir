// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.convert %0#0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %3 = stablehlo.convert %0#1 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %4 = stablehlo.compare  NE, %2, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.compare  NE, %3, %3,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %6 = stablehlo.or %4, %5 : tensor<20x20xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %8 = stablehlo.compare  EQ, %3, %7,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %10 = stablehlo.compare  EQ, %3, %9,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %12 = stablehlo.compare  LT, %3, %11,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %14 = stablehlo.compare  LE, %2, %13,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = stablehlo.or %12, %14 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %17 = stablehlo.compare  GT, %3, %16,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = stablehlo.compare  GT, %3, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %19 = stablehlo.and %17, %18 : tensor<20x20xi1>
    %20 = stablehlo.log %3 : tensor<20x20xf32>
    %21 = stablehlo.multiply %2, %20 : tensor<20x20xf32>
    %22 = stablehlo.subtract %21, %3 : tensor<20x20xf32>
    %23 = chlo.lgamma %2 : tensor<20x20xf32> -> tensor<20x20xf32>
    %24 = stablehlo.subtract %22, %23 : tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %25 = stablehlo.log %cst_4 : tensor<f32>
    %26 = stablehlo.negate %25 : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %28 = stablehlo.compare  LT, %24, %27,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %29 = stablehlo.exponential %24 : tensor<20x20xf32>
    %30 = stablehlo.or %8, %15 : tensor<20x20xi1>
    %31 = stablehlo.or %30, %28 : tensor<20x20xi1>
    %32 = stablehlo.or %31, %6 : tensor<20x20xi1>
    %33 = stablehlo.not %32 : tensor<20x20xi1>
    %34 = stablehlo.and %33, %19 : tensor<20x20xi1>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %36 = stablehlo.subtract %35, %2 : tensor<20x20xf32>
    %37 = stablehlo.add %3, %36 : tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %39 = stablehlo.add %37, %38 : tensor<20x20xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %42 = stablehlo.add %3, %41 : tensor<20x20xf32>
    %43 = stablehlo.multiply %39, %3 : tensor<20x20xf32>
    %44 = stablehlo.divide %42, %43 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %48 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %49 = stablehlo.negate %3 : tensor<20x20xf32>
    %50 = stablehlo.multiply %44, %49 : tensor<20x20xf32>
    %51 = stablehlo.subtract %48, %50 : tensor<20x20xf32>
    %52 = stablehlo.divide %51, %43 : tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53:15 = stablehlo.while(%iterArg = %34, %iterArg_22 = %44, %iterArg_23 = %45, %iterArg_24 = %36, %iterArg_25 = %39, %iterArg_26 = %cst_13, %iterArg_27 = %42, %iterArg_28 = %43, %iterArg_29 = %40, %iterArg_30 = %3, %iterArg_31 = %46, %iterArg_32 = %47, %iterArg_33 = %48, %iterArg_34 = %49, %iterArg_35 = %52) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %cst_36 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %75 = stablehlo.compare  LT, %iterArg_26, %cst_36,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %76 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %77 = stablehlo.and %75, %76 : tensor<i1>
      stablehlo.return %77 : tensor<i1>
    } do {
      %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %75 = stablehlo.add %iterArg_26, %cst_36 : tensor<f32>
      %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %76 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %77 = stablehlo.add %iterArg_24, %76 : tensor<20x20xf32>
      %cst_38 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %78 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %79 = stablehlo.add %iterArg_25, %78 : tensor<20x20xf32>
      %80 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %81 = stablehlo.multiply %77, %80 : tensor<20x20xf32>
      %82 = stablehlo.multiply %iterArg_27, %79 : tensor<20x20xf32>
      %83 = stablehlo.multiply %iterArg_29, %81 : tensor<20x20xf32>
      %84 = stablehlo.subtract %82, %83 : tensor<20x20xf32>
      %85 = stablehlo.multiply %iterArg_28, %79 : tensor<20x20xf32>
      %86 = stablehlo.multiply %iterArg_30, %81 : tensor<20x20xf32>
      %87 = stablehlo.subtract %85, %86 : tensor<20x20xf32>
      %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %88 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %89 = stablehlo.compare  NE, %87, %88,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %90 = stablehlo.divide %84, %87 : tensor<20x20xf32>
      %91 = stablehlo.subtract %iterArg_22, %90 : tensor<20x20xf32>
      %92 = stablehlo.divide %91, %90 : tensor<20x20xf32>
      %93 = stablehlo.abs %92 : tensor<20x20xf32>
      %cst_40 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %94 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %95 = stablehlo.select %89, %93, %94 : tensor<20x20xi1>, tensor<20x20xf32>
      %96 = stablehlo.select %89, %90, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %97 = stablehlo.multiply %iterArg_33, %79 : tensor<20x20xf32>
      %98 = stablehlo.subtract %97, %iterArg_27 : tensor<20x20xf32>
      %99 = stablehlo.multiply %iterArg_31, %81 : tensor<20x20xf32>
      %100 = stablehlo.subtract %98, %99 : tensor<20x20xf32>
      %101 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %102 = stablehlo.multiply %iterArg_29, %101 : tensor<20x20xf32>
      %103 = stablehlo.add %100, %102 : tensor<20x20xf32>
      %104 = stablehlo.multiply %iterArg_34, %79 : tensor<20x20xf32>
      %105 = stablehlo.subtract %104, %iterArg_28 : tensor<20x20xf32>
      %106 = stablehlo.multiply %iterArg_32, %81 : tensor<20x20xf32>
      %107 = stablehlo.subtract %105, %106 : tensor<20x20xf32>
      %108 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %109 = stablehlo.multiply %iterArg_30, %108 : tensor<20x20xf32>
      %110 = stablehlo.add %107, %109 : tensor<20x20xf32>
      %111 = stablehlo.multiply %96, %110 : tensor<20x20xf32>
      %112 = stablehlo.subtract %103, %111 : tensor<20x20xf32>
      %113 = stablehlo.divide %112, %87 : tensor<20x20xf32>
      %114 = stablehlo.select %89, %113, %iterArg_35 : tensor<20x20xi1>, tensor<20x20xf32>
      %115 = stablehlo.subtract %114, %iterArg_35 : tensor<20x20xf32>
      %116 = stablehlo.abs %115 : tensor<20x20xf32>
      %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %117 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %118 = stablehlo.select %89, %116, %117 : tensor<20x20xi1>, tensor<20x20xf32>
      %119 = stablehlo.abs %84 : tensor<20x20xf32>
      %cst_42 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %120 = func.call @integer_pow(%cst_42) : (tensor<f32>) -> tensor<f32>
      %121 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %122 = stablehlo.compare  GT, %119, %121,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %cst_43 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %123 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %124 = stablehlo.multiply %iterArg_27, %123 : tensor<20x20xf32>
      %125 = stablehlo.select %122, %124, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_44 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %127 = stablehlo.multiply %84, %126 : tensor<20x20xf32>
      %128 = stablehlo.select %122, %127, %84 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_45 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %129 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %130 = stablehlo.multiply %iterArg_28, %129 : tensor<20x20xf32>
      %131 = stablehlo.select %122, %130, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_46 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %132 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %133 = stablehlo.multiply %87, %132 : tensor<20x20xf32>
      %134 = stablehlo.select %122, %133, %87 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_47 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %135 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %136 = stablehlo.multiply %iterArg_33, %135 : tensor<20x20xf32>
      %137 = stablehlo.select %122, %136, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_48 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %138 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %139 = stablehlo.multiply %iterArg_34, %138 : tensor<20x20xf32>
      %140 = stablehlo.select %122, %139, %iterArg_34 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_49 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %141 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %142 = stablehlo.multiply %103, %141 : tensor<20x20xf32>
      %143 = stablehlo.select %122, %142, %103 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_50 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %144 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %145 = stablehlo.multiply %110, %144 : tensor<20x20xf32>
      %146 = stablehlo.select %122, %145, %110 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_51 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %147 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %148 = stablehlo.compare  GT, %95, %147,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %149 = stablehlo.and %iterArg, %148 : tensor<20x20xi1>
      %150 = stablehlo.select %iterArg, %96, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %151 = stablehlo.select %iterArg, %95, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %152 = stablehlo.select %iterArg, %77, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %153 = stablehlo.select %iterArg, %79, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %154 = stablehlo.select %iterArg, %128, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %155 = stablehlo.select %iterArg, %134, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %156 = stablehlo.select %iterArg, %125, %iterArg_29 : tensor<20x20xi1>, tensor<20x20xf32>
      %157 = stablehlo.select %iterArg, %131, %iterArg_30 : tensor<20x20xi1>, tensor<20x20xf32>
      %158 = stablehlo.select %iterArg, %137, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %159 = stablehlo.select %iterArg, %140, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %160 = stablehlo.select %iterArg, %143, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %161 = stablehlo.select %iterArg, %146, %iterArg_34 : tensor<20x20xi1>, tensor<20x20xf32>
      %162 = stablehlo.select %iterArg, %114, %iterArg_35 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %149, %150, %151, %152, %153, %75, %154, %155, %156, %157, %158, %159, %160, %161, %162 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %54 = stablehlo.multiply %53#1, %29 : tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %56 = stablehlo.subtract %55, %54 : tensor<20x20xf32>
    %57 = stablehlo.not %19 : tensor<20x20xi1>
    %58 = stablehlo.and %33, %57 : tensor<20x20xi1>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %63:7 = stablehlo.while(%iterArg = %58, %iterArg_22 = %2, %iterArg_23 = %59, %iterArg_24 = %60, %iterArg_25 = %3, %iterArg_26 = %61, %iterArg_27 = %62) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %c = stablehlo.constant dense<false> : tensor<i1>
      %75 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %75 : tensor<i1>
    } do {
      %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %75 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %76 = stablehlo.add %iterArg_22, %75 : tensor<20x20xf32>
      %77 = stablehlo.divide %iterArg_25, %76 : tensor<20x20xf32>
      %78 = stablehlo.multiply %iterArg_26, %77 : tensor<20x20xf32>
      %79 = stablehlo.multiply %iterArg_23, %iterArg_25 : tensor<20x20xf32>
      %80 = stablehlo.multiply %76, %76 : tensor<20x20xf32>
      %81 = stablehlo.divide %79, %80 : tensor<20x20xf32>
      %82 = stablehlo.subtract %78, %81 : tensor<20x20xf32>
      %83 = stablehlo.add %iterArg_27, %82 : tensor<20x20xf32>
      %84 = stablehlo.divide %iterArg_25, %76 : tensor<20x20xf32>
      %85 = stablehlo.multiply %iterArg_23, %84 : tensor<20x20xf32>
      %86 = stablehlo.add %iterArg_24, %85 : tensor<20x20xf32>
      %87 = stablehlo.divide %85, %86 : tensor<20x20xf32>
      %cst_29 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %88 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %89 = stablehlo.compare  GT, %87, %88,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %90 = stablehlo.and %iterArg, %89 : tensor<20x20xi1>
      %91 = stablehlo.select %iterArg, %76, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %92 = stablehlo.select %iterArg, %85, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %93 = stablehlo.select %iterArg, %86, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %94 = stablehlo.select %iterArg, %iterArg_25, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %95 = stablehlo.select %iterArg, %82, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %96 = stablehlo.select %iterArg, %83, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %90, %91, %92, %93, %94, %95, %96 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %64 = stablehlo.multiply %63#3, %29 : tensor<20x20xf32>
    %65 = stablehlo.divide %64, %2 : tensor<20x20xf32>
    %66 = stablehlo.select %19, %56, %65 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %68 = stablehlo.select %8, %67, %66 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %70 = stablehlo.select %10, %69, %68 : tensor<20x20xi1>, tensor<20x20xf32>
    %71 = stablehlo.or %15, %6 : tensor<20x20xi1>
    %cst_21 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %73 = stablehlo.select %71, %72, %70 : tensor<20x20xi1>, tensor<20x20xf32>
    %74 = stablehlo.convert %73 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    stablehlo.custom_call @check.expect_close(%74, %1) {has_side_effect = true} : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    return %74 : tensor<20x20xf16>
  }
  func.func private @inputs() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}, tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xE2C59AAF71C54CC0B0C2BDBE57C447378537CF417E3E2CC455424B42E8B891C250C1D444E33AF4BEC9C0E8BC7E3D593F0CC42FBCA0BEBC40CBC1FAB800B1113ED946263D7D3BE03F464634C34F3E1F43D6BE7B408539EBC7A642413D01403C2C51B6E4B6BB3D7743C83F0DC7C33F2540A93651C0E33E8D3CE4C32D384440AEC350BCADBF69456B3B0FC24ABBC13437BCF943FEBD07C0653D1235F4B203BF31B7E2C7883A9E3F81C5F63F1F3CA7C14DC3B2BD83C2CFC1E9B247C76840DDBDF844DD3ABF42414328BCD4B73046134326B7FDBEEFC0F7BF6C438636E7C1DA33A1C24638EE4420458544EEC08A438AC0833BFCC40C3E50BE454428C26C4475BE4A407542B03EB043EFBA5D44B13E67393CBF1FC3014243C5B3BEDDC4BA4547BF6848213A5B42D2C1E7C11543DF3EEABC70B46DC0D8318FB8AE3F483D15B48D3AAA382FB832C23042D9BF4C3D313416C664BCFDC1BEBC66276E4333BC433991C0E8C03F3CE4BD483FF199ED3CC34221C406C4404200C782AAA4B8D44479C3FD3E1639A8430AC81046B1468BBE3D40A63F4D4058C44640AC443A4407C0D643D03F623EC03DEDBBA7C010BF9BC606C201C458C65FC56CC0522D1FC40CC1FCB965BA88C70DC0383B773D964588C189C6C8BF4FC4B741E1C5AC458A3D0AC2BF413EC1D53A6DC8B1C319C05844B3C1C13E893FBAC269C3C02E2141393D2A4003C427C536C4F9C06DAD44C2583F48BCA7C03A44F13E6E31A2310ABDD5C417419BB87441B8C4933E7540B5C2893D7FBC46C647B9404446C063AA45C543C55FC2CE391340BD430136DFBB4DC69042EFBD3FB701347F3EDA3F59C03145C4BCE7BC75B9A946BE331644BF290542EC4060C196BF24C005BE8BBE0C436C406C40F143EF3F92AD57C0ECBF9BB0D8393ABE8C426F427AC1EA37ECB22A352D43C0C255BC393C2C4379C169421B42AFC36832E7BA3242EC409A438DBA00425744B63A4EB027468043D238D8B7BCBF3E3E7140664755B504C616C25E4458399A427EC356BECDC3BB415FB98A45E4446DBE473B8BBDF4BD55456C40E1C112C085C425BC94C2924110C063C75ABE3ABD03B877C02FC65DBAEF443BBDB344133D2A3B212CAD3C443DB0C1823E"> : tensor<20x20xf16>
    %cst_0 = stablehlo.constant dense<"0x3AC43F9F30C4E7B6EC32F53E24C00CC1CA3EF5320C28003B49BE892AA0BCA63DAB40F840324023BC61B3A3C062433A476D44A7B65D41663D333A93BF98C5464054B7D744DD4097A532403AC412400FC54CC61EBDF1C5DFC1844684B874B7D4C759283141593FB3C0FCC1F543FE4487396237F1C0C54448BF944356413F44353B01C060B6E0C00FBD9EC345446EBE973C37407247C7BE44C27CB8F244D8463D3B87B42AB89439FCC5D4B8A84268C105BCA83D5BBB8BC072C0D34534BF7DC381BEEF42FCB8CE3F1AC4CC3F52C4EBC1333E3342D83F03BFC5BCDA38E5C3374350BA5431E1450B40AF3DA83DD1BDF740284238C1643C764208357B446AC55EBE5640DEB6A7BC8DC00F398FC05C427BC1E2C32341E1B445C129C5D3C6A941D73D523F324420C280C258C2EA441EB76342ECBE6C4040C426C21A3ACE3A3FBEE3BC56C1F0B829C246C3E2C032B263C272C310C41C3FC7AF40440EAEA540B14109427EBECAC39740BEBD623E063E74C410BB65C109C5D1BD9DB93B39823F48C403C57D9903412AC3F04431424B3E79C42A44EB43BE352FBEFF31F43D43C539309146633918BE2A3C8FC0B3C141C1B42B8444E13A552C94B8014244C85AC1AAC059C79B478B446C3E8AC1C43217BE5F403C4037BACE43EDAF9CBD50BEAFA073BF3C3FB8B919C1313424C5FAB37B489FB965C8A1C41934D5A8643EBFC75B34B141E1B07F439C455D4434C1FF3C63C2D03FE331713C82BBEF42DF37C2418147CEBAD93C3EBA17C0ACC6D6BC47BE09B2B74042406D3884C24B41CCA820C2C53C23C0ABC56DC03F4134C575A72BB64DC1CABB9237363D3EC3BB41DF39F739AD3B1040F3C3DBBB40B9C23E3343DB45F22E47C2983F70B1A14477C506BE79B5CC3CADBA9D3A12C59FB57337D943C7B859B287BA094700B06E45BF430DC0C23A874496C0503E11B945C1F041E6BF6B4425BE973EBB3DD8368A405FBC50314BC45D36EF290EC2EEC4B23F5842EC3F664193B886C26640CD346F3FB1C1503CAD44D8C057B87440813C4A34AFC56ABA9D3AE0C022C5D1BCE042C1B110BD31C2AABA7F39D7C2AFB7E93C12BE0DC84244923D1540E8AFBE3E54C6FABDC9C485B1F63A4B4316381D438AA7"> : tensor<20x20xf16>
    return %cst, %cst_0 : tensor<20x20xf16>, tensor<20x20xf16>
  }
  func.func private @expected() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x007E007E007E007E007E007E007E007E853BC9171319007E007EC000007E007E007EF02F3C3B007E007E007E9A3BF73B007E007E007E9534007E007E007E193A007EE23B603B007E8023007ECC39007E007E007E007E007E853B007E007E007E007E007EC539007E007E007EB33B6B30C339007EBD3B007E007ED23B4B3B007E007E007E007E007E007E007E007E007E4531007E007E007E007E007E007E007E007E007EA431007E007EB23B007E007E007E007E007E007E007E007E007E007ED13B007E0132007E007E007E007E007E007E007E007E007E443A007EFB3B007E8936AD391B2A6D27007E007E007EAD3B007E6737007E0409007E007E007EC638007E007E007E007E007E0B3B007E007E007E007E007E007E007E882D007E8C09EE3B007E007E007E5D3A007E007E007E007E007E007E4D32ED36007E007E007E007E007E007E007E007E007E007E007E007E007E003C007E007EC93B007E007E007E007E007E007E9E39007E007E007E007E007E007E007E8F2A007E007E007E6634007ECF35952A007E007E6D3B1A3B007E007E5300432A007EFF00EC3B0234007E007E007E007E007E007E007E007E007E007EFD3B007E007E007E007E007E007E9F3A007E0700007E007E007E007E2F3A007E007E007E007E007E007E007E007E007E007E007E007E007E007E007E007E007E0435007E9526007E007E007E007E007E007E8336007E007E4C013636007EFC3B007E007EE93B007E5931007E007E007E007E007E007E007E007E0716007E007E007E007E007E007E007E007EE33B007E007E007E007E007EB53AFC37007E007E5A11007E007E007E007E007E007EF93B9039B43B007E007E007E007E007E007E007E007E7E28007E007E007E007E007EEB3B007E007E007E007E007E007EFB3B007E007E007E007E9A2F007E007E7538007EFF3B007EB832A0348016007E007E9500007E007E0000007E007E007E007EBD399839007E007E007E007E392D007E212C007E007E007EAF36007E1D00007E007ED738007E007E007E953A007E007E007E007E007E007E007E007E007E007E007E007E007E007EFA27007E007E007E007EDA3BB63B3C34007E007E"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
