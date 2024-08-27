// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xbf16>, tensor<20x20xbf16>)
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %3 = stablehlo.convert %0#1 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
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
    %74 = stablehlo.convert %73 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%74, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %74 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}, tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x98C05D4094C05B3F3CC0C0BFF83F42C012C039402DBE5AC0D0BFDDBF05BFCA3F0E3EEB3D9A40D540BEC0CABF1EBF4D40793F8C40BEBFAABF13BF1D3F15C04A4089BF8B409CBF85BFB5C0D63ED43F5940A6BFE63EA040A3BF4D3FEEBF23C0A14076400340E03F53C07DC050C0C43EB6408FBD1040CF3F3AC0524027402AC00440813F773FE2BE23C023C0633E81BFD83F9BC06FBF67C0553F67400FBF964038C0A940E53F0140FFBFAEC02BC099408440034002C0D1BFD9408AC087C0E23F793F9740553E4ABF9E3F9CC0B440024026C02BC04CC0ABBE2540D7BFAA3F20C02A4042C0FA3EDC3E95404CBF58C062C0854056BF10BE28BE8B40BA3F663F9AC0993F17405F3F77BFE33E9A3FE5BE5540B6BE0AC07EC022BF1240E0C00E3DD9C0A84007BFAD3EB3C099BFFC3F3E40114004BE2EC058BF3840B03FD74086C0ACC0AE400EBF5940BFBF3B40B83F54C0B2BF80C098BC2AC0B3BFEB3F9F3F0B40A23E94BFBF401EC02F4066BF65404AC0C33FAEC059C066BF75C0F03F3C40B53F59406740D8C069C0843F18C039BFABBF3D40F93FA6C031C036C02C40DD3D3AC074408CC03FC0DC3FB4BF3C3DF43F2FBF353FC6BF793F9D40C83F7CBF84C06F3F96C08BC011C0A3C05240324058C090BF0EC0F8C0D2BF393E11C0DB3F274045C0963E25C06D3F7DC0F9BF8AC07940E53FB5C0C340B4BFF5BFBF40323EB83F1E3F1940AD40EFBF843D09C143C02BC019C0EABFD5BF3340A1BFF13FCABF7F4004C075BD0941143F95C07FC087C0A9C0853F524098BF04C07B407F3F3740B1C0A3BF06402C409F3F79402D4028C00AC0E3BFFA3D11404C3F86C024C07140913F20C02340833F7F4038C09840E43F453D593F4D3F34BF81BFBCBF9F40FC40A0C01D406A4034408940FFC0B2BEED3F38406DC09A407E408A40A4C0213FA53E5A40D5BFDB3F573E23C019C020C0103FE9BE41405140A63E754099406C40B2BEAE3E683FC03FBB3E8F400640A7C06AC087BF3F3F74BFD63F053F63BF19C02D3F6DBF69C0C1BFCEBE2DC085BE723D0EC0544037C0EFBF8EC020C0B3BF1BC004C06CC0D3C0AFBF2A4077405EC0D740AFC043C0F7BF0BC027C087BFD5BF27404EC0754042C0203FB63F"> : tensor<20x20xbf16>
    %cst_0 = stablehlo.constant dense<"0xA94084405640674081405B3F72C000BF63C01DC076C0AB3ECF3FD640214088401BC0F53F5D40BBBF3540AA405B3FB3406AC096BFCE3F28403FC0ED406D401BC0C6BFF4BF883F933FACBEBD3F923F33C088C0F3BF273FCD40ED3E923F04409D3FA83E9EC0633F53C004C009BFFC3F883F08BD87C00C3F6CBF404087407C4025C0D9BFCBC0A03FD43F3E403340EF3FC93EDABF334026BFAC3FB73DF8BE7140E44024BFD1BEA7BE504056C0AA405940443DEEBEA3C0B83F87BF58C028405D40B0C01640D03F983F914096C089BD124091BF3BC06FBF3940D1C01940DCBF90C0BABF61409E3FF0BFFFBF7C405640ADC036404E408EBF56C0CEBED23FE7BE62C02CC076C0484018C0C73F88C06740DA3F18C0004016C089BF14BE54C0DFC0AE40F5BF3140B0BF2BC04EC012BF5DBF904098C0BCC0AAC024C09F40CC3F8CC081BFD1404D407A3FC03EB63FE8408340AB3EBC40B040DCBF21C07ABF6B3F51BF92BF8EC0EABF8FC090401BBFA8C0A7C030C01A4075C0C4C02FC0794060C0B63F5DC0EDC0BF40ABBFE93F354000C0733FC5BE5B3D5ABFD1BF12406D3F13BF1EBEFABF504015BCADBEEABE063F603F12C0684025405FC0613F10BFBBBE8A3FE93F19C0D8BF2EBF7A3F05402E4043C029C00DC0F53FB23F8840F4BF034111C02F3F2340EB3F873F47BF66C08F3F2EC095BF8EBFEA3E94BF96404EC0123FFC3FDBBFA43FCABFB4BF2A408A40B8C094BFFC3F0D4085BF7EC0343F50C087C019C04A400BC0174050C05D40CBBF97C015C000416CC0404091C0CC3D7540C1BF544077C0DB3E87408FBFA43F2BC0624087C0B64060C0783F143F0640983EAEBED4BFECBF70C0F6BFDA3E1FC03ABF953F954022C09E40EB3F53C0E93FC6C0FBBFE34006BEFE3F97C0D9BF5A401141743E80BF823FB140BB403F4076C084C016C054BD5B3E943FE6BF27C057BF9F3E83C0FABE63403CC03E40BB3F69BE2AC0F73AEA4081C0394062BF1BC097C071BF053F97C08E3F663D3040633E3F408CC08ABF7B40FDBF773F963F34BF46BFB53F47BEB5409DBFBF40F2C0DFC0B13F81C01A408FBE73BEE03FDDBFD2BFD0BFD9BF42C0843F66C0D9BFC83F86C0D3BFD7405FBE95BF563D834068BF"> : tensor<20x20xbf16>
    return %cst, %cst_0 : tensor<20x20xbf16>, tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC07F333FC07F7B3FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F753FC07F7E3F983EC07FC07FC07FC07F663FC07FC07FC07FC07FC07F803FC07FC07FC07FC07FC07FC07FC07F6F3FDB3EC07FC07FC07F173AC07FF43EC07FC07F043C133AC07F963EC07FC07FC07F783FBB3AC07FC07F403EC07F033F5A3FC07FC07FC07FC07FC07FC07FC07F7F3FC07FD93DC07FC07FC07F4C3F3D37C07FC43EC07FC07FC07FC07FC07FC07FC07F943EFF33C07FC07FC07FC07FC07FC07F653FC07FEC3D7A3FC07F7B3FC07FC07F283FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F633FC07FC07FC07FC07FC07F943EC07FC07FC07FC07F2A3FC07FC07FC07FC07F773FC07F6F3FC07FC07F3A3EC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F6A3FC07FC07FC07FC07F7C3F053BC07FC07F383FC07F353DC07F3E3E7F3FC07FC07FC07FC07FC07FC07FC07FFE3EC07FC07FC07FC07FC07F5D3FC07FC07FC07FC07FC07FC07FC07FC07F6A3FC07F1D3FC07FC07FC07FC07F543FC07FC07FC07FC07FDF3AC07FC07FC07FD23DC07FC07FC07FC07FC07FC07FC07F7A3F793EC07F7D3FC07FC07F253BC07FC07FC07F5B3FC07FC07FC07FC07F903E113FC07FC07FC07FC07FC07F803FC07F803FC07FC07F7D3FC07F2F3FC07FC07FC07FC07FC07FC07FE036C07FC07FC07F693F403FC07F843EC07FC07F803FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FB93AC07FC07FC07FC07FC07F803FC07FC07FC07FAB367A3FC07FC07FC07F6E3D573FC07F433DC07FC07FC07FC07FC07F413E0C3FC07FC07FC07FC07FC07FC07FC07F853AC07FC07FC93E803FC07F7F3FC07FC07FC07FC07FC07FC07FC07F433EC07FC07FC07FC07F0D3DC07FC07F2E3F573F943EC07FC07FC07FC07FC07FD43EC07FC07FC07FC07FC07FC07F2F3FC07F7E3F993DC07FC07FC07F803FC07F613FC07FC07FC07FC07FC07FC07F463FC07F543FF53EC07FC07FC07FC07FC07FC07FC07FC07FC07F7E3FC07F653FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F7A3FC07FC07FC07F7E3FC07F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
