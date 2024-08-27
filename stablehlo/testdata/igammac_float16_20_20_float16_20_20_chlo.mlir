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
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %5 = stablehlo.compare  LE, %3, %4,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %7 = stablehlo.compare  LE, %2, %6,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %8 = stablehlo.or %5, %7 : tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %10 = stablehlo.compare  LT, %3, %9,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = stablehlo.compare  LT, %3, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %12 = stablehlo.or %10, %11 : tensor<20x20xi1>
    %13 = stablehlo.log %3 : tensor<20x20xf32>
    %14 = stablehlo.multiply %2, %13 : tensor<20x20xf32>
    %15 = stablehlo.subtract %14, %3 : tensor<20x20xf32>
    %16 = chlo.lgamma %2 : tensor<20x20xf32> -> tensor<20x20xf32>
    %17 = stablehlo.subtract %15, %16 : tensor<20x20xf32>
    %cst_2 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %18 = stablehlo.log %cst_2 : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %21 = stablehlo.compare  LT, %17, %20,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %22 = stablehlo.or %8, %21 : tensor<20x20xi1>
    %23 = stablehlo.not %22 : tensor<20x20xi1>
    %24 = stablehlo.exponential %17 : tensor<20x20xf32>
    %25 = stablehlo.and %23, %12 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %30:7 = stablehlo.while(%iterArg = %25, %iterArg_20 = %2, %iterArg_21 = %26, %iterArg_22 = %27, %iterArg_23 = %3, %iterArg_24 = %28, %iterArg_25 = %29) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %c = stablehlo.constant dense<false> : tensor<i1>
      %65 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %65 : tensor<i1>
    } do {
      %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %65 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %66 = stablehlo.add %iterArg_20, %65 : tensor<20x20xf32>
      %67 = stablehlo.divide %iterArg_23, %66 : tensor<20x20xf32>
      %68 = stablehlo.multiply %iterArg_24, %67 : tensor<20x20xf32>
      %69 = stablehlo.multiply %iterArg_21, %iterArg_23 : tensor<20x20xf32>
      %70 = stablehlo.multiply %66, %66 : tensor<20x20xf32>
      %71 = stablehlo.divide %69, %70 : tensor<20x20xf32>
      %72 = stablehlo.subtract %68, %71 : tensor<20x20xf32>
      %73 = stablehlo.add %iterArg_25, %72 : tensor<20x20xf32>
      %74 = stablehlo.divide %iterArg_23, %66 : tensor<20x20xf32>
      %75 = stablehlo.multiply %iterArg_21, %74 : tensor<20x20xf32>
      %76 = stablehlo.add %iterArg_22, %75 : tensor<20x20xf32>
      %77 = stablehlo.divide %75, %76 : tensor<20x20xf32>
      %cst_27 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %78 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %79 = stablehlo.compare  GT, %77, %78,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %80 = stablehlo.and %iterArg, %79 : tensor<20x20xi1>
      %81 = stablehlo.select %iterArg, %66, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %82 = stablehlo.select %iterArg, %75, %iterArg_21 : tensor<20x20xi1>, tensor<20x20xf32>
      %83 = stablehlo.select %iterArg, %76, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %84 = stablehlo.select %iterArg, %iterArg_23, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %85 = stablehlo.select %iterArg, %72, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %86 = stablehlo.select %iterArg, %73, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %80, %81, %82, %83, %84, %85, %86 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %31 = stablehlo.multiply %30#3, %24 : tensor<20x20xf32>
    %32 = stablehlo.divide %31, %2 : tensor<20x20xf32>
    %33 = stablehlo.not %12 : tensor<20x20xi1>
    %34 = stablehlo.and %23, %33 : tensor<20x20xi1>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %36 = stablehlo.subtract %35, %2 : tensor<20x20xf32>
    %37 = stablehlo.add %3, %36 : tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %39 = stablehlo.add %37, %38 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %42 = stablehlo.add %3, %41 : tensor<20x20xf32>
    %43 = stablehlo.multiply %39, %3 : tensor<20x20xf32>
    %44 = stablehlo.divide %42, %43 : tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %48 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %49 = stablehlo.negate %3 : tensor<20x20xf32>
    %50 = stablehlo.multiply %44, %49 : tensor<20x20xf32>
    %51 = stablehlo.subtract %48, %50 : tensor<20x20xf32>
    %52 = stablehlo.divide %51, %43 : tensor<20x20xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53:15 = stablehlo.while(%iterArg = %34, %iterArg_20 = %44, %iterArg_21 = %45, %iterArg_22 = %36, %iterArg_23 = %39, %iterArg_24 = %cst_15, %iterArg_25 = %42, %iterArg_26 = %43, %iterArg_27 = %40, %iterArg_28 = %3, %iterArg_29 = %46, %iterArg_30 = %47, %iterArg_31 = %48, %iterArg_32 = %49, %iterArg_33 = %52) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %cst_34 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %65 = stablehlo.compare  LT, %iterArg_24, %cst_34,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %66 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %67 = stablehlo.and %65, %66 : tensor<i1>
      stablehlo.return %67 : tensor<i1>
    } do {
      %cst_34 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %65 = stablehlo.add %iterArg_24, %cst_34 : tensor<f32>
      %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %66 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %67 = stablehlo.add %iterArg_22, %66 : tensor<20x20xf32>
      %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %68 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %69 = stablehlo.add %iterArg_23, %68 : tensor<20x20xf32>
      %70 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %71 = stablehlo.multiply %67, %70 : tensor<20x20xf32>
      %72 = stablehlo.multiply %iterArg_25, %69 : tensor<20x20xf32>
      %73 = stablehlo.multiply %iterArg_27, %71 : tensor<20x20xf32>
      %74 = stablehlo.subtract %72, %73 : tensor<20x20xf32>
      %75 = stablehlo.multiply %iterArg_26, %69 : tensor<20x20xf32>
      %76 = stablehlo.multiply %iterArg_28, %71 : tensor<20x20xf32>
      %77 = stablehlo.subtract %75, %76 : tensor<20x20xf32>
      %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %78 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %79 = stablehlo.compare  NE, %77, %78,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %80 = stablehlo.divide %74, %77 : tensor<20x20xf32>
      %81 = stablehlo.subtract %iterArg_20, %80 : tensor<20x20xf32>
      %82 = stablehlo.divide %81, %80 : tensor<20x20xf32>
      %83 = stablehlo.abs %82 : tensor<20x20xf32>
      %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %84 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %85 = stablehlo.select %79, %83, %84 : tensor<20x20xi1>, tensor<20x20xf32>
      %86 = stablehlo.select %79, %80, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %87 = stablehlo.multiply %iterArg_31, %69 : tensor<20x20xf32>
      %88 = stablehlo.subtract %87, %iterArg_25 : tensor<20x20xf32>
      %89 = stablehlo.multiply %iterArg_29, %71 : tensor<20x20xf32>
      %90 = stablehlo.subtract %88, %89 : tensor<20x20xf32>
      %91 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %92 = stablehlo.multiply %iterArg_27, %91 : tensor<20x20xf32>
      %93 = stablehlo.add %90, %92 : tensor<20x20xf32>
      %94 = stablehlo.multiply %iterArg_32, %69 : tensor<20x20xf32>
      %95 = stablehlo.subtract %94, %iterArg_26 : tensor<20x20xf32>
      %96 = stablehlo.multiply %iterArg_30, %71 : tensor<20x20xf32>
      %97 = stablehlo.subtract %95, %96 : tensor<20x20xf32>
      %98 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %99 = stablehlo.multiply %iterArg_28, %98 : tensor<20x20xf32>
      %100 = stablehlo.add %97, %99 : tensor<20x20xf32>
      %101 = stablehlo.multiply %86, %100 : tensor<20x20xf32>
      %102 = stablehlo.subtract %93, %101 : tensor<20x20xf32>
      %103 = stablehlo.divide %102, %77 : tensor<20x20xf32>
      %104 = stablehlo.select %79, %103, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %105 = stablehlo.subtract %104, %iterArg_33 : tensor<20x20xf32>
      %106 = stablehlo.abs %105 : tensor<20x20xf32>
      %cst_39 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %107 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %108 = stablehlo.select %79, %106, %107 : tensor<20x20xi1>, tensor<20x20xf32>
      %109 = stablehlo.abs %74 : tensor<20x20xf32>
      %cst_40 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %110 = func.call @integer_pow(%cst_40) : (tensor<f32>) -> tensor<f32>
      %111 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %112 = stablehlo.compare  GT, %109, %111,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %cst_41 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %113 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %114 = stablehlo.multiply %iterArg_25, %113 : tensor<20x20xf32>
      %115 = stablehlo.select %112, %114, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_42 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %116 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %117 = stablehlo.multiply %74, %116 : tensor<20x20xf32>
      %118 = stablehlo.select %112, %117, %74 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_43 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %119 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %120 = stablehlo.multiply %iterArg_26, %119 : tensor<20x20xf32>
      %121 = stablehlo.select %112, %120, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_44 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %122 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %123 = stablehlo.multiply %77, %122 : tensor<20x20xf32>
      %124 = stablehlo.select %112, %123, %77 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_45 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %125 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %126 = stablehlo.multiply %iterArg_31, %125 : tensor<20x20xf32>
      %127 = stablehlo.select %112, %126, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_46 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %128 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %129 = stablehlo.multiply %iterArg_32, %128 : tensor<20x20xf32>
      %130 = stablehlo.select %112, %129, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_47 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %131 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %132 = stablehlo.multiply %93, %131 : tensor<20x20xf32>
      %133 = stablehlo.select %112, %132, %93 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_48 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %134 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %135 = stablehlo.multiply %100, %134 : tensor<20x20xf32>
      %136 = stablehlo.select %112, %135, %100 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_49 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %137 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %138 = stablehlo.compare  GT, %85, %137,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %139 = stablehlo.and %iterArg, %138 : tensor<20x20xi1>
      %140 = stablehlo.select %iterArg, %86, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %141 = stablehlo.select %iterArg, %85, %iterArg_21 : tensor<20x20xi1>, tensor<20x20xf32>
      %142 = stablehlo.select %iterArg, %67, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %143 = stablehlo.select %iterArg, %69, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %144 = stablehlo.select %iterArg, %118, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %145 = stablehlo.select %iterArg, %124, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %146 = stablehlo.select %iterArg, %115, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %147 = stablehlo.select %iterArg, %121, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %148 = stablehlo.select %iterArg, %127, %iterArg_29 : tensor<20x20xi1>, tensor<20x20xf32>
      %149 = stablehlo.select %iterArg, %130, %iterArg_30 : tensor<20x20xi1>, tensor<20x20xf32>
      %150 = stablehlo.select %iterArg, %133, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %151 = stablehlo.select %iterArg, %136, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %152 = stablehlo.select %iterArg, %104, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %139, %140, %141, %142, %143, %65, %144, %145, %146, %147, %148, %149, %150, %151, %152 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %54 = stablehlo.multiply %53#1, %24 : tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %56 = stablehlo.subtract %55, %32 : tensor<20x20xf32>
    %57 = stablehlo.select %12, %56, %54 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %59 = stablehlo.compare  EQ, %3, %58,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %61 = stablehlo.select %59, %60, %57 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %63 = stablehlo.select %8, %62, %61 : tensor<20x20xi1>, tensor<20x20xf32>
    %64 = stablehlo.convert %63 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    stablehlo.custom_call @check.expect_close(%64, %1) {has_side_effect = true} : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    return %64 : tensor<20x20xf16>
  }
  func.func private @inputs() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}, tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xB2C482ACAE4199BF7BB54142193E7E3070C6684260C124C4FE4399447EBC0F37FA380F3E9CBE0DBA2244DB44913973C16D3B503330BDA736F033AAC297BC9343F3BDEFBA4741EDC05140E0BC4CC3AFC1FB46B5C6B12E0243443C59B964BB0DBC2D41CC417E4018BC1AB99039153EC64110C055C21CC430BBE03C43BCC4BDACC0A0BE39443FBCEA4184BE343D4B353BBD5DC11A40503C65C39A458EBD5CC5C540A33EC139F2B9D340BEBE37BF79420F3A763F69C33D42EB3723B58831A5457043653DA6C11E43E1C328C6A4C400BEEE3A48415F430E44F741AB3D34C2AE40EB3ED74456BECF3CA7C021427540CCA6E63E4BC6DAC07DC4D63E944480C0043895C2DF233DC0D1C2C43FBABCD044B4C08B3CE9C79B468CC452C5A445CE4233371745703769B4544062C1B3475C39C7C1B445A33C05402B40C13FE9B6BA3C58C3B8462A433BC27CB970BD9A4422397C43E5C06645DAC18D445AC835B91BC6F33DEA33FB444AC219BE70BCABB46A3BA93D8B31F3BEF0C24C37C7BD74C5C3C067BEBB3D26BC07C0B4ACB2C2F94305C592C1D5386737C6BEDE436A3E914298C0E6B5FC3D94C00DC475C322C067BEC7C512B6C0C800C48946394462C08344C43ED4C0F4C0F33DCEB6F03A1CBD18409131A2456F3B74AC2A3A33B7D24543B89E3DE2C342C1AABD76B99B3563BC6743BF3CF8BFA730C93D864604BE9AC0F23BD7C321B6A4420BBEE3C65045A6C514410D4541328AC00F3C56AEC5C0C642FF4494B92AC0D3C4223993415F3E5CC1D8383845934380C2A640CB3FD1BF9C40AD3DCF4306315DB6C8404FBF654770443EC09DB8023FA93EBE441B385EC4EEC620BD1AC35CB853C3C642EA4353C59C453DBF6DBA3C3630BF15C4F5453F4288C433C3103DA13DF239D63E68C060C104B84941B3ACC940A2BEF641CD43CEB90E349A4169C37CC25D409E4538C408C41635EA3529C78DBD0B430E3D67C4873D43395938A0C17DB80CC1F9C4F4BFDEBE8EC3AC40184529C419BC734220B533B71D3585B23D3BE1C7EAC2FA4304412C388640004288BC5D1FDFC205453D3EC0C380C2AF3F5C4480C6904403390DC7693DBC3867BB8A4047BF1D38FE4336446A3D0DB66046713C3A3CFFC4"> : tensor<20x20xf16>
    %cst_0 = stablehlo.constant dense<"0x44438540C2B869C2CA2FF6BD4F971BBC05C47CC6B02E15445EC44242FDB8E3C4244063C35EBC11C287B8473B2CC38BBBB13C39A93D3940396FBDC4BF32BD1D3D9A39903DD3BEABC0B4BCC1C4D4C482B26CBEF93F37C14E3F943E574634A9BBC43ABCE041A2B065AFAB44BE3F453A0EC3BA4415BBB3C4434124C3DFC0F6406246FDBD1F2F00C631B7944474C670C44C4383BA414133457CBD573E123D3CB94441652FE3338B450644E8BCA8C579C1743FEEB9EFB782396AC44447C442F43A2CBF3EC00246B4BD7EBFAF419EB909448044644014347BBF3645014507443C4092435BC65BC052BE32B275C091B50A3F78BB0B419437F13A5543AFC26446BEB3304207C47AC1CAC551C350C41EBDEB450EBC943778BF08C525C04F44EE41AEBE5043344475BF22C22CC12FC383BD31409A3C093791A5CAC1113D15BF10C0CD3E26C331B0393D7D3C1542B143533EF8320A3B6DC15840E4450BC577B928C521C40F3B90BAD93A4DBF5642602E2CC4B243B83FCA4231B5554843381247AAAD1CC3B7B82FBE673DCFBF1E44CF38DEC42FC58DBCDDC0BF423030C9C0314253410844FDBD343F86C48FC0374444BD513E9EBE8843E4403EB55AB86038FD40743F2244E5C3FE34FFC00E2DEB3E85BDAA45E0C0E1413BC6D7C0A7BD71BECFC12DBB023DBCB7A9C1AEBC613693BEC1C0963C1D2FE6BE19BA233BDCBA893D33B04BB75F40C7C31B41F9436EC585C1B0C684331CBD20434E3844C2EAC211C22BB98FC448C6E84396C31735843B673FAA422641A0B4B73892C2D741D24235BF6AC1B138CEBB88341AA5003C9C3CC34290C1F7C19CC0422E8DB69B3E4EC2414102BEF435FBBC7E3937420C408DB1D13ADB3852BB1643DAC01BC17E3FED3EF54112C1DEC4BBBE12C192BBE0C61CC6D84462473C3C1141EB3F17C40440D6403BC009347CC43F28EDC79E3CC04197BC34C32DC6F437FD43B5BD2545D63F7BBC3E36D63FB8B9643DB1C1A9C269B1A53E373A1C4436C2DABD18B8E9AA40C13ABDBC3DBDC394432AC2423C0B448D4209B689410442DDBE0A43FBC1233CC1BC724092BA07418A445DC2E54108B7583D02BF60BA1B41B73D913DE73EB0C6A1C451470A407042473994448F48"> : tensor<20x20xf16>
    return %cst, %cst_0 : tensor<20x20xf16>, tensor<20x20xf16>
  }
  func.func private @expected() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x003C003C003C003C003C003C003C003C003C003C003C003C003CD839003C003C7E2B003C003C003C003CF93B003C003C8034003C003C9332003C003C003C913B003C003C003C003C003C003C003C003C003C003C003C8E3ACD32003C003C003C003C9C36003C003C003C302D6539003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C6334C81E003CEB3B003C003CAF35DC3B1B39003C8430003C003C003C6B2E003C003CC93B003C003CE317FF3B003C003C003C003C003C003C003C003C22204338FF3B003CCC2E1524003C8C37F62C003C003C003C003C003C003C003C003C003C003C003C562D003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C71393538003CAF39BB1A003C003C003C003C003C003CFC3BB939003C003CF538003C003C003C003C003C003C003C003CAF38E12E003C003C003C003C7533003C003C003C003C222D003C003C003C003C003C003CE7291A22003C003CB401003C003C003C003C003C003C003C003C003CF93B003C003C003C003C003C003C003C7D37003C003C003C003C003C003C003C003C003C003C003C003C003C003C003CB23A9A35003C003C1B3B003C1A3B003C003C8508003CE229003C003C003C003C003C003C003C003C003C003C3234003C003C4736003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003CBA38003C003C003C003C003C003C003C003CF23B0E39003C0723373B003C003C003C6E32003C003C003CF83B003C003C003C003C003C3439003C003C003CE53B003C982C003C003C003C003C003C003CD337C73A003CFF3B003C003C421D003C003CE53B313A003C003C003C003C003C003C003C003C003C2323003C1536003C003CBD3A003C003CF83B003C003C003CFB3B003C003C003C003C003C003C003C5221003C003CC237ED2A003C003C003C003C003C003C003CD92F003C003C003C003C003C003C632A003CB124003C003CC4362034003CBF34BE36003C4103003CF83B003C003C003C3E347436003C163A003C003C003C003C003C4B39003C362C003C003C7416003C5E3B8E38F921003C"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
