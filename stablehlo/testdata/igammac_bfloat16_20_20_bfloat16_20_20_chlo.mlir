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
    %64 = stablehlo.convert %63 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%64, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %64 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}, tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x184051C0AA3F31408E3FA8C0C93ECEC0663DCCBF353FC53FF83FD63FC7BE88BF24C053C081C000C018400AC00741D9BF8BBEF0BE2DC055C02E40ECBEE33FC93F9A40CCC0553C8640AEBF28C0D8BF0C3FCFBF97BF1FC0B3C025401DC0BEBF3F3FDB3FF7BD20406FC0C640F83F40BFD8BE71C0A5C036402440DF3F6040B44089400EC03A403AC0953F83400140A13FD640F63F433F37C04F4002C0923F12405E4058C012C084C069C0CABF1C408CBD07C08AC08DC009402FC0DFBFF2BFCEBF0AC0EEBF9DC02BC091C0D140EEBFA0C0D13F73408DBE8040A33F1A40A9BF874010408AC023C058C012BFC63F473F6E40883E94BDD5405F4038C03CC0D73FF5BF8C3FD43EE7BF8ABF74C054BF57C064400C41554087BF1BBF1E401DC0D0C027C0A640F3BFEDBF8C3E493F2D4092406A40C93FB5BCFEBFA5C05F404E40AB3FB7C087C0943F20C050BF15C096BE5640A54084C056C0603FA9BFD23FC2BFCE3F60400AC087C047BF43BFAEBF4EC017C02E40B2BF7A40CD3FDB3E09C0A5BFDDBF10C09F3FC6BFF33F8240764004C034BFAF3FA040DC3E2BC032BEA4BF1BC0A7C061C0D74094BE013FC73F3EBF2940334023BDC0C04DC02340234053401C40CB40D4BFB6BF18402540B4BFCD3EF3BD2A3FCBBFB3BF1F4020BFB5409BC085BFB43F31C0BABFE9BE6D40EC3F18C0BC40A7C041C0A0BF71C0CDBE34C098408FC06B405C3F5FC0B04064402A40463F01BF48C02C40D94061C0A0C086C0143F52C0B43E024091BF39C0C33F603F5EC038400440414028BFCB3F80BE36BFD03F27409F3FF43FA6C09D400240F2C08B4023C03BBF1A40BF407BBE33404B3F38BF3240D2BFB3C001407A4002400240F13E0A3F6340103E4F40C8BF91BF31408C3F3B40F1BF404018408ABF15C09C3FCA3FEC3E79BF3F40EF3F7140E0BF24C09740A63F38BFA33E44C0674070BFE93F2A406B3F9D4090BE73C0BD3FB4C0FA3F83C0FEBFEDBF88BF2DC03C4040C0413F143FDDBF1ABE61C0AE408A4031C0D03F0ABF1340A1407CBF9ABE9ABF2DC09C405F3F80406440B23F48403E4072BF1240C2402840E2C0FB3FD7BF0240E240623FF8BF3B40284048BFDCBF24C07240973E0EC083C039C049BF5C3F"> : tensor<20x20xbf16>
    %cst_0 = stablehlo.constant dense<"0xC33F103E0FC0BF3F8AC0B63F9FC01AC0A5BF4E40414033BF8D3F56C0104062BF144067C0A3C026C0334098C03940934011C0B3401FC09E40A3BFA64054BFFDC041409EBF81C014BF20BD4DC05BC0BC3F853E0BBFD740133F26C02640F33FF03E973FEEBFDC405DBF0240B13FF8C038BF6B3EE73FA3C037C00EBDC83F1AC022C01BC0B64062409A3FAAC034C026C0FEBF0540C73FB53E9D3FA9409A4040C019BF88C0B940D43EB7BF0240FFBFD03FC13FA5C007409DC0943F97BFD63F713E43C01C405240713DB63F283FA13FC3BFB7C08AC00740B6C01AC0074048C03C409A402440804020C0B1BF8140D5BFF53FFF4072409CBF8340E03E08C0A8C002BFBCBF643D84BEEBBFF43F80BD82402CBF14403C40EF3F3BBF93401DC0B24023C0BC3F4B3FBDC07FBFB3BF15BFE9401CC0813F25BEA6BE123EB9BE17C05B40E3BD0DBE9EBFB23F5BC02C401C4116403E3FB93FE8BF064097400AC0EEBF71BEE9C06E40423F9C40983F21C16C3FC7BF4240C440BBBF33C0794010406F3F91BFAF3F2CC09E40A43FE1BF00C092C0B1BEBBBF9CBD673FE8BF0640D33F77402B409FC0DF3F14BECBC097BDB8BE04C0C040F5BD3040D540E73F2DC09E3F85C036400BC083C083C015C024C02FC0D33FE6BE90C00E40303C51BF3D40533F04C024C0C13F783FA23F583F573F0B3E03C024BF963E07411D4067C04AC01AC088C078408540873FC440CBBF913F35405B402ABB34BF9540963C28BFDABF5A3FB43E97C0A9BFA2C0844008BF5A40B140CC3E4A409440B0408DC038BF3E40604006C0C8C0203E4C3E7A3D2BBF1BC0B5C0EDC016C0C4BF86C0E2BF5D3F7EBEC63FDC3F36BE8FBF64BE27C0ABC0E5BFC3C00C40183F2CBFF83E91BF88BFC7C00640EBBF02BF214001C1D2BF7240CEC0BE401DC08E3F94BE2840A33F75C03840C4BF04C0A13F8340054094BF4A40903F043FD0C08A40F7BFC940D0BE213FEAC0F1BFF0C073C0933F1F41553F47C039BEB9BF0D40E83FE23F19C027C02EC06DC0264035BFC8C001BE7B40EBBF93BF22C0C9405FBEFC3E30C03840E93FC1BFA54097BE4FC0EFBDF23F71BF63BFC4BFCF3EA23F793F8F3E3D40C34034403B4098BC78BE5740223F8D3FCE3E"> : tensor<20x20xbf16>
    return %cst, %cst_0 : tensor<20x20xbf16>, tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x293F803F803F443F803F803F803F803F803F803FD23C803F2E3F803F803F803F803F803F803F803FA33E803F7E3F803F803F803F803F803F803F803F803F803F493F803F803F803F803F803F803FC93D803F803F803F803F803F803F803FFA3E143F803F8E3C803F7C3F143F803F803F803F803F803F803F803F603F803F803F803F903D803FBA3E803F803F803F803FBA3E113E803F673F803F363C803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803FFC3E803F343F853D803F803F803F803F453D803F563FAC37803F803FA03E803F803F803F803F803F2A3F803F803F803F803F803F803F803F043F803F803FCA3D803F803F803F7D3F803F803F803F803F803FDF3D803F183F803F803F803F803F803F763D803F803F803F803F803F803F803F2A3F803F803F803FCA3D803F803F803F803F803F803F803F803F803F803F803F803FB33E803F803F803F843B803F803F803F803F803F803F1A3F803F803F803F803F803F803F193E803F803F803F803F803F803F7F3F803F803F803F803F803F4B3D803F803F803F1F3F803F693F803F733F803F803F803F803F803F803F803F803F803F803F803F803F653F803F803F803F803F803F803F7B3F413F803F803F803F803F803F803F803F803F803F803FC83E393C803FB13E803F573F173D803F803F803F493F803F803F803F693E803F803F803F803F803F803FD03C803F7D3F413E263E803F803F803F803FB03D803F803F7C3F803F803F803F803F803F803F803F803F803F803F6C3F803F803F343F803F803F803F803F803F803F803F283D7E3F803F7E3F803F803F803F133E803F803F0A3F803F803F803F803F143C803F803F803F6D3E723F803F803F803F803F803F023B803F803F803F223F783F803F0D3F803F803F803F803F803F803F803F803F803F803F713F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803FAE3A803F7F3F803FF73E383F803F533D803F803F803FD93E803F803F803F1C3F803F6B3F7E3F803F803F803F1F3F803F803F803F803F803F193F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
