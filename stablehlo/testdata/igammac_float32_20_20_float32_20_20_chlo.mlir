// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %3 = stablehlo.compare  LE, %0#1, %2,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %5 = stablehlo.compare  LE, %0#0, %4,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %6 = stablehlo.or %3, %5 : tensor<20x20xi1>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %8 = stablehlo.compare  LT, %0#1, %7,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %9 = stablehlo.compare  LT, %0#1, %0#0,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = stablehlo.or %8, %9 : tensor<20x20xi1>
    %11 = stablehlo.log %0#1 : tensor<20x20xf32>
    %12 = stablehlo.multiply %0#0, %11 : tensor<20x20xf32>
    %13 = stablehlo.subtract %12, %0#1 : tensor<20x20xf32>
    %14 = chlo.lgamma %0#0 : tensor<20x20xf32> -> tensor<20x20xf32>
    %15 = stablehlo.subtract %13, %14 : tensor<20x20xf32>
    %cst_2 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %16 = stablehlo.log %cst_2 : tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %19 = stablehlo.compare  LT, %15, %18,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %20 = stablehlo.or %6, %19 : tensor<20x20xi1>
    %21 = stablehlo.not %20 : tensor<20x20xi1>
    %22 = stablehlo.exponential %15 : tensor<20x20xf32>
    %23 = stablehlo.and %21, %10 : tensor<20x20xi1>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %28:7 = stablehlo.while(%iterArg = %23, %iterArg_20 = %0#0, %iterArg_21 = %24, %iterArg_22 = %25, %iterArg_23 = %0#1, %iterArg_24 = %26, %iterArg_25 = %27) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %c = stablehlo.constant dense<false> : tensor<i1>
      %62 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %62 : tensor<i1>
    } do {
      %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %63 = stablehlo.add %iterArg_20, %62 : tensor<20x20xf32>
      %64 = stablehlo.divide %iterArg_23, %63 : tensor<20x20xf32>
      %65 = stablehlo.multiply %iterArg_24, %64 : tensor<20x20xf32>
      %66 = stablehlo.multiply %iterArg_21, %iterArg_23 : tensor<20x20xf32>
      %67 = stablehlo.multiply %63, %63 : tensor<20x20xf32>
      %68 = stablehlo.divide %66, %67 : tensor<20x20xf32>
      %69 = stablehlo.subtract %65, %68 : tensor<20x20xf32>
      %70 = stablehlo.add %iterArg_25, %69 : tensor<20x20xf32>
      %71 = stablehlo.divide %iterArg_23, %63 : tensor<20x20xf32>
      %72 = stablehlo.multiply %iterArg_21, %71 : tensor<20x20xf32>
      %73 = stablehlo.add %iterArg_22, %72 : tensor<20x20xf32>
      %74 = stablehlo.divide %72, %73 : tensor<20x20xf32>
      %cst_27 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %75 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %76 = stablehlo.compare  GT, %74, %75,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %77 = stablehlo.and %iterArg, %76 : tensor<20x20xi1>
      %78 = stablehlo.select %iterArg, %63, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %79 = stablehlo.select %iterArg, %72, %iterArg_21 : tensor<20x20xi1>, tensor<20x20xf32>
      %80 = stablehlo.select %iterArg, %73, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %81 = stablehlo.select %iterArg, %iterArg_23, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %82 = stablehlo.select %iterArg, %69, %iterArg_24 : tensor<20x20xi1>, tensor<20x20xf32>
      %83 = stablehlo.select %iterArg, %70, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %77, %78, %79, %80, %81, %82, %83 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %29 = stablehlo.multiply %28#3, %22 : tensor<20x20xf32>
    %30 = stablehlo.divide %29, %0#0 : tensor<20x20xf32>
    %31 = stablehlo.not %10 : tensor<20x20xi1>
    %32 = stablehlo.and %21, %31 : tensor<20x20xi1>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %34 = stablehlo.subtract %33, %0#0 : tensor<20x20xf32>
    %35 = stablehlo.add %0#1, %34 : tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %37 = stablehlo.add %35, %36 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %40 = stablehlo.add %0#1, %39 : tensor<20x20xf32>
    %41 = stablehlo.multiply %37, %0#1 : tensor<20x20xf32>
    %42 = stablehlo.divide %40, %41 : tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %43 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %47 = stablehlo.negate %0#1 : tensor<20x20xf32>
    %48 = stablehlo.multiply %42, %47 : tensor<20x20xf32>
    %49 = stablehlo.subtract %46, %48 : tensor<20x20xf32>
    %50 = stablehlo.divide %49, %41 : tensor<20x20xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51:15 = stablehlo.while(%iterArg = %32, %iterArg_20 = %42, %iterArg_21 = %43, %iterArg_22 = %34, %iterArg_23 = %37, %iterArg_24 = %cst_15, %iterArg_25 = %40, %iterArg_26 = %41, %iterArg_27 = %38, %iterArg_28 = %0#1, %iterArg_29 = %44, %iterArg_30 = %45, %iterArg_31 = %46, %iterArg_32 = %47, %iterArg_33 = %50) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %cst_34 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %62 = stablehlo.compare  LT, %iterArg_24, %cst_34,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %63 = stablehlo.reduce(%iterArg init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %64 = stablehlo.and %62, %63 : tensor<i1>
      stablehlo.return %64 : tensor<i1>
    } do {
      %cst_34 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %62 = stablehlo.add %iterArg_24, %cst_34 : tensor<f32>
      %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %63 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %64 = stablehlo.add %iterArg_22, %63 : tensor<20x20xf32>
      %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %65 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %66 = stablehlo.add %iterArg_23, %65 : tensor<20x20xf32>
      %67 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %68 = stablehlo.multiply %64, %67 : tensor<20x20xf32>
      %69 = stablehlo.multiply %iterArg_25, %66 : tensor<20x20xf32>
      %70 = stablehlo.multiply %iterArg_27, %68 : tensor<20x20xf32>
      %71 = stablehlo.subtract %69, %70 : tensor<20x20xf32>
      %72 = stablehlo.multiply %iterArg_26, %66 : tensor<20x20xf32>
      %73 = stablehlo.multiply %iterArg_28, %68 : tensor<20x20xf32>
      %74 = stablehlo.subtract %72, %73 : tensor<20x20xf32>
      %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %75 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %76 = stablehlo.compare  NE, %74, %75,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %77 = stablehlo.divide %71, %74 : tensor<20x20xf32>
      %78 = stablehlo.subtract %iterArg_20, %77 : tensor<20x20xf32>
      %79 = stablehlo.divide %78, %77 : tensor<20x20xf32>
      %80 = stablehlo.abs %79 : tensor<20x20xf32>
      %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %81 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %82 = stablehlo.select %76, %80, %81 : tensor<20x20xi1>, tensor<20x20xf32>
      %83 = stablehlo.select %76, %77, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %84 = stablehlo.multiply %iterArg_31, %66 : tensor<20x20xf32>
      %85 = stablehlo.subtract %84, %iterArg_25 : tensor<20x20xf32>
      %86 = stablehlo.multiply %iterArg_29, %68 : tensor<20x20xf32>
      %87 = stablehlo.subtract %85, %86 : tensor<20x20xf32>
      %88 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %89 = stablehlo.multiply %iterArg_27, %88 : tensor<20x20xf32>
      %90 = stablehlo.add %87, %89 : tensor<20x20xf32>
      %91 = stablehlo.multiply %iterArg_32, %66 : tensor<20x20xf32>
      %92 = stablehlo.subtract %91, %iterArg_26 : tensor<20x20xf32>
      %93 = stablehlo.multiply %iterArg_30, %68 : tensor<20x20xf32>
      %94 = stablehlo.subtract %92, %93 : tensor<20x20xf32>
      %95 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %96 = stablehlo.multiply %iterArg_28, %95 : tensor<20x20xf32>
      %97 = stablehlo.add %94, %96 : tensor<20x20xf32>
      %98 = stablehlo.multiply %83, %97 : tensor<20x20xf32>
      %99 = stablehlo.subtract %90, %98 : tensor<20x20xf32>
      %100 = stablehlo.divide %99, %74 : tensor<20x20xf32>
      %101 = stablehlo.select %76, %100, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      %102 = stablehlo.subtract %101, %iterArg_33 : tensor<20x20xf32>
      %103 = stablehlo.abs %102 : tensor<20x20xf32>
      %cst_39 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %104 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %105 = stablehlo.select %76, %103, %104 : tensor<20x20xi1>, tensor<20x20xf32>
      %106 = stablehlo.abs %71 : tensor<20x20xf32>
      %cst_40 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %107 = func.call @integer_pow(%cst_40) : (tensor<f32>) -> tensor<f32>
      %108 = stablehlo.broadcast_in_dim %107, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %109 = stablehlo.compare  GT, %106, %108,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %cst_41 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %111 = stablehlo.multiply %iterArg_25, %110 : tensor<20x20xf32>
      %112 = stablehlo.select %109, %111, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_42 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %113 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %114 = stablehlo.multiply %71, %113 : tensor<20x20xf32>
      %115 = stablehlo.select %109, %114, %71 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_43 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %116 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %117 = stablehlo.multiply %iterArg_26, %116 : tensor<20x20xf32>
      %118 = stablehlo.select %109, %117, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_44 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %119 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %120 = stablehlo.multiply %74, %119 : tensor<20x20xf32>
      %121 = stablehlo.select %109, %120, %74 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_45 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %122 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %123 = stablehlo.multiply %iterArg_31, %122 : tensor<20x20xf32>
      %124 = stablehlo.select %109, %123, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_46 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %125 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %126 = stablehlo.multiply %iterArg_32, %125 : tensor<20x20xf32>
      %127 = stablehlo.select %109, %126, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_47 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %128 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %129 = stablehlo.multiply %90, %128 : tensor<20x20xf32>
      %130 = stablehlo.select %109, %129, %90 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_48 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %131 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %132 = stablehlo.multiply %97, %131 : tensor<20x20xf32>
      %133 = stablehlo.select %109, %132, %97 : tensor<20x20xi1>, tensor<20x20xf32>
      %cst_49 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %134 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %135 = stablehlo.compare  GT, %82, %134,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %136 = stablehlo.and %iterArg, %135 : tensor<20x20xi1>
      %137 = stablehlo.select %iterArg, %83, %iterArg_20 : tensor<20x20xi1>, tensor<20x20xf32>
      %138 = stablehlo.select %iterArg, %82, %iterArg_21 : tensor<20x20xi1>, tensor<20x20xf32>
      %139 = stablehlo.select %iterArg, %64, %iterArg_22 : tensor<20x20xi1>, tensor<20x20xf32>
      %140 = stablehlo.select %iterArg, %66, %iterArg_23 : tensor<20x20xi1>, tensor<20x20xf32>
      %141 = stablehlo.select %iterArg, %115, %iterArg_25 : tensor<20x20xi1>, tensor<20x20xf32>
      %142 = stablehlo.select %iterArg, %121, %iterArg_26 : tensor<20x20xi1>, tensor<20x20xf32>
      %143 = stablehlo.select %iterArg, %112, %iterArg_27 : tensor<20x20xi1>, tensor<20x20xf32>
      %144 = stablehlo.select %iterArg, %118, %iterArg_28 : tensor<20x20xi1>, tensor<20x20xf32>
      %145 = stablehlo.select %iterArg, %124, %iterArg_29 : tensor<20x20xi1>, tensor<20x20xf32>
      %146 = stablehlo.select %iterArg, %127, %iterArg_30 : tensor<20x20xi1>, tensor<20x20xf32>
      %147 = stablehlo.select %iterArg, %130, %iterArg_31 : tensor<20x20xi1>, tensor<20x20xf32>
      %148 = stablehlo.select %iterArg, %133, %iterArg_32 : tensor<20x20xi1>, tensor<20x20xf32>
      %149 = stablehlo.select %iterArg, %101, %iterArg_33 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %136, %137, %138, %139, %140, %62, %141, %142, %143, %144, %145, %146, %147, %148, %149 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %52 = stablehlo.multiply %51#1, %22 : tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %54 = stablehlo.subtract %53, %30 : tensor<20x20xf32>
    %55 = stablehlo.select %10, %54, %52 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %57 = stablehlo.compare  EQ, %0#1, %56,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %59 = stablehlo.select %57, %58, %55 : tensor<20x20xi1>, tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %61 = stablehlo.select %6, %60, %59 : tensor<20x20xi1>, tensor<20x20xf32>
    stablehlo.custom_call @check.expect_almost_eq(%61, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %61 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}, tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x0490303E114F64C0C7B2A3C07707A13FE659153EC5D911404433AA40A8A58FC04CBC30C0362861C0DBE927406298BDBF6D06AA3F37561940D53256BFE266ADC0BD65C2BF63FA19C05902C3BFA0201C40B4AD1BC054B1B5C03FC2BB4059398A40AF818E3F5DCEA04028DB4F4009F2EC3F4DCA36C07825BBBF775DA3C0F76A343F0C8DDFBF2755C6BF73CE3C4079ED863E75983DC088C791BF66371D3F877B8440D66F02C0CB0E44C035CD0F403E07C73FAA2374C0274F14BEDAC0BE40AD0A39409E008240640F6EC02831A740B6DF9F3FC887A3BEC3350C4002EDD4C081E20E4030F96540D250D83E584E5BC06651FDBE0A0D79BFE139FD3E8CB3163DD428E73F979D983F2DA4D93F715DC53FE22FAD409A4C0A40B00D68C0BFE381C0F72FA0400821CB3E6E8011BF0DE6553B9837C23F9D1C97406DA9CBBFB09692BFAB1D954057CF16BE1ED7FEBF309803408C0576C066698DC0905D6BBFF25CD9BFE303A0BED77DF63EDCA10FBF44860EC017AB383FD4C570408A71E5BF81F004400CED273FAAD757BE896E1BC0832099BBC1F8F53E093CBD3F414429C05834E53F9F8464406C52923F0C62AABF3536CD3E26D98BBF58F6B43FCB3E3ABF23D859BF86EF83C056820D40F995A13FBEC74BC032D59A3FECED6E409558983E39C72F40701392BE7904143FF800E0BE4EBA88BF39DE82BFBD008140D6FC913F102702C01CEEB6C02ACE8340208745BE0D8B3AC00662404044D87E3F0735D3C0E05EE23F6AF696BF8728EA40124F2ABFAFEFDDBFF5B12A40B6B9FB3FEACEFBBED02C32C0E456F23F17B8263F9B7278C0ECAC69BF26334FBFD920C440F7CE95C0548894402821B2C0B5AB1E40B3AA44BE41345C3E10DCAFBEF3E01E4008558ABE51EBA1BF23A4DFC008DCBFBFC47863C09CEB9A40FE14AABF820078C0C10A02C069866F3FC610BD401C442340E49E12C0ADE8AAC078A253C055ACBD3F199FF2BF30CD4340F15E90401B7382BF18E798C04106ABBEA832A3C0E6D903C0AE3DB73F1DCA933F3292853F7EE2FA3EE61A20408D54B0BFCA444C3E87D500C0F0C282BF61BB81C03D8620BFF4C1BDBE162C604091E70FC06F37B8C0879E063F3C2BF23E16E51B3FD54B2540E0B03040D31694BF9F24433FA4F7EF3FCBCBEBBFE87BEE3E509D08C06BB99FC032F9AC3F784756409670933F971976BE3B704A406B1EC4BED1F09B40ADC5A5BF0C7FF3C03949BB4053334C403B554640E09EA03F0CE9AFBF19C4933FEA316B40BAF79C3F2E8E5540506EBEC09084043F5AA054BE75763CBF842D39BF855C9ABFECB6A63F5EBD0E3F487538C0E02189C0B03E6140536636C00D78DDC06B6BF33DDC5C51C098962340F8438F40A489F3C0E47FA0BF07220E403D9A79C000F59E3FD0EC92400B666AC061368B40E2CDEB3F96389BC04D0410BFC5AA2E3FC8F886BE4C359CC0BE7D7240F954A8BEA11354C0B518A53F225AC7C0E7BDC83F69FBFE3E6C6803C050C6A040CA537ABF30209AC029A3733FEBC50BC083CE1440CA0999BFC8B4DF3FCF6FA8C0B57B8DBE5A37154005759C3F5BA6913F81C36840893D8FBEB2182840E77296BFFC86EB3FD9B024408E61FBBFCA7373C008872C3EC2683040A1B59A3FFE90503D87C6224034B301BEEDD75ABFED9A88BF3FEF994052F0C33FC0ABB6C04CC46DBF3EE9E4BF5B3717405C4616BFECEFA93EBC8061BFD0A57B408F9B2F3F48BEA0BF1CA25A4036E559BF837708C02F9CD83E94BEA9BF86E9A440ECC168C01BB7333F7615EEC0691EAC3F12737B40CFA8A33E3ACFCC3F0C0302400BE2ABC04830224004EC973E30689ABF6E6E0BBF033081BFCEA920BF9AB019BFFF1C3B3FCF4AF7BD887AAEBE4B2935C0475691C0A16E4CBEA5A0BB405266C33F9797C43F15E64740DD1D6CC0E8D7E5BF7A404C40064B9F3F434CBBBF08190C408EC599404DAB3CBF9FF0223F82E162C0CEFC83C03AD662402823773D26C6AAC0873A9140DA8020407FA60F40B0F9BFBF76D60140AD16F13FA3AD4AC0975F794085FB45C08C4F2840306F49C0146E2CC00DC68FC0CB53CB3EF9D60C403B4741405603C43F112F6BBF43084CBD68C31B4011CBD93F264D324037A966C068D6B63EB13AB9BF09B2C83F42CA6DC0F8DB1CC0D04024C0E60E65C02B8FBCBF3D7059C0D8ECF5BFC102CFBE328AB3BF8D4BFFBE0799643F86ED7640FC8DD8BEA291803F6976E83F8A1A8D40A891464034C61340E0F89440B81A063F9C18B6BF4FFB8BC0"> : tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<"0x58B47D406489FDBFCBAAA1BF99AA86C02A4E9DBF10EB1C40F43126403AB2C63F4A2C2FC0AEB825C09A87A5C0A3DABEBFA4C447408BA0AB404120CFBF6FE78A404AD689C0C34A09407DDF1F40DFA5E5BF8819B63DFB993240DE45AA403B09F5BF5C36AF3F79063F40180CD43F96B51CBF1758F13F428F0F40019E3FC0BE5803407AA69BBF5D63CE40055DFABF6CFA0F3E80B148408D60E440A2B0D4BEDAC262BFA50FD1BF75808CC00E6637C06EE67B3F612B9F40CD92FF3FEB298EC0515195C0D355453FECC786C0B2FF4240B166B3BF9DD3BDBFF9F906C0098A3BC02F73B9C0899A2540DB17673F81F451BF3BA89CBD9AB901C04A7F4C4097AAB8C0FA4AB73F245F6640EB0EF83E9B8D1A40F3C18BC0A50688BF53DD2BBF854C9D3F35023F3E23A4F6BD7FA201C01FC2F53F733D1740C7B5BB3F748480BFD7B2643FEACA85BFDA4F31C0BB2FA4C05A02F83E9944203FAA9D023F2A64AF3E4BAD86BFC024BB3F2CE7F5BFE76AA9BF2E750FBFAFF9FFBFFE51893CDB38F03F8CB485C01972D5BEB418A83F76E08740446AC4BFDC5595C0639DA63F7C931C4018341FBFABC6A5406221823DD24702C1EF5CC7BD96AF653F49919E40145180C04A1DCEBE2518823F28095FC0713A74BF0A0780C02EA6BEC059AA3CC0C1787C3E65FACF3E8D3848C062182BC09E6181BF411F53BFDDAD79BFBFED2A400906943F6A39B7BFE322A9BFC0219EBF02C1E0BE32778EC07E33C63F86DEEF3F401AE63E71B41AC0A74EBA40FA0B1CC0086319BF2748604080A0B4404C9C72C01A5386BF32C205C0F14108C0C55B823FA5782DC0F4E809C051E2E4BF4F7458405973CDBFD81E57C0C56DAFC09F4B5F40D04149BF4EBE6B3EE71EC5BF698295C0FDDBA5C0D83990C033BB3ABF20580E40D8CC0FC05CE73FC0E4636DC073FBFC3E9CD4E0BF112E1740C34957408189813FD2C33E40CB148EC07DAF0BBD4EF546C0CC4D81BF5968EBBF32276DBF57CD56C0C3A2C23FC17913C062A8DB3ED9E60FC05142F3BF894CF5BE07DF863F48025AC00C5882BE30E7DBBFDFB88140DF90CEBED22508BD0A8CB7BF7B839A3EB66017C00C855940C51F26BF223DB43E755B2B40010EACC06D61B8BF80E2EC3E79A061BF39C760C077B0D8BEE23E69BF36E39440E33E9BBF08AE1BC0FE4A5F40AFC278C03FD9133F7562BBBF2681C73F9B2B3BC048B3BA3F60F5BA3FC011BEBF7E7A4DC02ACCB9BE2EC4FCBF1626E1C0951A7C40F06E58C0375814C019EE22404C805C3FB87D6B40A2670FBE028D3CC040942FBF45260C407B0ACEC064C4F93FC8CCB2BF3D44CEBF1923123F90D0F2BFEFF78AC0982C87C067B851BE7B28823F16DE7640C61FDBBFE932F9BFCF0855C0BF06403D859D7B40492816BF7D478C40EE7B16C0200E0340CD8EDFC0768F5740AA1280BF579EBC3F6C4696BF97AC03C0933F4BBFB158103FDCE8E6BFDCF19C400759924018F3BFBF30F9B2409B83E93F471A1640F2C5EB3F07227D3FF080ED3FD2DDDC3DB4704A40EE3E3FC094E6853F64169D3DC8265A4010E5EE3EA923A43F2069CE3FA16DF63FCEB29ABF5BC80B408E299540D729A53F6CA0B9BF2BB93240A0FC81C0174BB63F1755DF3F462C15C02C03B5BF9C1D7BC0B3CDC73DFEC2B9BF760CBEBF6F034B408E89C14040508A3DEAFCF9BF3584943FC971E23E74F2FEC0E7184AC065B41CC0D61826405518B03F5509D5C0441B23C029B85EC03CBECEBEE3DDDB3F406A69400D2F3940977B99BF6CFC49BF2844BDBF744962406047F9BF603F1A40CE62C5BFEBB18CC0EF5A974007964240D5A5B8BD16D6E1BF956EA03DB01F0B404809CCBE558386BFBDD325BFA0FABD40224D69BFAC9B54BF31361FC0853C30403CE4B8C0D4D61740E9642AC035F0D4BF7371D5BFEDA39CBF8CAF5CC08B869C3F8C0346408CCBC6BFC9467FBF905E5EC0F3586FC0CFAE90C096DD59C0CB460F407C9339C065EFB44030532D403F0501C0CF2239C0D01F3240B7D474C05ACF1A40E776003FE89289C07420A73E595F17C07C4CB5BFD926F83F896A1AC07EEDDCBFFA8381402D6A973F00C68740BE320A40E1843A3FA0FBE6BE349C92BF80D1913F794F15BE7E1549BFE90157BEF3853E3FC2F1D640FA5C09C0979CFB3E044F03C014350340A42321C05765443FDF14463FA67136C054FD2040F78A48C00EBB1D407CF18E3EEE1AD340E8060CC0E470E73F119D2E40CC15A6BFD0EA09C0C5A535C00D8E7340098039C00997EF3F"> : tensor<20x20xf32>
    return %cst, %cst_0 : tensor<20x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x5B1E7C3A0000803F0000803F0000803F0000803F0332BD3EE54C683F0000803F0000803F0000803F0000803F0000803FC002A03D1C694E3D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F9490093F0000803F90BD963EE234523F70A1503F0000803F0000803F0000803F0000803FAAF5943D0000803F0000803F0000803F2EB1B73E0000803F0000803F0000803F0000803F0000803F0000803F0000803FDA6F193F0000803F0000803F0000803F0000803F632B7E3F0000803F8112563F0000803F0000803F0000803F0000803F0000803F12A2283F58BC153E0000803F0000803F0000803F7293383C0000803F74A6043F9DA9223DC61C5C3F54A8463E0000803F0000803F0000803F0000803FE5FF7F3F0000803F0000803FD4FB3B3958A5493E44A8793F0000803F0000803F0000803F0000803F0000803F771E6C3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC0E7E53E0000803F0000803F5CAF363E1516763F0000803F0000803F0000803FE3D0873C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF2E79B3E735A7C3F0000803F0000803F0000803F0000803F0000803FACEF393F24B9BD3E0000803F0000803F0000803F0000803F0000803FA8374C3FBB121C3E0000803F0000803F0000803F0000803F0000803F0000803FD2FC643D0000803F0000803F0000803F0000803FC2AB5A3E0000803F0000803F0000803FD477623F0000803F0000803F0000803F4AA15F3E0000803FF8A06E3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF1C4AB3D94BE5D3F9EDA5A3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F6DDDBB3E0000803F0000803F0000803F0B258A3A0000803F0000803F0000803F0000803F0000803F42D7E63E0000803F0000803F14A3B73C0000803F0000803F0518793F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5A387D3F0000803F0000803F0000803F0000803F6FE47A3F0000803F0000803F0000803F0000803F0000803F5ADB033D0000803F0000803FB0862F3FA8E4053FCC50B73E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5683E83C0000803F0000803F0000803F0000803F0000803F3920FE3D0000803F94ACA83C0000803F0000803F0000803F8212013E0000803F0000803F0000803F0000803F0000803F13EE7E3F0000803F0000803F904F9A3C0000803F56A6473C3A84643D0000803FAC29763F0000803F0000803FFA3C623F0000803F0000803F0000803FB24A7E3F0000803F0000803F58E5393F2516893E11C1373E0000803F0000803FA311E23D0000803F0000803FDEADBB3E0000803F0000803F19CF713C0000803F0000803F0000803FE6D07F3F0000803F0000803F0000803F0172803EE7137D3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F9FC8713F0000803F0000803F0000803F0000803F0000803FC9E7AA3B0000803F0000803F0000803F0000803F0000803F0000803FD3B3433F0000803F0000803FBB02583D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F3FAFA43A0000803F0000803F0000803F0000803F0000803F55ED753F0000803F0000803F0000803F0000803F0000803FCD4A663F77AF8F3D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0FCF1C380000803F0000803F0000803FF241943E0000803FC79C9F3ED5D7633F0000803F9DE07F3F0000803F0000803F0000803F0000803F0000803F5804443B9CAB393FF6DA543EBAC4723E0000803F0000803F0000803F6ED0163F0000803F0000803F0000803F0000803F7CE38D3B0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8E7D8E3D4EEC7F3F0000803F0000803FE2ACCE3E5A64473F0000803F0000803F0000803FAC59CE3B0000803F0000803F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @integer_pow(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
