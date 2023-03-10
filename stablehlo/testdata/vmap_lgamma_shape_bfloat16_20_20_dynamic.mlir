// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %2 = stablehlo.constant dense<1.14472985> : tensor<f32>
    %3 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %4 = stablehlo.constant dense<0.918938517> : tensor<f32>
    %5 = stablehlo.constant dense<2.01490307> : tensor<f32>
    %6 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
    %7 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %8 = stablehlo.constant dense<1.50563267E-7> : tensor<f32>
    %9 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %10 = stablehlo.constant dense<9.98436917E-6> : tensor<f32>
    %11 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.constant dense<-0.138571098> : tensor<f32>
    %13 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %14 = stablehlo.constant dense<12.5073433> : tensor<f32>
    %15 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %16 = stablehlo.constant dense<-176.615036> : tensor<f32>
    %17 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %18 = stablehlo.constant dense<771.323425> : tensor<f32>
    %19 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %20 = stablehlo.constant dense<-1259.13916> : tensor<f32>
    %21 = stablehlo.constant dense<676.520386> : tensor<f32>
    %22 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %23 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %24 = stablehlo.convert %arg1 : (tensor<?x20x20xbf16>) -> tensor<?x20x20xf32>
    %25 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.concatenate %26, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %28 = stablehlo.dynamic_broadcast_in_dim %23, %27, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %29 = stablehlo.compare  LT, %24, %28 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %30 = stablehlo.negate %24 : tensor<?x20x20xf32>
    %31 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %32 = stablehlo.reshape %31 : (tensor<i32>) -> tensor<1xi32>
    %33 = stablehlo.concatenate %32, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %34 = stablehlo.dynamic_broadcast_in_dim %22, %33, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %35 = stablehlo.subtract %24, %34 : tensor<?x20x20xf32>
    %36 = stablehlo.select %29, %30, %35 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %37 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %38 = stablehlo.reshape %37 : (tensor<i32>) -> tensor<1xi32>
    %39 = stablehlo.concatenate %38, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %40 = stablehlo.dynamic_broadcast_in_dim %22, %39, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %41 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %42 = stablehlo.reshape %41 : (tensor<i32>) -> tensor<1xi32>
    %43 = stablehlo.concatenate %42, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %44 = stablehlo.dynamic_broadcast_in_dim %21, %43, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %45 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %46 = stablehlo.reshape %45 : (tensor<i32>) -> tensor<1xi32>
    %47 = stablehlo.concatenate %46, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %48 = stablehlo.dynamic_broadcast_in_dim %22, %47, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %49 = stablehlo.add %36, %48 : tensor<?x20x20xf32>
    %50 = stablehlo.divide %44, %49 : tensor<?x20x20xf32>
    %51 = stablehlo.add %40, %50 : tensor<?x20x20xf32>
    %52 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %53 = stablehlo.reshape %52 : (tensor<i32>) -> tensor<1xi32>
    %54 = stablehlo.concatenate %53, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %55 = stablehlo.dynamic_broadcast_in_dim %20, %54, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %56 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %57 = stablehlo.reshape %56 : (tensor<i32>) -> tensor<1xi32>
    %58 = stablehlo.concatenate %57, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %59 = stablehlo.dynamic_broadcast_in_dim %19, %58, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %60 = stablehlo.add %36, %59 : tensor<?x20x20xf32>
    %61 = stablehlo.divide %55, %60 : tensor<?x20x20xf32>
    %62 = stablehlo.add %51, %61 : tensor<?x20x20xf32>
    %63 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %64 = stablehlo.reshape %63 : (tensor<i32>) -> tensor<1xi32>
    %65 = stablehlo.concatenate %64, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %66 = stablehlo.dynamic_broadcast_in_dim %18, %65, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %67 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %68 = stablehlo.reshape %67 : (tensor<i32>) -> tensor<1xi32>
    %69 = stablehlo.concatenate %68, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %70 = stablehlo.dynamic_broadcast_in_dim %17, %69, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %71 = stablehlo.add %36, %70 : tensor<?x20x20xf32>
    %72 = stablehlo.divide %66, %71 : tensor<?x20x20xf32>
    %73 = stablehlo.add %62, %72 : tensor<?x20x20xf32>
    %74 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %75 = stablehlo.reshape %74 : (tensor<i32>) -> tensor<1xi32>
    %76 = stablehlo.concatenate %75, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %77 = stablehlo.dynamic_broadcast_in_dim %16, %76, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %78 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %79 = stablehlo.reshape %78 : (tensor<i32>) -> tensor<1xi32>
    %80 = stablehlo.concatenate %79, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %81 = stablehlo.dynamic_broadcast_in_dim %15, %80, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %82 = stablehlo.add %36, %81 : tensor<?x20x20xf32>
    %83 = stablehlo.divide %77, %82 : tensor<?x20x20xf32>
    %84 = stablehlo.add %73, %83 : tensor<?x20x20xf32>
    %85 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %86 = stablehlo.reshape %85 : (tensor<i32>) -> tensor<1xi32>
    %87 = stablehlo.concatenate %86, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %88 = stablehlo.dynamic_broadcast_in_dim %14, %87, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %89 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %90 = stablehlo.reshape %89 : (tensor<i32>) -> tensor<1xi32>
    %91 = stablehlo.concatenate %90, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %92 = stablehlo.dynamic_broadcast_in_dim %13, %91, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %93 = stablehlo.add %36, %92 : tensor<?x20x20xf32>
    %94 = stablehlo.divide %88, %93 : tensor<?x20x20xf32>
    %95 = stablehlo.add %84, %94 : tensor<?x20x20xf32>
    %96 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %97 = stablehlo.reshape %96 : (tensor<i32>) -> tensor<1xi32>
    %98 = stablehlo.concatenate %97, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %99 = stablehlo.dynamic_broadcast_in_dim %12, %98, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %100 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %101 = stablehlo.reshape %100 : (tensor<i32>) -> tensor<1xi32>
    %102 = stablehlo.concatenate %101, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %103 = stablehlo.dynamic_broadcast_in_dim %11, %102, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %104 = stablehlo.add %36, %103 : tensor<?x20x20xf32>
    %105 = stablehlo.divide %99, %104 : tensor<?x20x20xf32>
    %106 = stablehlo.add %95, %105 : tensor<?x20x20xf32>
    %107 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %108 = stablehlo.reshape %107 : (tensor<i32>) -> tensor<1xi32>
    %109 = stablehlo.concatenate %108, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %110 = stablehlo.dynamic_broadcast_in_dim %10, %109, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %111 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %112 = stablehlo.reshape %111 : (tensor<i32>) -> tensor<1xi32>
    %113 = stablehlo.concatenate %112, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %114 = stablehlo.dynamic_broadcast_in_dim %9, %113, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %115 = stablehlo.add %36, %114 : tensor<?x20x20xf32>
    %116 = stablehlo.divide %110, %115 : tensor<?x20x20xf32>
    %117 = stablehlo.add %106, %116 : tensor<?x20x20xf32>
    %118 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %119 = stablehlo.reshape %118 : (tensor<i32>) -> tensor<1xi32>
    %120 = stablehlo.concatenate %119, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %121 = stablehlo.dynamic_broadcast_in_dim %8, %120, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %122 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %123 = stablehlo.reshape %122 : (tensor<i32>) -> tensor<1xi32>
    %124 = stablehlo.concatenate %123, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %125 = stablehlo.dynamic_broadcast_in_dim %7, %124, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %126 = stablehlo.add %36, %125 : tensor<?x20x20xf32>
    %127 = stablehlo.divide %121, %126 : tensor<?x20x20xf32>
    %128 = stablehlo.add %117, %127 : tensor<?x20x20xf32>
    %129 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %130 = stablehlo.reshape %129 : (tensor<i32>) -> tensor<1xi32>
    %131 = stablehlo.concatenate %130, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %132 = stablehlo.dynamic_broadcast_in_dim %6, %131, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %133 = stablehlo.add %132, %36 : tensor<?x20x20xf32>
    %134 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %135 = stablehlo.reshape %134 : (tensor<i32>) -> tensor<1xi32>
    %136 = stablehlo.concatenate %135, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %137 = stablehlo.dynamic_broadcast_in_dim %5, %136, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %138 = stablehlo.divide %36, %132 : tensor<?x20x20xf32>
    %139 = stablehlo.log_plus_one %138 : tensor<?x20x20xf32>
    %140 = stablehlo.add %137, %139 : tensor<?x20x20xf32>
    %141 = stablehlo.divide %133, %140 : tensor<?x20x20xf32>
    %142 = stablehlo.add %36, %28 : tensor<?x20x20xf32>
    %143 = stablehlo.subtract %142, %141 : tensor<?x20x20xf32>
    %144 = stablehlo.multiply %143, %140 : tensor<?x20x20xf32>
    %145 = stablehlo.log %128 : tensor<?x20x20xf32>
    %146 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %147 = stablehlo.reshape %146 : (tensor<i32>) -> tensor<1xi32>
    %148 = stablehlo.concatenate %147, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %149 = stablehlo.dynamic_broadcast_in_dim %4, %148, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %150 = stablehlo.add %149, %144 : tensor<?x20x20xf32>
    %151 = stablehlo.add %150, %145 : tensor<?x20x20xf32>
    %152 = stablehlo.abs %24 : tensor<?x20x20xf32>
    %153 = stablehlo.floor %152 : tensor<?x20x20xf32>
    %154 = stablehlo.subtract %152, %153 : tensor<?x20x20xf32>
    %155 = stablehlo.compare  LT, %28, %154 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %156 = stablehlo.subtract %34, %154 : tensor<?x20x20xf32>
    %157 = stablehlo.select %155, %156, %154 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %158 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %159 = stablehlo.reshape %158 : (tensor<i32>) -> tensor<1xi32>
    %160 = stablehlo.concatenate %159, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %161 = stablehlo.dynamic_broadcast_in_dim %3, %160, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %162 = stablehlo.multiply %161, %157 : tensor<?x20x20xf32>
    %163 = stablehlo.sine %162 : tensor<?x20x20xf32>
    %164 = stablehlo.log %163 : tensor<?x20x20xf32>
    %165 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %166 = stablehlo.reshape %165 : (tensor<i32>) -> tensor<1xi32>
    %167 = stablehlo.concatenate %166, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %168 = stablehlo.dynamic_broadcast_in_dim %2, %167, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %169 = stablehlo.subtract %168, %164 : tensor<?x20x20xf32>
    %170 = stablehlo.subtract %169, %151 : tensor<?x20x20xf32>
    %171 = stablehlo.is_finite %164 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %172 = stablehlo.negate %164 : tensor<?x20x20xf32>
    %173 = stablehlo.select %171, %170, %172 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %174 = stablehlo.select %29, %173, %151 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %175 = stablehlo.abs %24 : tensor<?x20x20xf32>
    %176 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %177 = stablehlo.reshape %176 : (tensor<i32>) -> tensor<1xi32>
    %178 = stablehlo.concatenate %177, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %179 = stablehlo.dynamic_broadcast_in_dim %1, %178, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %180 = stablehlo.compare  EQ, %175, %179 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %181 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %182 = stablehlo.reshape %181 : (tensor<i32>) -> tensor<1xi32>
    %183 = stablehlo.concatenate %182, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %184 = stablehlo.dynamic_broadcast_in_dim %1, %183, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %185 = stablehlo.select %180, %184, %174 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %186 = stablehlo.convert %185 : (tensor<?x20x20xf32>) -> tensor<?x20x20xbf16>
    return %186 : tensor<?x20x20xbf16>
  }
}

