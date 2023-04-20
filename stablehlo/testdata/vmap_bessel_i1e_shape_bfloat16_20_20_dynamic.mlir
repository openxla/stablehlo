// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<0.778576254> : tensor<f32>
    %2 = stablehlo.constant dense<-0.00976109784> : tensor<f32>
    %3 = stablehlo.constant dense<-1.10588939E-4> : tensor<f32>
    %4 = stablehlo.constant dense<-3.88256467E-6> : tensor<f32>
    %5 = stablehlo.constant dense<-2.51223611E-7> : tensor<f32>
    %6 = stablehlo.constant dense<-2.63146891E-8> : tensor<f32>
    %7 = stablehlo.constant dense<-3.83538046E-9> : tensor<f32>
    %8 = stablehlo.constant dense<0.252587199> : tensor<f32>
    %9 = stablehlo.constant dense<-0.176416516> : tensor<f32>
    %10 = stablehlo.constant dense<0.102643661> : tensor<f32>
    %11 = stablehlo.constant dense<-0.0529459827> : tensor<f32>
    %12 = stablehlo.constant dense<0.0247264486> : tensor<f32>
    %13 = stablehlo.constant dense<-0.0105640851> : tensor<f32>
    %14 = stablehlo.constant dense<0.0041564228> : tensor<f32>
    %15 = stablehlo.constant dense<-0.00151357241> : tensor<f32>
    %16 = stablehlo.constant dense<5.122860e-04> : tensor<f32>
    %17 = stablehlo.constant dense<-1.61760821E-4> : tensor<f32>
    %18 = stablehlo.constant dense<4.78156508E-5> : tensor<f32>
    %19 = stablehlo.constant dense<-1.32731639E-5> : tensor<f32>
    %20 = stablehlo.constant dense<3.47025139E-6> : tensor<f32>
    %21 = stablehlo.constant dense<-8.568720e-07> : tensor<f32>
    %22 = stablehlo.constant dense<2.00329481E-7> : tensor<f32>
    %23 = stablehlo.constant dense<-4.44505908E-8> : tensor<f32>
    %24 = stablehlo.constant dense<9.38153732E-9> : tensor<f32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %27 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %28 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %29 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %30 = stablehlo.convert %arg1 : (tensor<?x20x20xbf16>) -> tensor<?x20x20xf32>
    %31 = stablehlo.abs %30 : tensor<?x20x20xf32>
    %32 = stablehlo.get_dimension_size %30, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %33 = stablehlo.reshape %32 : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.concatenate %33, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %35 = stablehlo.dynamic_broadcast_in_dim %29, %34, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %36 = stablehlo.get_dimension_size %30, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %37 = stablehlo.reshape %36 : (tensor<i32>) -> tensor<1xi32>
    %38 = stablehlo.concatenate %37, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %39 = stablehlo.dynamic_broadcast_in_dim %28, %38, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %40 = stablehlo.get_dimension_size %30, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %41 = stablehlo.reshape %40 : (tensor<i32>) -> tensor<1xi32>
    %42 = stablehlo.concatenate %41, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %43 = stablehlo.dynamic_broadcast_in_dim %27, %42, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %44 = stablehlo.get_dimension_size %30, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %45 = stablehlo.reshape %44 : (tensor<i32>) -> tensor<1xi32>
    %46 = stablehlo.concatenate %45, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %47 = stablehlo.dynamic_broadcast_in_dim %26, %46, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %48 = stablehlo.multiply %35, %31 : tensor<?x20x20xf32>
    %49 = stablehlo.subtract %48, %39 : tensor<?x20x20xf32>
    %50 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %51 = stablehlo.reshape %50 : (tensor<i32>) -> tensor<1xi32>
    %52 = stablehlo.concatenate %51, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %53 = stablehlo.dynamic_broadcast_in_dim %25, %52, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %54 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %55 = stablehlo.reshape %54 : (tensor<i32>) -> tensor<1xi32>
    %56 = stablehlo.concatenate %55, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %57 = stablehlo.dynamic_broadcast_in_dim %25, %56, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %58 = stablehlo.multiply %49, %53 : tensor<?x20x20xf32>
    %59 = stablehlo.subtract %58, %57 : tensor<?x20x20xf32>
    %60 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %61 = stablehlo.reshape %60 : (tensor<i32>) -> tensor<1xi32>
    %62 = stablehlo.concatenate %61, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %63 = stablehlo.dynamic_broadcast_in_dim %24, %62, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %64 = stablehlo.add %59, %63 : tensor<?x20x20xf32>
    %65 = stablehlo.multiply %49, %64 : tensor<?x20x20xf32>
    %66 = stablehlo.subtract %65, %53 : tensor<?x20x20xf32>
    %67 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %68 = stablehlo.reshape %67 : (tensor<i32>) -> tensor<1xi32>
    %69 = stablehlo.concatenate %68, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %70 = stablehlo.dynamic_broadcast_in_dim %23, %69, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %71 = stablehlo.add %66, %70 : tensor<?x20x20xf32>
    %72 = stablehlo.multiply %49, %71 : tensor<?x20x20xf32>
    %73 = stablehlo.subtract %72, %64 : tensor<?x20x20xf32>
    %74 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %75 = stablehlo.reshape %74 : (tensor<i32>) -> tensor<1xi32>
    %76 = stablehlo.concatenate %75, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %77 = stablehlo.dynamic_broadcast_in_dim %22, %76, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %78 = stablehlo.add %73, %77 : tensor<?x20x20xf32>
    %79 = stablehlo.multiply %49, %78 : tensor<?x20x20xf32>
    %80 = stablehlo.subtract %79, %71 : tensor<?x20x20xf32>
    %81 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %82 = stablehlo.reshape %81 : (tensor<i32>) -> tensor<1xi32>
    %83 = stablehlo.concatenate %82, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %84 = stablehlo.dynamic_broadcast_in_dim %21, %83, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %85 = stablehlo.add %80, %84 : tensor<?x20x20xf32>
    %86 = stablehlo.multiply %49, %85 : tensor<?x20x20xf32>
    %87 = stablehlo.subtract %86, %78 : tensor<?x20x20xf32>
    %88 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %89 = stablehlo.reshape %88 : (tensor<i32>) -> tensor<1xi32>
    %90 = stablehlo.concatenate %89, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %91 = stablehlo.dynamic_broadcast_in_dim %20, %90, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %92 = stablehlo.add %87, %91 : tensor<?x20x20xf32>
    %93 = stablehlo.multiply %49, %92 : tensor<?x20x20xf32>
    %94 = stablehlo.subtract %93, %85 : tensor<?x20x20xf32>
    %95 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %96 = stablehlo.reshape %95 : (tensor<i32>) -> tensor<1xi32>
    %97 = stablehlo.concatenate %96, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %98 = stablehlo.dynamic_broadcast_in_dim %19, %97, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %99 = stablehlo.add %94, %98 : tensor<?x20x20xf32>
    %100 = stablehlo.multiply %49, %99 : tensor<?x20x20xf32>
    %101 = stablehlo.subtract %100, %92 : tensor<?x20x20xf32>
    %102 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %103 = stablehlo.reshape %102 : (tensor<i32>) -> tensor<1xi32>
    %104 = stablehlo.concatenate %103, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %105 = stablehlo.dynamic_broadcast_in_dim %18, %104, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %106 = stablehlo.add %101, %105 : tensor<?x20x20xf32>
    %107 = stablehlo.multiply %49, %106 : tensor<?x20x20xf32>
    %108 = stablehlo.subtract %107, %99 : tensor<?x20x20xf32>
    %109 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %110 = stablehlo.reshape %109 : (tensor<i32>) -> tensor<1xi32>
    %111 = stablehlo.concatenate %110, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %112 = stablehlo.dynamic_broadcast_in_dim %17, %111, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %113 = stablehlo.add %108, %112 : tensor<?x20x20xf32>
    %114 = stablehlo.multiply %49, %113 : tensor<?x20x20xf32>
    %115 = stablehlo.subtract %114, %106 : tensor<?x20x20xf32>
    %116 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %117 = stablehlo.reshape %116 : (tensor<i32>) -> tensor<1xi32>
    %118 = stablehlo.concatenate %117, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %119 = stablehlo.dynamic_broadcast_in_dim %16, %118, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %120 = stablehlo.add %115, %119 : tensor<?x20x20xf32>
    %121 = stablehlo.multiply %49, %120 : tensor<?x20x20xf32>
    %122 = stablehlo.subtract %121, %113 : tensor<?x20x20xf32>
    %123 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %124 = stablehlo.reshape %123 : (tensor<i32>) -> tensor<1xi32>
    %125 = stablehlo.concatenate %124, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %126 = stablehlo.dynamic_broadcast_in_dim %15, %125, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %127 = stablehlo.add %122, %126 : tensor<?x20x20xf32>
    %128 = stablehlo.multiply %49, %127 : tensor<?x20x20xf32>
    %129 = stablehlo.subtract %128, %120 : tensor<?x20x20xf32>
    %130 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %131 = stablehlo.reshape %130 : (tensor<i32>) -> tensor<1xi32>
    %132 = stablehlo.concatenate %131, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %133 = stablehlo.dynamic_broadcast_in_dim %14, %132, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %134 = stablehlo.add %129, %133 : tensor<?x20x20xf32>
    %135 = stablehlo.multiply %49, %134 : tensor<?x20x20xf32>
    %136 = stablehlo.subtract %135, %127 : tensor<?x20x20xf32>
    %137 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %138 = stablehlo.reshape %137 : (tensor<i32>) -> tensor<1xi32>
    %139 = stablehlo.concatenate %138, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %140 = stablehlo.dynamic_broadcast_in_dim %13, %139, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %141 = stablehlo.add %136, %140 : tensor<?x20x20xf32>
    %142 = stablehlo.multiply %49, %141 : tensor<?x20x20xf32>
    %143 = stablehlo.subtract %142, %134 : tensor<?x20x20xf32>
    %144 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %145 = stablehlo.reshape %144 : (tensor<i32>) -> tensor<1xi32>
    %146 = stablehlo.concatenate %145, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %147 = stablehlo.dynamic_broadcast_in_dim %12, %146, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %148 = stablehlo.add %143, %147 : tensor<?x20x20xf32>
    %149 = stablehlo.multiply %49, %148 : tensor<?x20x20xf32>
    %150 = stablehlo.subtract %149, %141 : tensor<?x20x20xf32>
    %151 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %152 = stablehlo.reshape %151 : (tensor<i32>) -> tensor<1xi32>
    %153 = stablehlo.concatenate %152, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %154 = stablehlo.dynamic_broadcast_in_dim %11, %153, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %155 = stablehlo.add %150, %154 : tensor<?x20x20xf32>
    %156 = stablehlo.multiply %49, %155 : tensor<?x20x20xf32>
    %157 = stablehlo.subtract %156, %148 : tensor<?x20x20xf32>
    %158 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %159 = stablehlo.reshape %158 : (tensor<i32>) -> tensor<1xi32>
    %160 = stablehlo.concatenate %159, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %161 = stablehlo.dynamic_broadcast_in_dim %10, %160, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %162 = stablehlo.add %157, %161 : tensor<?x20x20xf32>
    %163 = stablehlo.multiply %49, %162 : tensor<?x20x20xf32>
    %164 = stablehlo.subtract %163, %155 : tensor<?x20x20xf32>
    %165 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %166 = stablehlo.reshape %165 : (tensor<i32>) -> tensor<1xi32>
    %167 = stablehlo.concatenate %166, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %168 = stablehlo.dynamic_broadcast_in_dim %9, %167, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %169 = stablehlo.add %164, %168 : tensor<?x20x20xf32>
    %170 = stablehlo.multiply %49, %169 : tensor<?x20x20xf32>
    %171 = stablehlo.subtract %170, %162 : tensor<?x20x20xf32>
    %172 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %173 = stablehlo.reshape %172 : (tensor<i32>) -> tensor<1xi32>
    %174 = stablehlo.concatenate %173, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %175 = stablehlo.dynamic_broadcast_in_dim %8, %174, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %176 = stablehlo.add %171, %175 : tensor<?x20x20xf32>
    %177 = stablehlo.subtract %176, %162 : tensor<?x20x20xf32>
    %178 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %179 = stablehlo.reshape %178 : (tensor<i32>) -> tensor<1xi32>
    %180 = stablehlo.concatenate %179, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %181 = stablehlo.dynamic_broadcast_in_dim %29, %180, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %182 = stablehlo.multiply %177, %181 : tensor<?x20x20xf32>
    %183 = stablehlo.multiply %31, %182 : tensor<?x20x20xf32>
    %184 = stablehlo.divide %43, %31 : tensor<?x20x20xf32>
    %185 = stablehlo.subtract %184, %39 : tensor<?x20x20xf32>
    %186 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %187 = stablehlo.reshape %186 : (tensor<i32>) -> tensor<1xi32>
    %188 = stablehlo.concatenate %187, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %189 = stablehlo.dynamic_broadcast_in_dim %25, %188, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %190 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %191 = stablehlo.reshape %190 : (tensor<i32>) -> tensor<1xi32>
    %192 = stablehlo.concatenate %191, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %193 = stablehlo.dynamic_broadcast_in_dim %25, %192, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %194 = stablehlo.multiply %185, %189 : tensor<?x20x20xf32>
    %195 = stablehlo.subtract %194, %193 : tensor<?x20x20xf32>
    %196 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %197 = stablehlo.reshape %196 : (tensor<i32>) -> tensor<1xi32>
    %198 = stablehlo.concatenate %197, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %199 = stablehlo.dynamic_broadcast_in_dim %7, %198, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %200 = stablehlo.add %195, %199 : tensor<?x20x20xf32>
    %201 = stablehlo.multiply %185, %200 : tensor<?x20x20xf32>
    %202 = stablehlo.subtract %201, %189 : tensor<?x20x20xf32>
    %203 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %204 = stablehlo.reshape %203 : (tensor<i32>) -> tensor<1xi32>
    %205 = stablehlo.concatenate %204, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %206 = stablehlo.dynamic_broadcast_in_dim %6, %205, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %207 = stablehlo.add %202, %206 : tensor<?x20x20xf32>
    %208 = stablehlo.multiply %185, %207 : tensor<?x20x20xf32>
    %209 = stablehlo.subtract %208, %200 : tensor<?x20x20xf32>
    %210 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %211 = stablehlo.reshape %210 : (tensor<i32>) -> tensor<1xi32>
    %212 = stablehlo.concatenate %211, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %213 = stablehlo.dynamic_broadcast_in_dim %5, %212, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %214 = stablehlo.add %209, %213 : tensor<?x20x20xf32>
    %215 = stablehlo.multiply %185, %214 : tensor<?x20x20xf32>
    %216 = stablehlo.subtract %215, %207 : tensor<?x20x20xf32>
    %217 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %218 = stablehlo.reshape %217 : (tensor<i32>) -> tensor<1xi32>
    %219 = stablehlo.concatenate %218, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %220 = stablehlo.dynamic_broadcast_in_dim %4, %219, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %221 = stablehlo.add %216, %220 : tensor<?x20x20xf32>
    %222 = stablehlo.multiply %185, %221 : tensor<?x20x20xf32>
    %223 = stablehlo.subtract %222, %214 : tensor<?x20x20xf32>
    %224 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %225 = stablehlo.reshape %224 : (tensor<i32>) -> tensor<1xi32>
    %226 = stablehlo.concatenate %225, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %227 = stablehlo.dynamic_broadcast_in_dim %3, %226, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %228 = stablehlo.add %223, %227 : tensor<?x20x20xf32>
    %229 = stablehlo.multiply %185, %228 : tensor<?x20x20xf32>
    %230 = stablehlo.subtract %229, %221 : tensor<?x20x20xf32>
    %231 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %232 = stablehlo.reshape %231 : (tensor<i32>) -> tensor<1xi32>
    %233 = stablehlo.concatenate %232, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %234 = stablehlo.dynamic_broadcast_in_dim %2, %233, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %235 = stablehlo.add %230, %234 : tensor<?x20x20xf32>
    %236 = stablehlo.multiply %185, %235 : tensor<?x20x20xf32>
    %237 = stablehlo.subtract %236, %228 : tensor<?x20x20xf32>
    %238 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %239 = stablehlo.reshape %238 : (tensor<i32>) -> tensor<1xi32>
    %240 = stablehlo.concatenate %239, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %241 = stablehlo.dynamic_broadcast_in_dim %1, %240, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %242 = stablehlo.add %237, %241 : tensor<?x20x20xf32>
    %243 = stablehlo.subtract %242, %228 : tensor<?x20x20xf32>
    %244 = stablehlo.get_dimension_size %185, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %245 = stablehlo.reshape %244 : (tensor<i32>) -> tensor<1xi32>
    %246 = stablehlo.concatenate %245, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %247 = stablehlo.dynamic_broadcast_in_dim %29, %246, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %248 = stablehlo.multiply %243, %247 : tensor<?x20x20xf32>
    %249 = stablehlo.sqrt %31 : tensor<?x20x20xf32>
    %250 = stablehlo.divide %248, %249 : tensor<?x20x20xf32>
    %251 = stablehlo.compare  LE, %31, %47 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %252 = stablehlo.select %251, %183, %250 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %253 = stablehlo.sign %30 : tensor<?x20x20xf32>
    %254 = stablehlo.multiply %253, %252 : tensor<?x20x20xf32>
    %255 = stablehlo.convert %254 : (tensor<?x20x20xf32>) -> tensor<?x20x20xbf16>
    return %255 : tensor<?x20x20xbf16>
  }
}

