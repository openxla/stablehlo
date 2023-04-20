// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %2 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %3 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %4 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %7 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %8 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %9 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %10 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %11 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %12 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %13 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %14 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %15 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %16 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %17 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %18 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %21 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %22 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %23 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %24 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %25 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.convert %arg1 : (tensor<?x20x20xbf16>) -> tensor<?x20x20xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<?x20x20xf32>
    %31 = stablehlo.negate %30 : tensor<?x20x20xf32>
    %32 = stablehlo.abs %29 : tensor<?x20x20xf32>
    %33 = stablehlo.get_dimension_size %29, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %34 = stablehlo.reshape %33 : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.concatenate %34, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %36 = stablehlo.dynamic_broadcast_in_dim %28, %35, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %37 = stablehlo.divide %36, %30 : tensor<?x20x20xf32>
    %38 = stablehlo.exponential %31 : tensor<?x20x20xf32>
    %39 = stablehlo.divide %36, %32 : tensor<?x20x20xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<?x20x20xf32>
    %41 = stablehlo.get_dimension_size %29, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %42 = stablehlo.reshape %41 : (tensor<i32>) -> tensor<1xi32>
    %43 = stablehlo.concatenate %42, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %44 = stablehlo.dynamic_broadcast_in_dim %27, %43, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %45 = stablehlo.compare  LT, %32, %44 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %46 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %47 = stablehlo.reshape %46 : (tensor<i32>) -> tensor<1xi32>
    %48 = stablehlo.concatenate %47, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %49 = stablehlo.dynamic_broadcast_in_dim %26, %48, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %50 = stablehlo.multiply %49, %37 : tensor<?x20x20xf32>
    %51 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %52 = stablehlo.reshape %51 : (tensor<i32>) -> tensor<1xi32>
    %53 = stablehlo.concatenate %52, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %54 = stablehlo.dynamic_broadcast_in_dim %25, %53, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %55 = stablehlo.add %50, %54 : tensor<?x20x20xf32>
    %56 = stablehlo.multiply %55, %37 : tensor<?x20x20xf32>
    %57 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %58 = stablehlo.reshape %57 : (tensor<i32>) -> tensor<1xi32>
    %59 = stablehlo.concatenate %58, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %60 = stablehlo.dynamic_broadcast_in_dim %24, %59, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %61 = stablehlo.add %56, %60 : tensor<?x20x20xf32>
    %62 = stablehlo.multiply %61, %37 : tensor<?x20x20xf32>
    %63 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %64 = stablehlo.reshape %63 : (tensor<i32>) -> tensor<1xi32>
    %65 = stablehlo.concatenate %64, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %66 = stablehlo.dynamic_broadcast_in_dim %23, %65, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %67 = stablehlo.add %62, %66 : tensor<?x20x20xf32>
    %68 = stablehlo.multiply %67, %37 : tensor<?x20x20xf32>
    %69 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %70 = stablehlo.reshape %69 : (tensor<i32>) -> tensor<1xi32>
    %71 = stablehlo.concatenate %70, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %72 = stablehlo.dynamic_broadcast_in_dim %22, %71, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %73 = stablehlo.add %68, %72 : tensor<?x20x20xf32>
    %74 = stablehlo.multiply %73, %37 : tensor<?x20x20xf32>
    %75 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %76 = stablehlo.reshape %75 : (tensor<i32>) -> tensor<1xi32>
    %77 = stablehlo.concatenate %76, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %78 = stablehlo.dynamic_broadcast_in_dim %21, %77, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %79 = stablehlo.add %74, %78 : tensor<?x20x20xf32>
    %80 = stablehlo.multiply %79, %37 : tensor<?x20x20xf32>
    %81 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %82 = stablehlo.reshape %81 : (tensor<i32>) -> tensor<1xi32>
    %83 = stablehlo.concatenate %82, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %84 = stablehlo.dynamic_broadcast_in_dim %20, %83, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %85 = stablehlo.add %80, %84 : tensor<?x20x20xf32>
    %86 = stablehlo.multiply %85, %37 : tensor<?x20x20xf32>
    %87 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %88 = stablehlo.reshape %87 : (tensor<i32>) -> tensor<1xi32>
    %89 = stablehlo.concatenate %88, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %90 = stablehlo.dynamic_broadcast_in_dim %19, %89, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %91 = stablehlo.add %86, %90 : tensor<?x20x20xf32>
    %92 = stablehlo.multiply %91, %37 : tensor<?x20x20xf32>
    %93 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %94 = stablehlo.reshape %93 : (tensor<i32>) -> tensor<1xi32>
    %95 = stablehlo.concatenate %94, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %96 = stablehlo.dynamic_broadcast_in_dim %18, %95, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %97 = stablehlo.add %92, %96 : tensor<?x20x20xf32>
    %98 = stablehlo.multiply %97, %37 : tensor<?x20x20xf32>
    %99 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %100 = stablehlo.reshape %99 : (tensor<i32>) -> tensor<1xi32>
    %101 = stablehlo.concatenate %100, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %102 = stablehlo.dynamic_broadcast_in_dim %17, %101, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %103 = stablehlo.add %98, %102 : tensor<?x20x20xf32>
    %104 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %105 = stablehlo.reshape %104 : (tensor<i32>) -> tensor<1xi32>
    %106 = stablehlo.concatenate %105, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %107 = stablehlo.dynamic_broadcast_in_dim %26, %106, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %108 = stablehlo.multiply %107, %37 : tensor<?x20x20xf32>
    %109 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %110 = stablehlo.reshape %109 : (tensor<i32>) -> tensor<1xi32>
    %111 = stablehlo.concatenate %110, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %112 = stablehlo.dynamic_broadcast_in_dim %16, %111, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %113 = stablehlo.add %108, %112 : tensor<?x20x20xf32>
    %114 = stablehlo.multiply %113, %37 : tensor<?x20x20xf32>
    %115 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %116 = stablehlo.reshape %115 : (tensor<i32>) -> tensor<1xi32>
    %117 = stablehlo.concatenate %116, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %118 = stablehlo.dynamic_broadcast_in_dim %15, %117, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %119 = stablehlo.add %114, %118 : tensor<?x20x20xf32>
    %120 = stablehlo.multiply %119, %37 : tensor<?x20x20xf32>
    %121 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %122 = stablehlo.reshape %121 : (tensor<i32>) -> tensor<1xi32>
    %123 = stablehlo.concatenate %122, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %124 = stablehlo.dynamic_broadcast_in_dim %14, %123, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %125 = stablehlo.add %120, %124 : tensor<?x20x20xf32>
    %126 = stablehlo.multiply %125, %37 : tensor<?x20x20xf32>
    %127 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %128 = stablehlo.reshape %127 : (tensor<i32>) -> tensor<1xi32>
    %129 = stablehlo.concatenate %128, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %130 = stablehlo.dynamic_broadcast_in_dim %13, %129, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %131 = stablehlo.add %126, %130 : tensor<?x20x20xf32>
    %132 = stablehlo.multiply %131, %37 : tensor<?x20x20xf32>
    %133 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %134 = stablehlo.reshape %133 : (tensor<i32>) -> tensor<1xi32>
    %135 = stablehlo.concatenate %134, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %136 = stablehlo.dynamic_broadcast_in_dim %12, %135, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %137 = stablehlo.add %132, %136 : tensor<?x20x20xf32>
    %138 = stablehlo.multiply %137, %37 : tensor<?x20x20xf32>
    %139 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %140 = stablehlo.reshape %139 : (tensor<i32>) -> tensor<1xi32>
    %141 = stablehlo.concatenate %140, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %142 = stablehlo.dynamic_broadcast_in_dim %11, %141, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %143 = stablehlo.add %138, %142 : tensor<?x20x20xf32>
    %144 = stablehlo.multiply %143, %37 : tensor<?x20x20xf32>
    %145 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %146 = stablehlo.reshape %145 : (tensor<i32>) -> tensor<1xi32>
    %147 = stablehlo.concatenate %146, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %148 = stablehlo.dynamic_broadcast_in_dim %10, %147, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %149 = stablehlo.add %144, %148 : tensor<?x20x20xf32>
    %150 = stablehlo.multiply %149, %37 : tensor<?x20x20xf32>
    %151 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %152 = stablehlo.reshape %151 : (tensor<i32>) -> tensor<1xi32>
    %153 = stablehlo.concatenate %152, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %154 = stablehlo.dynamic_broadcast_in_dim %9, %153, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %155 = stablehlo.add %150, %154 : tensor<?x20x20xf32>
    %156 = stablehlo.select %45, %103, %155 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %157 = stablehlo.multiply %40, %156 : tensor<?x20x20xf32>
    %158 = stablehlo.get_dimension_size %29, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %159 = stablehlo.reshape %158 : (tensor<i32>) -> tensor<1xi32>
    %160 = stablehlo.concatenate %159, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %161 = stablehlo.dynamic_broadcast_in_dim %8, %160, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %162 = stablehlo.compare  LT, %31, %161 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %163 = stablehlo.get_dimension_size %29, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %164 = stablehlo.reshape %163 : (tensor<i32>) -> tensor<1xi32>
    %165 = stablehlo.concatenate %164, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %166 = stablehlo.dynamic_broadcast_in_dim %26, %165, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %167 = stablehlo.select %162, %166, %157 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %168 = stablehlo.compare  LT, %29, %166 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %169 = stablehlo.subtract %44, %167 : tensor<?x20x20xf32>
    %170 = stablehlo.select %168, %169, %167 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %171 = stablehlo.get_dimension_size %29, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %172 = stablehlo.reshape %171 : (tensor<i32>) -> tensor<1xi32>
    %173 = stablehlo.concatenate %172, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %174 = stablehlo.dynamic_broadcast_in_dim %28, %173, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %175 = stablehlo.multiply %29, %29 : tensor<?x20x20xf32>
    %176 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %177 = stablehlo.reshape %176 : (tensor<i32>) -> tensor<1xi32>
    %178 = stablehlo.concatenate %177, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %179 = stablehlo.dynamic_broadcast_in_dim %26, %178, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %180 = stablehlo.multiply %179, %175 : tensor<?x20x20xf32>
    %181 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %182 = stablehlo.reshape %181 : (tensor<i32>) -> tensor<1xi32>
    %183 = stablehlo.concatenate %182, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %184 = stablehlo.dynamic_broadcast_in_dim %7, %183, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %185 = stablehlo.add %180, %184 : tensor<?x20x20xf32>
    %186 = stablehlo.multiply %185, %175 : tensor<?x20x20xf32>
    %187 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %188 = stablehlo.reshape %187 : (tensor<i32>) -> tensor<1xi32>
    %189 = stablehlo.concatenate %188, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %190 = stablehlo.dynamic_broadcast_in_dim %6, %189, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %191 = stablehlo.add %186, %190 : tensor<?x20x20xf32>
    %192 = stablehlo.multiply %191, %175 : tensor<?x20x20xf32>
    %193 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %194 = stablehlo.reshape %193 : (tensor<i32>) -> tensor<1xi32>
    %195 = stablehlo.concatenate %194, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %196 = stablehlo.dynamic_broadcast_in_dim %5, %195, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %197 = stablehlo.add %192, %196 : tensor<?x20x20xf32>
    %198 = stablehlo.multiply %197, %175 : tensor<?x20x20xf32>
    %199 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %200 = stablehlo.reshape %199 : (tensor<i32>) -> tensor<1xi32>
    %201 = stablehlo.concatenate %200, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %202 = stablehlo.dynamic_broadcast_in_dim %4, %201, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %203 = stablehlo.add %198, %202 : tensor<?x20x20xf32>
    %204 = stablehlo.multiply %203, %175 : tensor<?x20x20xf32>
    %205 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %206 = stablehlo.reshape %205 : (tensor<i32>) -> tensor<1xi32>
    %207 = stablehlo.concatenate %206, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %208 = stablehlo.dynamic_broadcast_in_dim %3, %207, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %209 = stablehlo.add %204, %208 : tensor<?x20x20xf32>
    %210 = stablehlo.multiply %209, %175 : tensor<?x20x20xf32>
    %211 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %212 = stablehlo.reshape %211 : (tensor<i32>) -> tensor<1xi32>
    %213 = stablehlo.concatenate %212, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %214 = stablehlo.dynamic_broadcast_in_dim %2, %213, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %215 = stablehlo.add %210, %214 : tensor<?x20x20xf32>
    %216 = stablehlo.multiply %215, %175 : tensor<?x20x20xf32>
    %217 = stablehlo.get_dimension_size %175, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %218 = stablehlo.reshape %217 : (tensor<i32>) -> tensor<1xi32>
    %219 = stablehlo.concatenate %218, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %220 = stablehlo.dynamic_broadcast_in_dim %1, %219, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %221 = stablehlo.add %216, %220 : tensor<?x20x20xf32>
    %222 = stablehlo.multiply %29, %221 : tensor<?x20x20xf32>
    %223 = stablehlo.subtract %174, %222 : tensor<?x20x20xf32>
    %224 = stablehlo.abs %29 : tensor<?x20x20xf32>
    %225 = stablehlo.compare  LT, %224, %174 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %226 = stablehlo.select %225, %223, %170 : tensor<?x20x20xi1>, tensor<?x20x20xf32>
    %227 = stablehlo.convert %226 : (tensor<?x20x20xf32>) -> tensor<?x20x20xbf16>
    return %227 : tensor<?x20x20xbf16>
  }
}

