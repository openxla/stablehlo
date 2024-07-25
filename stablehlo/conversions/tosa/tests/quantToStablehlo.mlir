// RUN: stablehlo-opt %s --stablehlo-quant-legalize-to-tosa-rescale --tosa-rescale-legalize-to-stablehlo --split-input-file -verify-each | FileCheck %s

// -----
// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  // CHECK-DAG: %[[V_c:.+]] = stablehlo.constant dense<127> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_0:.+]] = stablehlo.constant dense<-128> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_1:.+]] = stablehlo.constant dense<50> : tensor<2x2xi8>
  // CHECK-DAG: %[[V_c_2:.+]] = stablehlo.constant dense<11> : tensor<2x2xi8>
  // CHECK-DAG: %[[V_c_3:.+]] = stablehlo.constant dense<1073741824> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_4:.+]] = stablehlo.constant dense<2147483647> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_5:.+]] = stablehlo.constant dense<-2147483648> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_6:.+]] = stablehlo.constant dense<1> : tensor<2x2xi64>
  // CHECK-DAG: %[[V_c_7:.+]] = stablehlo.constant dense<0> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_8:.+]] = stablehlo.constant dense<-1> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_9:.+]] = stablehlo.constant dense<13> : tensor<2x2xi8>
  // CHECK-DAG: %[[V_c_10:.+]] = stablehlo.constant dense<1431655765> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_0:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<2x2x!quant.uniform<i8:f32, 2.500000e-02:-1>>) -> tensor<2x2xi8>
  // CHECK-DAG: %[[V_1:.+]] = stablehlo.convert %[[V_c_10]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_2:.+]] = stablehlo.convert %[[V_c_9]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_3:.+]] = stablehlo.convert %[[V_c_8]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_4:.+]] = stablehlo.convert %[[V_c_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_5:.+]] = stablehlo.convert %[[V_0]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_6:.+]] = stablehlo.convert %[[V_c_5]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_7:.+]] = stablehlo.convert %[[V_c_4]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_8:.+]] = stablehlo.subtract %[[V_5]], %[[V_3]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_9:.+]] = stablehlo.subtract %[[V_2]], %[[V_c_6]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_10:.+]] = stablehlo.shift_left %[[V_c_6]], %[[V_9]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_11:.+]] = stablehlo.multiply %[[V_8]], %[[V_1]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_12:.+]] = stablehlo.add %[[V_11]], %[[V_10]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_13:.+]] = stablehlo.shift_right_arithmetic %[[V_12]], %[[V_2]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_14:.+]] = stablehlo.add %[[V_13]], %[[V_4]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_15:.+]] = stablehlo.clamp %[[V_6]], %[[V_14]], %[[V_7]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_16:.+]] = stablehlo.convert %[[V_15]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK-DAG: %[[V_17:.+]] = stablehlo.bitcast_convert %arg1 : (tensor<2x2x!quant.uniform<i8:f32, 0.074999999999999997:-1>>) -> tensor<2x2xi8>
  // CHECK-DAG: %[[V_18:.+]] = stablehlo.convert %[[V_c_3]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_19:.+]] = stablehlo.convert %[[V_c_2]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_20:.+]] = stablehlo.convert %[[V_c_8]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_21:.+]] = stablehlo.convert %[[V_c_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_22:.+]] = stablehlo.convert %[[V_17]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_23:.+]] = stablehlo.convert %[[V_c_5]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_24:.+]] = stablehlo.convert %[[V_c_4]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_25:.+]] = stablehlo.subtract %[[V_22]], %[[V_20]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_26:.+]] = stablehlo.subtract %[[V_19]], %[[V_c_6]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_27:.+]] = stablehlo.shift_left %[[V_c_6]], %[[V_26]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_28:.+]] = stablehlo.multiply %[[V_25]], %[[V_18]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_29:.+]] = stablehlo.add %[[V_28]], %[[V_27]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_30:.+]] = stablehlo.shift_right_arithmetic %[[V_29]], %[[V_19]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_31:.+]] = stablehlo.add %[[V_30]], %[[V_21]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_32:.+]] = stablehlo.clamp %[[V_23]], %[[V_31]], %[[V_24]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_33:.+]] = stablehlo.convert %[[V_32]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK-DAG: %[[V_34:.+]] = stablehlo.add %[[V_16]], %[[V_33]] : tensor<2x2xi32>
  // CHECK-DAG: %[[V_35:.+]] = stablehlo.convert %[[V_c_3]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_36:.+]] = stablehlo.convert %[[V_c_1]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_37:.+]] = stablehlo.convert %[[V_c_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_38:.+]] = stablehlo.convert %[[V_c_8]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_39:.+]] = stablehlo.convert %[[V_34]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_40:.+]] = stablehlo.convert %[[V_c_0]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_41:.+]] = stablehlo.convert %[[V_c]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_42:.+]] = stablehlo.subtract %[[V_39]], %[[V_37]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_43:.+]] = stablehlo.subtract %[[V_36]], %[[V_c_6]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_44:.+]] = stablehlo.shift_left %[[V_c_6]], %[[V_43]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_45:.+]] = stablehlo.multiply %[[V_42]], %[[V_35]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_46:.+]] = stablehlo.add %[[V_45]], %[[V_44]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_47:.+]] = stablehlo.shift_right_arithmetic %[[V_46]], %[[V_36]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_48:.+]] = stablehlo.add %[[V_47]], %[[V_38]] : tensor<2x2xi64>
  // CHECK: %[[V_49:.+]] = stablehlo.clamp %[[V_40]], %[[V_48]], %[[V_41]] : tensor<2x2xi64>
  // CHECK: %[[V_50:.+]] = stablehlo.convert %[[V_49]] : (tensor<2x2xi64>) -> tensor<2x2xi8>
  // CHECK: %[[V_51:.+]] = stablehlo.bitcast_convert %[[V_50]] : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  // CHECK: return %[[V_51]]

  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @sub
func.func @sub(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  // CHECK-DAG: %[[V_c:.+]] = stablehlo.constant dense<127> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_0:.+]] = stablehlo.constant dense<-128> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_1:.+]] = stablehlo.constant dense<50> : tensor<2x2xi8>
  // CHECK-DAG: %[[V_c_2:.+]] = stablehlo.constant dense<11> : tensor<2x2xi8>
  // CHECK-DAG: %[[V_c_3:.+]] = stablehlo.constant dense<1073741824> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_4:.+]] = stablehlo.constant dense<2147483647> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_5:.+]] = stablehlo.constant dense<-2147483648> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_6:.+]] = stablehlo.constant dense<1> : tensor<2x2xi64>
  // CHECK-DAG: %[[V_c_7:.+]] = stablehlo.constant dense<0> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_8:.+]] = stablehlo.constant dense<-1> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_c_9:.+]] = stablehlo.constant dense<13> : tensor<2x2xi8>
  // CHECK-DAG: %[[V_c_10:.+]] = stablehlo.constant dense<1431655765> : tensor<2x2xi32>
  // CHECK-DAG: %[[V_0:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<2x2x!quant.uniform<i8:f32, 2.500000e-02:-1>>) -> tensor<2x2xi8>
  // CHECK-DAG: %[[V_1:.+]] = stablehlo.convert %[[V_c_10]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_2:.+]] = stablehlo.convert %[[V_c_9]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_3:.+]] = stablehlo.convert %[[V_c_8]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_4:.+]] = stablehlo.convert %[[V_c_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_5:.+]] = stablehlo.convert %[[V_0]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_6:.+]] = stablehlo.convert %[[V_c_5]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_7:.+]] = stablehlo.convert %[[V_c_4]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_8:.+]] = stablehlo.subtract %[[V_5]], %[[V_3]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_9:.+]] = stablehlo.subtract %[[V_2]], %[[V_c_6]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_10:.+]] = stablehlo.shift_left %[[V_c_6]], %[[V_9]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_11:.+]] = stablehlo.multiply %[[V_8]], %[[V_1]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_12:.+]] = stablehlo.add %[[V_11]], %[[V_10]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_13:.+]] = stablehlo.shift_right_arithmetic %[[V_12]], %[[V_2]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_14:.+]] = stablehlo.add %[[V_13]], %[[V_4]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_15:.+]] = stablehlo.clamp %[[V_6]], %[[V_14]], %[[V_7]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_16:.+]] = stablehlo.convert %[[V_15]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK-DAG: %[[V_17:.+]] = stablehlo.bitcast_convert %arg1 : (tensor<2x2x!quant.uniform<i8:f32, 0.074999999999999997:-1>>) -> tensor<2x2xi8>
  // CHECK-DAG: %[[V_18:.+]] = stablehlo.convert %[[V_c_3]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_19:.+]] = stablehlo.convert %[[V_c_2]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_20:.+]] = stablehlo.convert %[[V_c_8]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_21:.+]] = stablehlo.convert %[[V_c_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_22:.+]] = stablehlo.convert %[[V_17]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_23:.+]] = stablehlo.convert %[[V_c_5]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_24:.+]] = stablehlo.convert %[[V_c_4]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_25:.+]] = stablehlo.subtract %[[V_22]], %[[V_20]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_26:.+]] = stablehlo.subtract %[[V_19]], %[[V_c_6]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_27:.+]] = stablehlo.shift_left %[[V_c_6]], %[[V_26]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_28:.+]] = stablehlo.multiply %[[V_25]], %[[V_18]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_29:.+]] = stablehlo.add %[[V_28]], %[[V_27]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_30:.+]] = stablehlo.shift_right_arithmetic %[[V_29]], %[[V_19]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_31:.+]] = stablehlo.add %[[V_30]], %[[V_21]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_32:.+]] = stablehlo.clamp %[[V_23]], %[[V_31]], %[[V_24]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_33:.+]] = stablehlo.convert %[[V_32]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK-DAG: %[[V_34:.+]] = stablehlo.subtract %[[V_16]], %[[V_33]] : tensor<2x2xi32>
  // CHECK-DAG: %[[V_35:.+]] = stablehlo.convert %[[V_c_3]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_36:.+]] = stablehlo.convert %[[V_c_1]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_37:.+]] = stablehlo.convert %[[V_c_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_38:.+]] = stablehlo.convert %[[V_c_8]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_39:.+]] = stablehlo.convert %[[V_34]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_40:.+]] = stablehlo.convert %[[V_c_0]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_41:.+]] = stablehlo.convert %[[V_c]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[V_42:.+]] = stablehlo.subtract %[[V_39]], %[[V_37]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_43:.+]] = stablehlo.subtract %[[V_36]], %[[V_c_6]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_44:.+]] = stablehlo.shift_left %[[V_c_6]], %[[V_43]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_45:.+]] = stablehlo.multiply %[[V_42]], %[[V_35]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_46:.+]] = stablehlo.add %[[V_45]], %[[V_44]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_47:.+]] = stablehlo.shift_right_arithmetic %[[V_46]], %[[V_36]] : tensor<2x2xi64>
  // CHECK-DAG: %[[V_48:.+]] = stablehlo.add %[[V_47]], %[[V_38]] : tensor<2x2xi64>
  // CHECK: %[[V_49:.+]] = stablehlo.clamp %[[V_40]], %[[V_48]], %[[V_41]] : tensor<2x2xi64>
  // CHECK: %[[V_50:.+]] = stablehlo.convert %[[V_49]] : (tensor<2x2xi64>) -> tensor<2x2xi8>
  // CHECK: %[[V_51:.+]] = stablehlo.bitcast_convert %[[V_50]] : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  // CHECK: return %[[V_51]]

  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}
