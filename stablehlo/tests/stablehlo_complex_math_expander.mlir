// RUN: stablehlo-opt --stablehlo-complex-math-expander --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @log_plus_one_complex_f32(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<6.500000e+01> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<9.99999968E+37> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<0x4D000000> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<2.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0.00999999977> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.abs %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.abs %[[VAL_13]] : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.maximum %[[VAL_12]], %[[VAL_14]] : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.sqrt %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.multiply %[[VAL_16]], %[[VAL_9]] : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.compare  GT, %[[VAL_15]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_19:.*]] = stablehlo.log %[[VAL_15]] : tensor<f32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.minimum %[[VAL_12]], %[[VAL_14]] : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.compare  EQ, %[[VAL_20]], %[[VAL_15]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_22:.*]] = stablehlo.divide %[[VAL_20]], %[[VAL_15]] : tensor<f32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_22]] : tensor<f32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.select %[[VAL_21]], %[[VAL_7]], %[[VAL_23]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.log_plus_one %[[VAL_24]] : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.multiply %[[VAL_8]], %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.add %[[VAL_19]], %[[VAL_26]] : tensor<f32>
// CHECK:           %[[VAL_28:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_7]] : tensor<f32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.abs %[[VAL_28]] : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = stablehlo.add %[[VAL_29]], %[[VAL_14]] : tensor<f32>
// CHECK:           %[[VAL_31:.*]] = stablehlo.compare  LT, %[[VAL_30]], %[[VAL_6]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_32:.*]] = stablehlo.multiply %[[VAL_28]], %[[VAL_28]] : tensor<f32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_13]] : tensor<f32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.add %[[VAL_32]], %[[VAL_33]] : tensor<f32>
// CHECK:           %[[VAL_35:.*]] = stablehlo.log %[[VAL_34]] : tensor<f32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.multiply %[[VAL_8]], %[[VAL_35]] : tensor<f32>
// CHECK:           %[[VAL_37:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_38:.*]] = stablehlo.add %[[VAL_37]], %[[VAL_33]] : tensor<f32>
// CHECK:           %[[VAL_39:.*]] = stablehlo.multiply %[[VAL_11]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_40:.*]] = stablehlo.add %[[VAL_38]], %[[VAL_39]] : tensor<f32>
// CHECK:           %[[VAL_41:.*]] = stablehlo.negate %[[VAL_33]] : tensor<f32>
// CHECK:           %[[VAL_42:.*]] = stablehlo.compare  GT, %[[VAL_10]], %[[VAL_5]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_43:.*]] = stablehlo.compare  GT, %[[VAL_10]], %[[VAL_3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_44:.*]] = stablehlo.select %[[VAL_43]], %[[VAL_2]], %[[VAL_1]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.select %[[VAL_42]], %[[VAL_4]], %[[VAL_44]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_46:.*]] = stablehlo.multiply %[[VAL_45]], %[[VAL_13]] : tensor<f32>
// CHECK:           %[[VAL_47:.*]] = stablehlo.subtract %[[VAL_13]], %[[VAL_46]] : tensor<f32>
// CHECK:           %[[VAL_48:.*]] = stablehlo.add %[[VAL_46]], %[[VAL_47]] : tensor<f32>
// CHECK:           %[[VAL_49:.*]] = stablehlo.multiply %[[VAL_48]], %[[VAL_48]] : tensor<f32>
// CHECK:           %[[VAL_50:.*]] = stablehlo.add %[[VAL_41]], %[[VAL_49]] : tensor<f32>
// CHECK:           %[[VAL_51:.*]] = stablehlo.subtract %[[VAL_13]], %[[VAL_48]] : tensor<f32>
// CHECK:           %[[VAL_52:.*]] = stablehlo.multiply %[[VAL_48]], %[[VAL_51]] : tensor<f32>
// CHECK:           %[[VAL_53:.*]] = stablehlo.add %[[VAL_50]], %[[VAL_52]] : tensor<f32>
// CHECK:           %[[VAL_54:.*]] = stablehlo.add %[[VAL_53]], %[[VAL_52]] : tensor<f32>
// CHECK:           %[[VAL_55:.*]] = stablehlo.multiply %[[VAL_51]], %[[VAL_51]] : tensor<f32>
// CHECK:           %[[VAL_56:.*]] = stablehlo.add %[[VAL_54]], %[[VAL_55]] : tensor<f32>
// CHECK:           %[[VAL_57:.*]] = stablehlo.add %[[VAL_40]], %[[VAL_56]] : tensor<f32>
// CHECK:           %[[VAL_58:.*]] = stablehlo.negate %[[VAL_39]] : tensor<f32>
// CHECK:           %[[VAL_59:.*]] = stablehlo.multiply %[[VAL_45]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_60:.*]] = stablehlo.subtract %[[VAL_11]], %[[VAL_59]] : tensor<f32>
// CHECK:           %[[VAL_61:.*]] = stablehlo.add %[[VAL_59]], %[[VAL_60]] : tensor<f32>
// CHECK:           %[[VAL_62:.*]] = stablehlo.multiply %[[VAL_61]], %[[VAL_61]] : tensor<f32>
// CHECK:           %[[VAL_63:.*]] = stablehlo.add %[[VAL_58]], %[[VAL_62]] : tensor<f32>
// CHECK:           %[[VAL_64:.*]] = stablehlo.subtract %[[VAL_11]], %[[VAL_61]] : tensor<f32>
// CHECK:           %[[VAL_65:.*]] = stablehlo.multiply %[[VAL_61]], %[[VAL_64]] : tensor<f32>
// CHECK:           %[[VAL_66:.*]] = stablehlo.add %[[VAL_63]], %[[VAL_65]] : tensor<f32>
// CHECK:           %[[VAL_67:.*]] = stablehlo.add %[[VAL_66]], %[[VAL_65]] : tensor<f32>
// CHECK:           %[[VAL_68:.*]] = stablehlo.multiply %[[VAL_64]], %[[VAL_64]] : tensor<f32>
// CHECK:           %[[VAL_69:.*]] = stablehlo.add %[[VAL_67]], %[[VAL_68]] : tensor<f32>
// CHECK:           %[[VAL_70:.*]] = stablehlo.add %[[VAL_57]], %[[VAL_69]] : tensor<f32>
// CHECK:           %[[VAL_71:.*]] = stablehlo.subtract %[[VAL_38]], %[[VAL_37]] : tensor<f32>
// CHECK:           %[[VAL_72:.*]] = stablehlo.subtract %[[VAL_38]], %[[VAL_71]] : tensor<f32>
// CHECK:           %[[VAL_73:.*]] = stablehlo.subtract %[[VAL_37]], %[[VAL_72]] : tensor<f32>
// CHECK:           %[[VAL_74:.*]] = stablehlo.subtract %[[VAL_33]], %[[VAL_71]] : tensor<f32>
// CHECK:           %[[VAL_75:.*]] = stablehlo.add %[[VAL_73]], %[[VAL_74]] : tensor<f32>
// CHECK:           %[[VAL_76:.*]] = stablehlo.subtract %[[VAL_40]], %[[VAL_38]] : tensor<f32>
// CHECK:           %[[VAL_77:.*]] = stablehlo.subtract %[[VAL_40]], %[[VAL_76]] : tensor<f32>
// CHECK:           %[[VAL_78:.*]] = stablehlo.subtract %[[VAL_38]], %[[VAL_77]] : tensor<f32>
// CHECK:           %[[VAL_79:.*]] = stablehlo.subtract %[[VAL_39]], %[[VAL_76]] : tensor<f32>
// CHECK:           %[[VAL_80:.*]] = stablehlo.add %[[VAL_78]], %[[VAL_79]] : tensor<f32>
// CHECK:           %[[VAL_81:.*]] = stablehlo.add %[[VAL_75]], %[[VAL_80]] : tensor<f32>
// CHECK:           %[[VAL_82:.*]] = stablehlo.subtract %[[VAL_57]], %[[VAL_40]] : tensor<f32>
// CHECK:           %[[VAL_83:.*]] = stablehlo.subtract %[[VAL_57]], %[[VAL_82]] : tensor<f32>
// CHECK:           %[[VAL_84:.*]] = stablehlo.subtract %[[VAL_40]], %[[VAL_83]] : tensor<f32>
// CHECK:           %[[VAL_85:.*]] = stablehlo.subtract %[[VAL_56]], %[[VAL_82]] : tensor<f32>
// CHECK:           %[[VAL_86:.*]] = stablehlo.add %[[VAL_84]], %[[VAL_85]] : tensor<f32>
// CHECK:           %[[VAL_87:.*]] = stablehlo.add %[[VAL_81]], %[[VAL_86]] : tensor<f32>
// CHECK:           %[[VAL_88:.*]] = stablehlo.subtract %[[VAL_70]], %[[VAL_57]] : tensor<f32>
// CHECK:           %[[VAL_89:.*]] = stablehlo.subtract %[[VAL_70]], %[[VAL_88]] : tensor<f32>
// CHECK:           %[[VAL_90:.*]] = stablehlo.subtract %[[VAL_57]], %[[VAL_89]] : tensor<f32>
// CHECK:           %[[VAL_91:.*]] = stablehlo.subtract %[[VAL_69]], %[[VAL_88]] : tensor<f32>
// CHECK:           %[[VAL_92:.*]] = stablehlo.add %[[VAL_90]], %[[VAL_91]] : tensor<f32>
// CHECK:           %[[VAL_93:.*]] = stablehlo.add %[[VAL_87]], %[[VAL_92]] : tensor<f32>
// CHECK:           %[[VAL_94:.*]] = stablehlo.add %[[VAL_70]], %[[VAL_93]] : tensor<f32>
// CHECK:           %[[VAL_95:.*]] = stablehlo.log_plus_one %[[VAL_94]] : tensor<f32>
// CHECK:           %[[VAL_96:.*]] = stablehlo.multiply %[[VAL_8]], %[[VAL_95]] : tensor<f32>
// CHECK:           %[[VAL_97:.*]] = stablehlo.select %[[VAL_31]], %[[VAL_36]], %[[VAL_96]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_98:.*]] = stablehlo.select %[[VAL_18]], %[[VAL_27]], %[[VAL_97]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_99:.*]] = stablehlo.atan2 %[[VAL_13]], %[[VAL_28]] : tensor<f32>
// CHECK:           %[[VAL_100:.*]] = stablehlo.complex %[[VAL_98]], %[[VAL_99]] : tensor<complex<f32>>
// CHECK:           return %[[VAL_100]] : tensor<complex<f32>>
// CHECK:         }
func.func @log_plus_one_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "stablehlo.log_plus_one"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @log_plus_one_complex_result_accuracy_f32(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK:           %0 = stablehlo.log_plus_one %arg0 {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e-05, ulps = 1, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : tensor<complex<f32>>
// CHECK:    return %0 : tensor<complex<f32>>
// CHECK:  }
// CHECK: }
func.func @log_plus_one_complex_result_accuracy_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "stablehlo.log_plus_one"(%arg) {
    result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e-05, rtol = 0.000000e+00, ulps = 1, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

