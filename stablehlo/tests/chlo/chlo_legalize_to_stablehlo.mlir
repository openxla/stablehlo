// RUN: stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @asin_bf16(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<bf16>) -> tensor<bf16> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
// CHECK:           %[[VAL_2:.*]] = stablehlo.subtract %[[VAL_1]], %[[VAL_0]] : tensor<bf16>
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_0]] : tensor<bf16>
// CHECK:           %[[VAL_4:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<bf16>
// CHECK:           %[[VAL_5:.*]] = stablehlo.sqrt %[[VAL_4]] : tensor<bf16>
// CHECK:           %[[VAL_6:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_5]] : tensor<bf16>
// CHECK:           %[[VAL_7:.*]] = stablehlo.atan2 %[[VAL_0]], %[[VAL_6]] : tensor<bf16>
// CHECK:           %[[VAL_8:.*]] = stablehlo.add %[[VAL_7]], %[[VAL_7]] : tensor<bf16>
// CHECK:           return %[[VAL_8]] : tensor<bf16>
// CHECK:         }
func.func @asin_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  %result = "chlo.asin"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %result : tensor<bf16>
}

// -----

// CHECK-LABEL:   func.func @asin_f16(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<f16>) -> tensor<f16> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f16>
// CHECK:           %[[VAL_2:.*]] = stablehlo.subtract %[[VAL_1]], %[[VAL_0]] : tensor<f16>
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_0]] : tensor<f16>
// CHECK:           %[[VAL_4:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<f16>
// CHECK:           %[[VAL_5:.*]] = stablehlo.sqrt %[[VAL_4]] : tensor<f16>
// CHECK:           %[[VAL_6:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_5]] : tensor<f16>
// CHECK:           %[[VAL_7:.*]] = stablehlo.atan2 %[[VAL_0]], %[[VAL_6]] : tensor<f16>
// CHECK:           %[[VAL_8:.*]] = stablehlo.add %[[VAL_7]], %[[VAL_7]] : tensor<f16>
// CHECK:           return %[[VAL_8]] : tensor<f16>
// CHECK:         }
func.func @asin_f16(%arg : tensor<f16>) -> tensor<f16> {
  %result = "chlo.asin"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %result : tensor<f16>
}

// -----

// CHECK-LABEL:   func.func @asin_f32(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.subtract %[[VAL_1]], %[[VAL_0]] : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_0]] : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.sqrt %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.atan2 %[[VAL_0]], %[[VAL_6]] : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.add %[[VAL_7]], %[[VAL_7]] : tensor<f32>
// CHECK:           return %[[VAL_8]] : tensor<f32>
// CHECK:         }
func.func @asin_f32(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.asin"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @asin_f64(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<f64>) -> tensor<f64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.subtract %[[VAL_1]], %[[VAL_0]] : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_0]] : tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.sqrt %[[VAL_4]] : tensor<f64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_5]] : tensor<f64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.atan2 %[[VAL_0]], %[[VAL_6]] : tensor<f64>
// CHECK:           %[[VAL_8:.*]] = stablehlo.add %[[VAL_7]], %[[VAL_7]] : tensor<f64>
// CHECK:           return %[[VAL_8]] : tensor<f64>
// CHECK:         }
func.func @asin_f64(%arg : tensor<f64>) -> tensor<f64> {
  %result = "chlo.asin"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL:   func.func @asin_complex_f32(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.abs %[[VAL_2]] : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.abs %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.maximum %[[VAL_3]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.sqrt %[[VAL_7]] : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.divide %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.compare  GE, %[[VAL_6]], %[[VAL_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.compare  LE, %[[VAL_3]], %[[VAL_12]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.add %[[VAL_3]], %[[VAL_12]] : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.abs %[[VAL_15]] : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.maximum %[[VAL_16]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.minimum %[[VAL_16]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.compare  EQ, %[[VAL_17]], %[[VAL_18]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_20:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.sqrt %[[VAL_20]] : tensor<f32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.multiply %[[VAL_21]], %[[VAL_17]] : tensor<f32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.divide %[[VAL_18]], %[[VAL_17]] : tensor<f32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.multiply %[[VAL_23]], %[[VAL_23]] : tensor<f32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_24]] : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.sqrt %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.compare  EQ, %[[VAL_26]], %[[VAL_12]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_28:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.compare  GT, %[[VAL_24]], %[[VAL_28]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_30:.*]] = stablehlo.and %[[VAL_27]], %[[VAL_29]] : tensor<i1>
// CHECK:           %[[VAL_31:.*]] = stablehlo.multiply %[[VAL_17]], %[[VAL_24]] : tensor<f32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.divide %[[VAL_31]], %[[VAL_20]] : tensor<f32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.add %[[VAL_17]], %[[VAL_32]] : tensor<f32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.multiply %[[VAL_17]], %[[VAL_26]] : tensor<f32>
// CHECK:           %[[VAL_35:.*]] = stablehlo.select %[[VAL_30]], %[[VAL_33]], %[[VAL_34]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.select %[[VAL_19]], %[[VAL_22]], %[[VAL_35]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_37:.*]] = stablehlo.subtract %[[VAL_3]], %[[VAL_12]] : tensor<f32>
// CHECK:           %[[VAL_38:.*]] = stablehlo.abs %[[VAL_37]] : tensor<f32>
// CHECK:           %[[VAL_39:.*]] = stablehlo.maximum %[[VAL_38]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_40:.*]] = stablehlo.minimum %[[VAL_38]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_41:.*]] = stablehlo.compare  EQ, %[[VAL_39]], %[[VAL_40]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_42:.*]] = stablehlo.multiply %[[VAL_21]], %[[VAL_39]] : tensor<f32>
// CHECK:           %[[VAL_43:.*]] = stablehlo.divide %[[VAL_40]], %[[VAL_39]] : tensor<f32>
// CHECK:           %[[VAL_44:.*]] = stablehlo.multiply %[[VAL_43]], %[[VAL_43]] : tensor<f32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_44]] : tensor<f32>
// CHECK:           %[[VAL_46:.*]] = stablehlo.sqrt %[[VAL_45]] : tensor<f32>
// CHECK:           %[[VAL_47:.*]] = stablehlo.compare  EQ, %[[VAL_46]], %[[VAL_12]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_48:.*]] = stablehlo.compare  GT, %[[VAL_44]], %[[VAL_28]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_49:.*]] = stablehlo.and %[[VAL_47]], %[[VAL_48]] : tensor<i1>
// CHECK:           %[[VAL_50:.*]] = stablehlo.multiply %[[VAL_39]], %[[VAL_44]] : tensor<f32>
// CHECK:           %[[VAL_51:.*]] = stablehlo.divide %[[VAL_50]], %[[VAL_20]] : tensor<f32>
// CHECK:           %[[VAL_52:.*]] = stablehlo.add %[[VAL_39]], %[[VAL_51]] : tensor<f32>
// CHECK:           %[[VAL_53:.*]] = stablehlo.multiply %[[VAL_39]], %[[VAL_46]] : tensor<f32>
// CHECK:           %[[VAL_54:.*]] = stablehlo.select %[[VAL_49]], %[[VAL_52]], %[[VAL_53]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_55:.*]] = stablehlo.select %[[VAL_41]], %[[VAL_42]], %[[VAL_54]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_56:.*]] = stablehlo.add %[[VAL_36]], %[[VAL_55]] : tensor<f32>
// CHECK:           %[[VAL_57:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_56]] : tensor<f32>
// CHECK:           %[[VAL_58:.*]] = stablehlo.add %[[VAL_57]], %[[VAL_3]] : tensor<f32>
// CHECK:           %[[VAL_59:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_58]] : tensor<f32>
// CHECK:           %[[VAL_60:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_61:.*]] = stablehlo.add %[[VAL_36]], %[[VAL_15]] : tensor<f32>
// CHECK:           %[[VAL_62:.*]] = stablehlo.divide %[[VAL_60]], %[[VAL_61]] : tensor<f32>
// CHECK:           %[[VAL_63:.*]] = stablehlo.subtract %[[VAL_55]], %[[VAL_37]] : tensor<f32>
// CHECK:           %[[VAL_64:.*]] = stablehlo.add %[[VAL_62]], %[[VAL_63]] : tensor<f32>
// CHECK:           %[[VAL_65:.*]] = stablehlo.multiply %[[VAL_59]], %[[VAL_64]] : tensor<f32>
// CHECK:           %[[VAL_66:.*]] = stablehlo.sqrt %[[VAL_65]] : tensor<f32>
// CHECK:           %[[VAL_67:.*]] = stablehlo.divide %[[VAL_59]], %[[VAL_61]] : tensor<f32>
// CHECK:           %[[VAL_68:.*]] = stablehlo.add %[[VAL_55]], %[[VAL_37]] : tensor<f32>
// CHECK:           %[[VAL_69:.*]] = stablehlo.divide %[[VAL_59]], %[[VAL_68]] : tensor<f32>
// CHECK:           %[[VAL_70:.*]] = stablehlo.add %[[VAL_67]], %[[VAL_69]] : tensor<f32>
// CHECK:           %[[VAL_71:.*]] = stablehlo.sqrt %[[VAL_70]] : tensor<f32>
// CHECK:           %[[VAL_72:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_71]] : tensor<f32>
// CHECK:           %[[VAL_73:.*]] = stablehlo.select %[[VAL_13]], %[[VAL_66]], %[[VAL_72]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_74:.*]] = stablehlo.select %[[VAL_11]], %[[VAL_5]], %[[VAL_73]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_75:.*]] = stablehlo.constant dense<9.99999995E+11> : tensor<f32>
// CHECK:           %[[VAL_76:.*]] = stablehlo.multiply %[[VAL_10]], %[[VAL_75]] : tensor<f32>
// CHECK:           %[[VAL_77:.*]] = stablehlo.compare  LT, %[[VAL_3]], %[[VAL_76]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_78:.*]] = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
// CHECK:           %[[VAL_79:.*]] = stablehlo.multiply %[[VAL_10]], %[[VAL_78]] : tensor<f32>
// CHECK:           %[[VAL_80:.*]] = stablehlo.constant dense<1.000000e+02> : tensor<f32>
// CHECK:           %[[VAL_81:.*]] = stablehlo.multiply %[[VAL_10]], %[[VAL_80]] : tensor<f32>
// CHECK:           %[[VAL_82:.*]] = stablehlo.select %[[VAL_77]], %[[VAL_79]], %[[VAL_81]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_83:.*]] = stablehlo.compare  GE, %[[VAL_5]], %[[VAL_82]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_84:.*]] = stablehlo.select %[[VAL_83]], %[[VAL_5]], %[[VAL_3]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_85:.*]] = stablehlo.select %[[VAL_83]], %[[VAL_82]], %[[VAL_10]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_86:.*]] = stablehlo.compare  GE, %[[VAL_84]], %[[VAL_85]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_87:.*]] = stablehlo.log %[[VAL_20]] : tensor<f32>
// CHECK:           %[[VAL_88:.*]] = stablehlo.log %[[VAL_84]] : tensor<f32>
// CHECK:           %[[VAL_89:.*]] = stablehlo.add %[[VAL_87]], %[[VAL_88]] : tensor<f32>
// CHECK:           %[[VAL_90:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           %[[VAL_91:.*]] = stablehlo.compare  EQ, %[[VAL_5]], %[[VAL_90]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_92:.*]] = stablehlo.not %[[VAL_91]] : tensor<i1>
// CHECK:           %[[VAL_93:.*]] = stablehlo.and %[[VAL_83]], %[[VAL_92]] : tensor<i1>
// CHECK:           %[[VAL_94:.*]] = stablehlo.divide %[[VAL_3]], %[[VAL_5]] : tensor<f32>
// CHECK:           %[[VAL_95:.*]] = stablehlo.select %[[VAL_93]], %[[VAL_94]], %[[VAL_28]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_96:.*]] = stablehlo.multiply %[[VAL_95]], %[[VAL_95]] : tensor<f32>
// CHECK:           %[[VAL_97:.*]] = stablehlo.log_plus_one %[[VAL_96]] : tensor<f32>
// CHECK:           %[[VAL_98:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_97]] : tensor<f32>
// CHECK:           %[[VAL_99:.*]] = stablehlo.add %[[VAL_89]], %[[VAL_98]] : tensor<f32>
// CHECK:           %[[VAL_100:.*]] = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
// CHECK:           %[[VAL_101:.*]] = stablehlo.sqrt %[[VAL_100]] : tensor<f32>
// CHECK:           %[[VAL_102:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_103:.*]] = stablehlo.multiply %[[VAL_101]], %[[VAL_102]] : tensor<f32>
// CHECK:           %[[VAL_104:.*]] = stablehlo.compare  LT, %[[VAL_5]], %[[VAL_103]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_105:.*]] = stablehlo.compare  LT, %[[VAL_3]], %[[VAL_12]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_106:.*]] = stablehlo.and %[[VAL_104]], %[[VAL_105]] : tensor<i1>
// CHECK:           %[[VAL_107:.*]] = stablehlo.multiply %[[VAL_15]], %[[VAL_37]] : tensor<f32>
// CHECK:           %[[VAL_108:.*]] = stablehlo.add %[[VAL_57]], %[[VAL_12]] : tensor<f32>
// CHECK:           %[[VAL_109:.*]] = stablehlo.divide %[[VAL_107]], %[[VAL_108]] : tensor<f32>
// CHECK:           %[[VAL_110:.*]] = stablehlo.negate %[[VAL_109]] : tensor<f32>
// CHECK:           %[[VAL_111:.*]] = stablehlo.compare  GE, %[[VAL_3]], %[[VAL_12]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_112:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_60]] : tensor<f32>
// CHECK:           %[[VAL_113:.*]] = stablehlo.divide %[[VAL_112]], %[[VAL_61]] : tensor<f32>
// CHECK:           %[[VAL_114:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_68]] : tensor<f32>
// CHECK:           %[[VAL_115:.*]] = stablehlo.add %[[VAL_113]], %[[VAL_114]] : tensor<f32>
// CHECK:           %[[VAL_116:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK:           %[[VAL_117:.*]] = stablehlo.compare  LE, %[[VAL_57]], %[[VAL_116]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_118:.*]] = stablehlo.divide %[[VAL_112]], %[[VAL_63]] : tensor<f32>
// CHECK:           %[[VAL_119:.*]] = stablehlo.add %[[VAL_113]], %[[VAL_118]] : tensor<f32>
// CHECK:           %[[VAL_120:.*]] = stablehlo.subtract %[[VAL_57]], %[[VAL_12]] : tensor<f32>
// CHECK:           %[[VAL_121:.*]] = stablehlo.select %[[VAL_117]], %[[VAL_119]], %[[VAL_120]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_122:.*]] = stablehlo.select %[[VAL_111]], %[[VAL_115]], %[[VAL_121]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_123:.*]] = stablehlo.select %[[VAL_106]], %[[VAL_110]], %[[VAL_122]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_124:.*]] = stablehlo.multiply %[[VAL_123]], %[[VAL_108]] : tensor<f32>
// CHECK:           %[[VAL_125:.*]] = stablehlo.sqrt %[[VAL_124]] : tensor<f32>
// CHECK:           %[[VAL_126:.*]] = stablehlo.divide %[[VAL_5]], %[[VAL_125]] : tensor<f32>
// CHECK:           %[[VAL_127:.*]] = stablehlo.add %[[VAL_123]], %[[VAL_125]] : tensor<f32>
// CHECK:           %[[VAL_128:.*]] = stablehlo.log_plus_one %[[VAL_127]] : tensor<f32>
// CHECK:           %[[VAL_129:.*]] = stablehlo.select %[[VAL_106]], %[[VAL_126]], %[[VAL_128]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_130:.*]] = stablehlo.select %[[VAL_86]], %[[VAL_99]], %[[VAL_129]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_131:.*]] = stablehlo.complex %[[VAL_74]], %[[VAL_130]] : tensor<complex<f32>>
// CHECK:           %[[VAL_132:.*]] = stablehlo.real %[[VAL_131]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_133:.*]] = stablehlo.atan2 %[[VAL_1]], %[[VAL_132]] : tensor<f32>
// CHECK:           %[[VAL_134:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_135:.*]] = stablehlo.imag %[[VAL_131]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_136:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_137:.*]] = stablehlo.compare  LT, %[[VAL_134]], %[[VAL_136]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_138:.*]] = stablehlo.negate %[[VAL_135]] : tensor<f32>
// CHECK:           %[[VAL_139:.*]] = stablehlo.select %[[VAL_137]], %[[VAL_138]], %[[VAL_135]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_140:.*]] = stablehlo.complex %[[VAL_133]], %[[VAL_139]] : tensor<complex<f32>>
// CHECK:           return %[[VAL_140]] : tensor<complex<f32>>
// CHECK:         }
func.func @asin_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.asin"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @asin_complex_f64_dynamic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.real %[[VAL_0]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.real %[[VAL_0]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.abs %[[VAL_2]] : tensor<?xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.abs %[[VAL_4]] : tensor<?xf64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.maximum %[[VAL_3]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK:           %[[VAL_8:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_9:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_7]], %[[VAL_8]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.sqrt %[[VAL_9]] : tensor<?xf64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_12:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_13:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_11]], %[[VAL_12]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_14:.*]] = stablehlo.divide %[[VAL_10]], %[[VAL_13]] : tensor<?xf64>
// CHECK:           %[[VAL_15:.*]] = stablehlo.compare  GE, %[[VAL_6]], %[[VAL_14]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_16:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_17:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_18:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_16]], %[[VAL_17]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_19:.*]] = stablehlo.compare  LE, %[[VAL_3]], %[[VAL_18]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_20:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK:           %[[VAL_21:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_22:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_20]], %[[VAL_21]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_23:.*]] = stablehlo.add %[[VAL_3]], %[[VAL_18]] : tensor<?xf64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.abs %[[VAL_23]] : tensor<?xf64>
// CHECK:           %[[VAL_25:.*]] = stablehlo.maximum %[[VAL_24]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_26:.*]] = stablehlo.minimum %[[VAL_24]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_27:.*]] = stablehlo.compare  EQ, %[[VAL_25]], %[[VAL_26]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_28:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_29:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_30:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_28]], %[[VAL_29]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_31:.*]] = stablehlo.sqrt %[[VAL_30]] : tensor<?xf64>
// CHECK:           %[[VAL_32:.*]] = stablehlo.multiply %[[VAL_31]], %[[VAL_25]] : tensor<?xf64>
// CHECK:           %[[VAL_33:.*]] = stablehlo.divide %[[VAL_26]], %[[VAL_25]] : tensor<?xf64>
// CHECK:           %[[VAL_34:.*]] = stablehlo.multiply %[[VAL_33]], %[[VAL_33]] : tensor<?xf64>
// CHECK:           %[[VAL_35:.*]] = stablehlo.add %[[VAL_18]], %[[VAL_34]] : tensor<?xf64>
// CHECK:           %[[VAL_36:.*]] = stablehlo.sqrt %[[VAL_35]] : tensor<?xf64>
// CHECK:           %[[VAL_37:.*]] = stablehlo.compare  EQ, %[[VAL_36]], %[[VAL_18]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_38:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_39:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_40:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_38]], %[[VAL_39]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_41:.*]] = stablehlo.compare  GT, %[[VAL_34]], %[[VAL_40]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_42:.*]] = stablehlo.and %[[VAL_37]], %[[VAL_41]] : tensor<?xi1>
// CHECK:           %[[VAL_43:.*]] = stablehlo.multiply %[[VAL_25]], %[[VAL_34]] : tensor<?xf64>
// CHECK:           %[[VAL_44:.*]] = stablehlo.divide %[[VAL_43]], %[[VAL_30]] : tensor<?xf64>
// CHECK:           %[[VAL_45:.*]] = stablehlo.add %[[VAL_25]], %[[VAL_44]] : tensor<?xf64>
// CHECK:           %[[VAL_46:.*]] = stablehlo.multiply %[[VAL_25]], %[[VAL_36]] : tensor<?xf64>
// CHECK:           %[[VAL_47:.*]] = stablehlo.select %[[VAL_42]], %[[VAL_45]], %[[VAL_46]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_48:.*]] = stablehlo.select %[[VAL_27]], %[[VAL_32]], %[[VAL_47]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_49:.*]] = stablehlo.subtract %[[VAL_3]], %[[VAL_18]] : tensor<?xf64>
// CHECK:           %[[VAL_50:.*]] = stablehlo.abs %[[VAL_49]] : tensor<?xf64>
// CHECK:           %[[VAL_51:.*]] = stablehlo.maximum %[[VAL_50]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_52:.*]] = stablehlo.minimum %[[VAL_50]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_53:.*]] = stablehlo.compare  EQ, %[[VAL_51]], %[[VAL_52]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_54:.*]] = stablehlo.multiply %[[VAL_31]], %[[VAL_51]] : tensor<?xf64>
// CHECK:           %[[VAL_55:.*]] = stablehlo.divide %[[VAL_52]], %[[VAL_51]] : tensor<?xf64>
// CHECK:           %[[VAL_56:.*]] = stablehlo.multiply %[[VAL_55]], %[[VAL_55]] : tensor<?xf64>
// CHECK:           %[[VAL_57:.*]] = stablehlo.add %[[VAL_18]], %[[VAL_56]] : tensor<?xf64>
// CHECK:           %[[VAL_58:.*]] = stablehlo.sqrt %[[VAL_57]] : tensor<?xf64>
// CHECK:           %[[VAL_59:.*]] = stablehlo.compare  EQ, %[[VAL_58]], %[[VAL_18]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_60:.*]] = stablehlo.compare  GT, %[[VAL_56]], %[[VAL_40]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_61:.*]] = stablehlo.and %[[VAL_59]], %[[VAL_60]] : tensor<?xi1>
// CHECK:           %[[VAL_62:.*]] = stablehlo.multiply %[[VAL_51]], %[[VAL_56]] : tensor<?xf64>
// CHECK:           %[[VAL_63:.*]] = stablehlo.divide %[[VAL_62]], %[[VAL_30]] : tensor<?xf64>
// CHECK:           %[[VAL_64:.*]] = stablehlo.add %[[VAL_51]], %[[VAL_63]] : tensor<?xf64>
// CHECK:           %[[VAL_65:.*]] = stablehlo.multiply %[[VAL_51]], %[[VAL_58]] : tensor<?xf64>
// CHECK:           %[[VAL_66:.*]] = stablehlo.select %[[VAL_61]], %[[VAL_64]], %[[VAL_65]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_67:.*]] = stablehlo.select %[[VAL_53]], %[[VAL_54]], %[[VAL_66]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_68:.*]] = stablehlo.add %[[VAL_48]], %[[VAL_67]] : tensor<?xf64>
// CHECK:           %[[VAL_69:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_68]] : tensor<?xf64>
// CHECK:           %[[VAL_70:.*]] = stablehlo.add %[[VAL_69]], %[[VAL_3]] : tensor<?xf64>
// CHECK:           %[[VAL_71:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_70]] : tensor<?xf64>
// CHECK:           %[[VAL_72:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_73:.*]] = stablehlo.add %[[VAL_48]], %[[VAL_23]] : tensor<?xf64>
// CHECK:           %[[VAL_74:.*]] = stablehlo.divide %[[VAL_72]], %[[VAL_73]] : tensor<?xf64>
// CHECK:           %[[VAL_75:.*]] = stablehlo.subtract %[[VAL_67]], %[[VAL_49]] : tensor<?xf64>
// CHECK:           %[[VAL_76:.*]] = stablehlo.add %[[VAL_74]], %[[VAL_75]] : tensor<?xf64>
// CHECK:           %[[VAL_77:.*]] = stablehlo.multiply %[[VAL_71]], %[[VAL_76]] : tensor<?xf64>
// CHECK:           %[[VAL_78:.*]] = stablehlo.sqrt %[[VAL_77]] : tensor<?xf64>
// CHECK:           %[[VAL_79:.*]] = stablehlo.divide %[[VAL_71]], %[[VAL_73]] : tensor<?xf64>
// CHECK:           %[[VAL_80:.*]] = stablehlo.add %[[VAL_67]], %[[VAL_49]] : tensor<?xf64>
// CHECK:           %[[VAL_81:.*]] = stablehlo.divide %[[VAL_71]], %[[VAL_80]] : tensor<?xf64>
// CHECK:           %[[VAL_82:.*]] = stablehlo.add %[[VAL_79]], %[[VAL_81]] : tensor<?xf64>
// CHECK:           %[[VAL_83:.*]] = stablehlo.sqrt %[[VAL_82]] : tensor<?xf64>
// CHECK:           %[[VAL_84:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_83]] : tensor<?xf64>
// CHECK:           %[[VAL_85:.*]] = stablehlo.select %[[VAL_19]], %[[VAL_78]], %[[VAL_84]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_86:.*]] = stablehlo.select %[[VAL_15]], %[[VAL_5]], %[[VAL_85]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_87:.*]] = stablehlo.constant dense<1.000000e+12> : tensor<f64>
// CHECK:           %[[VAL_88:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_89:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_87]], %[[VAL_88]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_90:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_89]] : tensor<?xf64>
// CHECK:           %[[VAL_91:.*]] = stablehlo.compare  LT, %[[VAL_3]], %[[VAL_90]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_92:.*]] = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
// CHECK:           %[[VAL_93:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_94:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_92]], %[[VAL_93]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_95:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_94]] : tensor<?xf64>
// CHECK:           %[[VAL_96:.*]] = stablehlo.constant dense<1.000000e+02> : tensor<f64>
// CHECK:           %[[VAL_97:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_98:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_96]], %[[VAL_97]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_99:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_98]] : tensor<?xf64>
// CHECK:           %[[VAL_100:.*]] = stablehlo.select %[[VAL_91]], %[[VAL_95]], %[[VAL_99]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_101:.*]] = stablehlo.compare  GE, %[[VAL_5]], %[[VAL_100]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_102:.*]] = stablehlo.select %[[VAL_101]], %[[VAL_5]], %[[VAL_3]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_103:.*]] = stablehlo.select %[[VAL_101]], %[[VAL_100]], %[[VAL_14]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_104:.*]] = stablehlo.compare  GE, %[[VAL_102]], %[[VAL_103]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_105:.*]] = stablehlo.log %[[VAL_30]] : tensor<?xf64>
// CHECK:           %[[VAL_106:.*]] = stablehlo.log %[[VAL_102]] : tensor<?xf64>
// CHECK:           %[[VAL_107:.*]] = stablehlo.add %[[VAL_105]], %[[VAL_106]] : tensor<?xf64>
// CHECK:           %[[VAL_108:.*]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK:           %[[VAL_109:.*]] = shape.shape_of %[[VAL_4]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_110:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_108]], %[[VAL_109]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_111:.*]] = stablehlo.compare  EQ, %[[VAL_5]], %[[VAL_110]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_112:.*]] = stablehlo.not %[[VAL_111]] : tensor<?xi1>
// CHECK:           %[[VAL_113:.*]] = stablehlo.and %[[VAL_101]], %[[VAL_112]] : tensor<?xi1>
// CHECK:           %[[VAL_114:.*]] = stablehlo.divide %[[VAL_3]], %[[VAL_5]] : tensor<?xf64>
// CHECK:           %[[VAL_115:.*]] = stablehlo.select %[[VAL_113]], %[[VAL_114]], %[[VAL_40]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_116:.*]] = stablehlo.multiply %[[VAL_115]], %[[VAL_115]] : tensor<?xf64>
// CHECK:           %[[VAL_117:.*]] = stablehlo.log_plus_one %[[VAL_116]] : tensor<?xf64>
// CHECK:           %[[VAL_118:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_117]] : tensor<?xf64>
// CHECK:           %[[VAL_119:.*]] = stablehlo.add %[[VAL_107]], %[[VAL_118]] : tensor<?xf64>
// CHECK:           %[[VAL_120:.*]] = stablehlo.constant dense<2.2250738585072014E-308> : tensor<f64>
// CHECK:           %[[VAL_121:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_122:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_120]], %[[VAL_121]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_123:.*]] = stablehlo.sqrt %[[VAL_122]] : tensor<?xf64>
// CHECK:           %[[VAL_124:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_125:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_126:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_124]], %[[VAL_125]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_127:.*]] = stablehlo.multiply %[[VAL_123]], %[[VAL_126]] : tensor<?xf64>
// CHECK:           %[[VAL_128:.*]] = stablehlo.compare  LT, %[[VAL_5]], %[[VAL_127]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_129:.*]] = stablehlo.compare  LT, %[[VAL_3]], %[[VAL_18]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_130:.*]] = stablehlo.and %[[VAL_128]], %[[VAL_129]] : tensor<?xi1>
// CHECK:           %[[VAL_131:.*]] = stablehlo.multiply %[[VAL_23]], %[[VAL_49]] : tensor<?xf64>
// CHECK:           %[[VAL_132:.*]] = stablehlo.add %[[VAL_69]], %[[VAL_18]] : tensor<?xf64>
// CHECK:           %[[VAL_133:.*]] = stablehlo.divide %[[VAL_131]], %[[VAL_132]] : tensor<?xf64>
// CHECK:           %[[VAL_134:.*]] = stablehlo.negate %[[VAL_133]] : tensor<?xf64>
// CHECK:           %[[VAL_135:.*]] = stablehlo.compare  GE, %[[VAL_3]], %[[VAL_18]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_136:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_72]] : tensor<?xf64>
// CHECK:           %[[VAL_137:.*]] = stablehlo.divide %[[VAL_136]], %[[VAL_73]] : tensor<?xf64>
// CHECK:           %[[VAL_138:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_80]] : tensor<?xf64>
// CHECK:           %[[VAL_139:.*]] = stablehlo.add %[[VAL_137]], %[[VAL_138]] : tensor<?xf64>
// CHECK:           %[[VAL_140:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<f64>
// CHECK:           %[[VAL_141:.*]] = shape.shape_of %[[VAL_2]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_142:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_140]], %[[VAL_141]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_143:.*]] = stablehlo.compare  LE, %[[VAL_69]], %[[VAL_142]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_144:.*]] = stablehlo.divide %[[VAL_136]], %[[VAL_75]] : tensor<?xf64>
// CHECK:           %[[VAL_145:.*]] = stablehlo.add %[[VAL_137]], %[[VAL_144]] : tensor<?xf64>
// CHECK:           %[[VAL_146:.*]] = stablehlo.subtract %[[VAL_69]], %[[VAL_18]] : tensor<?xf64>
// CHECK:           %[[VAL_147:.*]] = stablehlo.select %[[VAL_143]], %[[VAL_145]], %[[VAL_146]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_148:.*]] = stablehlo.select %[[VAL_135]], %[[VAL_139]], %[[VAL_147]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_149:.*]] = stablehlo.select %[[VAL_130]], %[[VAL_134]], %[[VAL_148]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_150:.*]] = stablehlo.multiply %[[VAL_149]], %[[VAL_132]] : tensor<?xf64>
// CHECK:           %[[VAL_151:.*]] = stablehlo.sqrt %[[VAL_150]] : tensor<?xf64>
// CHECK:           %[[VAL_152:.*]] = stablehlo.divide %[[VAL_5]], %[[VAL_151]] : tensor<?xf64>
// CHECK:           %[[VAL_153:.*]] = stablehlo.add %[[VAL_149]], %[[VAL_151]] : tensor<?xf64>
// CHECK:           %[[VAL_154:.*]] = stablehlo.log_plus_one %[[VAL_153]] : tensor<?xf64>
// CHECK:           %[[VAL_155:.*]] = stablehlo.select %[[VAL_130]], %[[VAL_152]], %[[VAL_154]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_156:.*]] = stablehlo.select %[[VAL_104]], %[[VAL_119]], %[[VAL_155]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_157:.*]] = stablehlo.complex %[[VAL_86]], %[[VAL_156]] : tensor<?xcomplex<f64>>
// CHECK:           %[[VAL_158:.*]] = stablehlo.real %[[VAL_157]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK:           %[[VAL_159:.*]] = stablehlo.atan2 %[[VAL_1]], %[[VAL_158]] : tensor<?xf64>
// CHECK:           %[[VAL_160:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK:           %[[VAL_161:.*]] = stablehlo.imag %[[VAL_157]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK:           %[[VAL_162:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_163:.*]] = shape.shape_of %[[VAL_161]] : tensor<?xf64> -> tensor<1xindex>
// CHECK:           %[[VAL_164:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_162]], %[[VAL_163]], dims = [] : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK:           %[[VAL_165:.*]] = stablehlo.compare  LT, %[[VAL_160]], %[[VAL_164]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK:           %[[VAL_166:.*]] = stablehlo.negate %[[VAL_161]] : tensor<?xf64>
// CHECK:           %[[VAL_167:.*]] = stablehlo.select %[[VAL_165]], %[[VAL_166]], %[[VAL_161]] : tensor<?xi1>, tensor<?xf64>
// CHECK:           %[[VAL_168:.*]] = stablehlo.complex %[[VAL_159]], %[[VAL_167]] : tensor<?xcomplex<f64>>
// CHECK:           return %[[VAL_168]] : tensor<?xcomplex<f64>>
// CHECK:         }
func.func @asin_complex_f64_dynamic(%arg : tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>> {
  %result = "chlo.asin"(%arg) : (tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>>
  func.return %result : tensor<?xcomplex<f64>>
}

// -----

// CHECK-LABEL: @asinh_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @asinh_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // Check for the bf16-specific max value.
  // CHECK: stablehlo.constant dense<3.389{{.*}}e+38>
  %result = "chlo.asinh"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %result : tensor<bf16>
}

// -----

// CHECK-LABEL: @asinh_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @asinh_f16(%arg : tensor<f16>) -> tensor<f16> {
  // Check for the f16-specific max value.
  // CHECK: stablehlo.constant dense<6.550{{.*}}e+04>
  %result = "chlo.asinh"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %result : tensor<f16>
}

// -----

// CHECK-LABEL: @asinh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @asinh_f32(%arg : tensor<f32>) -> tensor<f32> {
  // Check for the f32-specific max value.
  // CHECK: stablehlo.constant dense<3.402{{.*}}E+38>
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----


// CHECK-LABEL:   func.func @asinh_f64(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<f64>) -> tensor<f64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.sign %[[VAL_0]] : tensor<f64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.abs %[[VAL_0]] : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.sqrt %[[VAL_3]] : tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.compare  GE, %[[VAL_2]], %[[VAL_4]] : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.log %[[VAL_6]] : tensor<f64>
// CHECK:           %[[VAL_8:.*]] = stablehlo.log %[[VAL_2]] : tensor<f64>
// CHECK:           %[[VAL_9:.*]] = stablehlo.add %[[VAL_7]], %[[VAL_8]] : tensor<f64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_2]] : tensor<f64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_12:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_10]] : tensor<f64>
// CHECK:           %[[VAL_13:.*]] = stablehlo.sqrt %[[VAL_12]] : tensor<f64>
// CHECK:           %[[VAL_14:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_13]] : tensor<f64>
// CHECK:           %[[VAL_15:.*]] = stablehlo.divide %[[VAL_10]], %[[VAL_14]] : tensor<f64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.add %[[VAL_2]], %[[VAL_15]] : tensor<f64>
// CHECK:           %[[VAL_17:.*]] = stablehlo.log_plus_one %[[VAL_16]] : tensor<f64>
// CHECK:           %[[VAL_18:.*]] = stablehlo.select %[[VAL_5]], %[[VAL_9]], %[[VAL_17]] : tensor<i1>, tensor<f64>
// CHECK:           %[[VAL_19:.*]] = stablehlo.multiply %[[VAL_1]], %[[VAL_18]] : tensor<f64>
// CHECK:           return %[[VAL_19]] : tensor<f64>
// CHECK:         }
func.func @asinh_f64(%arg : tensor<f64>) -> tensor<f64> {
  %result = "chlo.asinh"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL:   func.func @asinh_complex_f32(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.compare  LT, %[[VAL_1]], %[[VAL_2]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_4:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.negate %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.complex %[[VAL_5]], %[[VAL_1]] : tensor<complex<f32>>
// CHECK:           %[[VAL_7:.*]] = stablehlo.real %[[VAL_6]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.abs %[[VAL_7]] : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.imag %[[VAL_6]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.abs %[[VAL_9]] : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.maximum %[[VAL_8]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.sqrt %[[VAL_12]] : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.divide %[[VAL_13]], %[[VAL_14]] : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.compare  GE, %[[VAL_11]], %[[VAL_15]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_17:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.compare  LE, %[[VAL_8]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_19:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.add %[[VAL_8]], %[[VAL_17]] : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.abs %[[VAL_20]] : tensor<f32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.maximum %[[VAL_21]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.minimum %[[VAL_21]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.compare  EQ, %[[VAL_22]], %[[VAL_23]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_25:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.sqrt %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.multiply %[[VAL_26]], %[[VAL_22]] : tensor<f32>
// CHECK:           %[[VAL_28:.*]] = stablehlo.divide %[[VAL_23]], %[[VAL_22]] : tensor<f32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.multiply %[[VAL_28]], %[[VAL_28]] : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = stablehlo.add %[[VAL_17]], %[[VAL_29]] : tensor<f32>
// CHECK:           %[[VAL_31:.*]] = stablehlo.sqrt %[[VAL_30]] : tensor<f32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.compare  EQ, %[[VAL_31]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_33:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.compare  GT, %[[VAL_29]], %[[VAL_33]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_35:.*]] = stablehlo.and %[[VAL_32]], %[[VAL_34]] : tensor<i1>
// CHECK:           %[[VAL_36:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_29]] : tensor<f32>
// CHECK:           %[[VAL_37:.*]] = stablehlo.divide %[[VAL_36]], %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_38:.*]] = stablehlo.add %[[VAL_22]], %[[VAL_37]] : tensor<f32>
// CHECK:           %[[VAL_39:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_31]] : tensor<f32>
// CHECK:           %[[VAL_40:.*]] = stablehlo.select %[[VAL_35]], %[[VAL_38]], %[[VAL_39]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_41:.*]] = stablehlo.select %[[VAL_24]], %[[VAL_27]], %[[VAL_40]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_42:.*]] = stablehlo.subtract %[[VAL_8]], %[[VAL_17]] : tensor<f32>
// CHECK:           %[[VAL_43:.*]] = stablehlo.abs %[[VAL_42]] : tensor<f32>
// CHECK:           %[[VAL_44:.*]] = stablehlo.maximum %[[VAL_43]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.minimum %[[VAL_43]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_46:.*]] = stablehlo.compare  EQ, %[[VAL_44]], %[[VAL_45]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_47:.*]] = stablehlo.multiply %[[VAL_26]], %[[VAL_44]] : tensor<f32>
// CHECK:           %[[VAL_48:.*]] = stablehlo.divide %[[VAL_45]], %[[VAL_44]] : tensor<f32>
// CHECK:           %[[VAL_49:.*]] = stablehlo.multiply %[[VAL_48]], %[[VAL_48]] : tensor<f32>
// CHECK:           %[[VAL_50:.*]] = stablehlo.add %[[VAL_17]], %[[VAL_49]] : tensor<f32>
// CHECK:           %[[VAL_51:.*]] = stablehlo.sqrt %[[VAL_50]] : tensor<f32>
// CHECK:           %[[VAL_52:.*]] = stablehlo.compare  EQ, %[[VAL_51]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_53:.*]] = stablehlo.compare  GT, %[[VAL_49]], %[[VAL_33]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_54:.*]] = stablehlo.and %[[VAL_52]], %[[VAL_53]] : tensor<i1>
// CHECK:           %[[VAL_55:.*]] = stablehlo.multiply %[[VAL_44]], %[[VAL_49]] : tensor<f32>
// CHECK:           %[[VAL_56:.*]] = stablehlo.divide %[[VAL_55]], %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_57:.*]] = stablehlo.add %[[VAL_44]], %[[VAL_56]] : tensor<f32>
// CHECK:           %[[VAL_58:.*]] = stablehlo.multiply %[[VAL_44]], %[[VAL_51]] : tensor<f32>
// CHECK:           %[[VAL_59:.*]] = stablehlo.select %[[VAL_54]], %[[VAL_57]], %[[VAL_58]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_60:.*]] = stablehlo.select %[[VAL_46]], %[[VAL_47]], %[[VAL_59]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_61:.*]] = stablehlo.add %[[VAL_41]], %[[VAL_60]] : tensor<f32>
// CHECK:           %[[VAL_62:.*]] = stablehlo.multiply %[[VAL_19]], %[[VAL_61]] : tensor<f32>
// CHECK:           %[[VAL_63:.*]] = stablehlo.add %[[VAL_62]], %[[VAL_8]] : tensor<f32>
// CHECK:           %[[VAL_64:.*]] = stablehlo.multiply %[[VAL_19]], %[[VAL_63]] : tensor<f32>
// CHECK:           %[[VAL_65:.*]] = stablehlo.multiply %[[VAL_10]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_66:.*]] = stablehlo.add %[[VAL_41]], %[[VAL_20]] : tensor<f32>
// CHECK:           %[[VAL_67:.*]] = stablehlo.divide %[[VAL_65]], %[[VAL_66]] : tensor<f32>
// CHECK:           %[[VAL_68:.*]] = stablehlo.subtract %[[VAL_60]], %[[VAL_42]] : tensor<f32>
// CHECK:           %[[VAL_69:.*]] = stablehlo.add %[[VAL_67]], %[[VAL_68]] : tensor<f32>
// CHECK:           %[[VAL_70:.*]] = stablehlo.multiply %[[VAL_64]], %[[VAL_69]] : tensor<f32>
// CHECK:           %[[VAL_71:.*]] = stablehlo.sqrt %[[VAL_70]] : tensor<f32>
// CHECK:           %[[VAL_72:.*]] = stablehlo.divide %[[VAL_64]], %[[VAL_66]] : tensor<f32>
// CHECK:           %[[VAL_73:.*]] = stablehlo.add %[[VAL_60]], %[[VAL_42]] : tensor<f32>
// CHECK:           %[[VAL_74:.*]] = stablehlo.divide %[[VAL_64]], %[[VAL_73]] : tensor<f32>
// CHECK:           %[[VAL_75:.*]] = stablehlo.add %[[VAL_72]], %[[VAL_74]] : tensor<f32>
// CHECK:           %[[VAL_76:.*]] = stablehlo.sqrt %[[VAL_75]] : tensor<f32>
// CHECK:           %[[VAL_77:.*]] = stablehlo.multiply %[[VAL_10]], %[[VAL_76]] : tensor<f32>
// CHECK:           %[[VAL_78:.*]] = stablehlo.select %[[VAL_18]], %[[VAL_71]], %[[VAL_77]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_79:.*]] = stablehlo.select %[[VAL_16]], %[[VAL_10]], %[[VAL_78]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_80:.*]] = stablehlo.constant dense<9.99999995E+11> : tensor<f32>
// CHECK:           %[[VAL_81:.*]] = stablehlo.multiply %[[VAL_15]], %[[VAL_80]] : tensor<f32>
// CHECK:           %[[VAL_82:.*]] = stablehlo.compare  LT, %[[VAL_8]], %[[VAL_81]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_83:.*]] = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
// CHECK:           %[[VAL_84:.*]] = stablehlo.multiply %[[VAL_15]], %[[VAL_83]] : tensor<f32>
// CHECK:           %[[VAL_85:.*]] = stablehlo.constant dense<1.000000e+02> : tensor<f32>
// CHECK:           %[[VAL_86:.*]] = stablehlo.multiply %[[VAL_15]], %[[VAL_85]] : tensor<f32>
// CHECK:           %[[VAL_87:.*]] = stablehlo.select %[[VAL_82]], %[[VAL_84]], %[[VAL_86]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_88:.*]] = stablehlo.compare  GE, %[[VAL_10]], %[[VAL_87]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_89:.*]] = stablehlo.select %[[VAL_88]], %[[VAL_10]], %[[VAL_8]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_90:.*]] = stablehlo.select %[[VAL_88]], %[[VAL_87]], %[[VAL_15]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_91:.*]] = stablehlo.compare  GE, %[[VAL_89]], %[[VAL_90]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_92:.*]] = stablehlo.log %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_93:.*]] = stablehlo.log %[[VAL_89]] : tensor<f32>
// CHECK:           %[[VAL_94:.*]] = stablehlo.add %[[VAL_92]], %[[VAL_93]] : tensor<f32>
// CHECK:           %[[VAL_95:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           %[[VAL_96:.*]] = stablehlo.compare  EQ, %[[VAL_10]], %[[VAL_95]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_97:.*]] = stablehlo.not %[[VAL_96]] : tensor<i1>
// CHECK:           %[[VAL_98:.*]] = stablehlo.and %[[VAL_88]], %[[VAL_97]] : tensor<i1>
// CHECK:           %[[VAL_99:.*]] = stablehlo.divide %[[VAL_8]], %[[VAL_10]] : tensor<f32>
// CHECK:           %[[VAL_100:.*]] = stablehlo.select %[[VAL_98]], %[[VAL_99]], %[[VAL_33]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_101:.*]] = stablehlo.multiply %[[VAL_100]], %[[VAL_100]] : tensor<f32>
// CHECK:           %[[VAL_102:.*]] = stablehlo.log_plus_one %[[VAL_101]] : tensor<f32>
// CHECK:           %[[VAL_103:.*]] = stablehlo.multiply %[[VAL_19]], %[[VAL_102]] : tensor<f32>
// CHECK:           %[[VAL_104:.*]] = stablehlo.add %[[VAL_94]], %[[VAL_103]] : tensor<f32>
// CHECK:           %[[VAL_105:.*]] = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
// CHECK:           %[[VAL_106:.*]] = stablehlo.sqrt %[[VAL_105]] : tensor<f32>
// CHECK:           %[[VAL_107:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_108:.*]] = stablehlo.multiply %[[VAL_106]], %[[VAL_107]] : tensor<f32>
// CHECK:           %[[VAL_109:.*]] = stablehlo.compare  LT, %[[VAL_10]], %[[VAL_108]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_110:.*]] = stablehlo.compare  LT, %[[VAL_8]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_111:.*]] = stablehlo.and %[[VAL_109]], %[[VAL_110]] : tensor<i1>
// CHECK:           %[[VAL_112:.*]] = stablehlo.multiply %[[VAL_20]], %[[VAL_42]] : tensor<f32>
// CHECK:           %[[VAL_113:.*]] = stablehlo.add %[[VAL_62]], %[[VAL_17]] : tensor<f32>
// CHECK:           %[[VAL_114:.*]] = stablehlo.divide %[[VAL_112]], %[[VAL_113]] : tensor<f32>
// CHECK:           %[[VAL_115:.*]] = stablehlo.negate %[[VAL_114]] : tensor<f32>
// CHECK:           %[[VAL_116:.*]] = stablehlo.compare  GE, %[[VAL_8]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_117:.*]] = stablehlo.multiply %[[VAL_19]], %[[VAL_65]] : tensor<f32>
// CHECK:           %[[VAL_118:.*]] = stablehlo.divide %[[VAL_117]], %[[VAL_66]] : tensor<f32>
// CHECK:           %[[VAL_119:.*]] = stablehlo.multiply %[[VAL_19]], %[[VAL_73]] : tensor<f32>
// CHECK:           %[[VAL_120:.*]] = stablehlo.add %[[VAL_118]], %[[VAL_119]] : tensor<f32>
// CHECK:           %[[VAL_121:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK:           %[[VAL_122:.*]] = stablehlo.compare  LE, %[[VAL_62]], %[[VAL_121]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_123:.*]] = stablehlo.divide %[[VAL_117]], %[[VAL_68]] : tensor<f32>
// CHECK:           %[[VAL_124:.*]] = stablehlo.add %[[VAL_118]], %[[VAL_123]] : tensor<f32>
// CHECK:           %[[VAL_125:.*]] = stablehlo.subtract %[[VAL_62]], %[[VAL_17]] : tensor<f32>
// CHECK:           %[[VAL_126:.*]] = stablehlo.select %[[VAL_122]], %[[VAL_124]], %[[VAL_125]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_127:.*]] = stablehlo.select %[[VAL_116]], %[[VAL_120]], %[[VAL_126]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_128:.*]] = stablehlo.select %[[VAL_111]], %[[VAL_115]], %[[VAL_127]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_129:.*]] = stablehlo.multiply %[[VAL_128]], %[[VAL_113]] : tensor<f32>
// CHECK:           %[[VAL_130:.*]] = stablehlo.sqrt %[[VAL_129]] : tensor<f32>
// CHECK:           %[[VAL_131:.*]] = stablehlo.divide %[[VAL_10]], %[[VAL_130]] : tensor<f32>
// CHECK:           %[[VAL_132:.*]] = stablehlo.add %[[VAL_128]], %[[VAL_130]] : tensor<f32>
// CHECK:           %[[VAL_133:.*]] = stablehlo.log_plus_one %[[VAL_132]] : tensor<f32>
// CHECK:           %[[VAL_134:.*]] = stablehlo.select %[[VAL_111]], %[[VAL_131]], %[[VAL_133]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_135:.*]] = stablehlo.select %[[VAL_91]], %[[VAL_104]], %[[VAL_134]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_136:.*]] = stablehlo.complex %[[VAL_79]], %[[VAL_135]] : tensor<complex<f32>>
// CHECK:           %[[VAL_137:.*]] = stablehlo.imag %[[VAL_136]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_138:.*]] = stablehlo.negate %[[VAL_137]] : tensor<f32>
// CHECK:           %[[VAL_139:.*]] = stablehlo.select %[[VAL_3]], %[[VAL_138]], %[[VAL_137]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_140:.*]] = stablehlo.real %[[VAL_136]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_141:.*]] = stablehlo.atan2 %[[VAL_4]], %[[VAL_140]] : tensor<f32>
// CHECK:           %[[VAL_142:.*]] = stablehlo.complex %[[VAL_139]], %[[VAL_141]] : tensor<complex<f32>>
// CHECK:           return %[[VAL_142]] : tensor<complex<f32>>
// CHECK:         }
func.func @asinh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.asinh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// Lower statically shaped `constant_like` to constant.
// CHECK-LABEL: @constant_like_static_shape
func.func @constant_like_static_shape(%arg : tensor<1x2xi64>) -> tensor<1x2xf32> {
  // CHECK: %[[RESULT:.*]] = stablehlo.constant dense<3.200000e+00> : tensor<1x2xf32>
  // CHECK: return %[[RESULT]]
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<1x2xi64>) -> tensor<1x2xf32>
  func.return %result : tensor<1x2xf32>
}

// -----

// Lower dynamically shaped `constant_like` to broadcasted constant.
// CHECK-LABEL: constant_like_dynamic_shape
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x?xi64>)
func.func @constant_like_dynamic_shape(%arg : tensor<?x?xi64>) -> tensor<?x?xf32> {
  // CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<3.200000e+00> : tensor<f32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x?xi64> -> tensor<2xindex>
  // CHECK: %[[BROADCASTED_CONSTANT:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONSTANT]], %[[SHAPE]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: return %[[BROADCASTED_CONSTANT]] : tensor<?x?xf32>
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<?x?xi64>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @conj
func.func @conj(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-SAME: ([[INPUT:%.*]]: tensor
  // CHECK-NEXT: [[R1:%.*]] = stablehlo.real [[INPUT]]
  // CHECK-NEXT: [[R2:%.*]] = stablehlo.imag [[INPUT]]
  // CHECK-NEXT: [[R3:%.*]] = stablehlo.negate [[R2]]
  // CHECK-NEXT: [[R4:%.*]] = stablehlo.complex [[R1]], [[R3]]
  %1 = "chlo.conj"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  func.return %1 : tensor<3xcomplex<f32>>
}

// -----

// CHECK-LABEL: @erf_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erf_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = stablehlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.constant dense<9.6049737398705161>
  // CHECK: %[[TMP_5:.*]] = stablehlo.multiply %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.constant dense<90.026019720384269>
  // CHECK: %[[TMP_7:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_0]]
  // CHECK: %[[TMP_9:.*]] = stablehlo.constant dense<2232.0053459468431>
  // CHECK: %[[TMP_10:.*]] = stablehlo.add %[[TMP_8]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.multiply %[[TMP_10]], %[[TMP_0]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.constant dense<7003.3251411280507>
  // CHECK: %[[TMP_13:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = stablehlo.multiply %[[TMP_13]], %[[TMP_0]]
  // CHECK: %[[TMP_15:.*]] = stablehlo.constant dense<55592.301301039493>
  // CHECK: %[[TMP_16:.*]] = stablehlo.add %[[TMP_14]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.multiply %[[ARG]], %[[TMP_16]]
  // CHECK: %[[TMP_20:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_22:.*]] = stablehlo.multiply %[[TMP_20]], %[[TMP_0]]
  // CHECK: %[[TMP_23:.*]] = stablehlo.constant dense<33.561714164750313>
  // CHECK: %[[TMP_24:.*]] = stablehlo.add %[[TMP_22]], %[[TMP_23]]
  // CHECK: %[[TMP_25:.*]] = stablehlo.multiply %[[TMP_24]], %[[TMP_0]]
  // CHECK: %[[TMP_26:.*]] = stablehlo.constant dense<521.35794978015269>
  // CHECK: %[[TMP_27:.*]] = stablehlo.add %[[TMP_25]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = stablehlo.multiply %[[TMP_27]], %[[TMP_0]]
  // CHECK: %[[TMP_29:.*]] = stablehlo.constant dense<4594.3238297098014>
  // CHECK: %[[TMP_30:.*]] = stablehlo.add %[[TMP_28]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.multiply %[[TMP_30]], %[[TMP_0]]
  // CHECK: %[[TMP_32:.*]] = stablehlo.constant dense<22629.000061389095>
  // CHECK: %[[TMP_33:.*]] = stablehlo.add %[[TMP_31]], %[[TMP_32]]
  // CHECK: %[[TMP_34:.*]] = stablehlo.multiply %[[TMP_33]], %[[TMP_0]]
  // CHECK: %[[TMP_35:.*]] = stablehlo.constant dense<49267.394260863592>
  // CHECK: %[[TMP_36:.*]] = stablehlo.add %[[TMP_34]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.divide %[[TMP_17]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_39:.*]] = stablehlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_40:.*]] = stablehlo.negate %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.exponential %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_45:.*]] = stablehlo.constant dense<2.4619698147353052E-10>
  // CHECK: %[[TMP_47:.*]] = stablehlo.multiply %[[TMP_45]], %[[TMP_42]]
  // CHECK: %[[TMP_48:.*]] = stablehlo.constant dense<0.56418956483106886>
  // CHECK: %[[TMP_49:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_48]]
  // CHECK: %[[TMP_50:.*]] = stablehlo.multiply %[[TMP_49]], %[[TMP_42]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.constant dense<7.4632105644226989>
  // CHECK: %[[TMP_52:.*]] = stablehlo.add %[[TMP_50]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = stablehlo.multiply %[[TMP_52]], %[[TMP_42]]
  // CHECK: %[[TMP_54:.*]] = stablehlo.constant dense<48.637197098568137>
  // CHECK: %[[TMP_55:.*]] = stablehlo.add %[[TMP_53]], %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = stablehlo.multiply %[[TMP_55]], %[[TMP_42]]
  // CHECK: %[[TMP_57:.*]] = stablehlo.constant dense<196.5208329560771>
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_56]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.multiply %[[TMP_58]], %[[TMP_42]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.constant dense<526.44519499547732>
  // CHECK: %[[TMP_61:.*]] = stablehlo.add %[[TMP_59]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = stablehlo.multiply %[[TMP_61]], %[[TMP_42]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.constant dense<934.52852717195765>
  // CHECK: %[[TMP_64:.*]] = stablehlo.add %[[TMP_62]], %[[TMP_63]]
  // CHECK: %[[TMP_65:.*]] = stablehlo.multiply %[[TMP_64]], %[[TMP_42]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.constant dense<1027.5518868951572>
  // CHECK: %[[TMP_67:.*]] = stablehlo.add %[[TMP_65]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.multiply %[[TMP_67]], %[[TMP_42]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.constant dense<557.53533536939938>
  // CHECK: %[[TMP_70:.*]] = stablehlo.add %[[TMP_68]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.multiply %[[TMP_41]], %[[TMP_70]]
  // CHECK: %[[TMP_74:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_76:.*]] = stablehlo.multiply %[[TMP_74]], %[[TMP_42]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.constant dense<13.228195115474499>
  // CHECK: %[[TMP_78:.*]] = stablehlo.add %[[TMP_76]], %[[TMP_77]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.multiply %[[TMP_78]], %[[TMP_42]]
  // CHECK: %[[TMP_80:.*]] = stablehlo.constant dense<86.707214088598973>
  // CHECK: %[[TMP_81:.*]] = stablehlo.add %[[TMP_79]], %[[TMP_80]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.multiply %[[TMP_81]], %[[TMP_42]]
  // CHECK: %[[TMP_83:.*]] = stablehlo.constant dense<354.93777888781989>
  // CHECK: %[[TMP_84:.*]] = stablehlo.add %[[TMP_82]], %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_42]]
  // CHECK: %[[TMP_86:.*]] = stablehlo.constant dense<975.70850174320549>
  // CHECK: %[[TMP_87:.*]] = stablehlo.add %[[TMP_85]], %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = stablehlo.multiply %[[TMP_87]], %[[TMP_42]]
  // CHECK: %[[TMP_89:.*]] = stablehlo.constant dense<1823.9091668790973>
  // CHECK: %[[TMP_90:.*]] = stablehlo.add %[[TMP_88]], %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = stablehlo.multiply %[[TMP_90]], %[[TMP_42]]
  // CHECK: %[[TMP_92:.*]] = stablehlo.constant dense<2246.3376081871097>
  // CHECK: %[[TMP_93:.*]] = stablehlo.add %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.multiply %[[TMP_93]], %[[TMP_42]]
  // CHECK: %[[TMP_95:.*]] = stablehlo.constant dense<1656.6630919416134>
  // CHECK: %[[TMP_96:.*]] = stablehlo.add %[[TMP_94]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.multiply %[[TMP_96]], %[[TMP_42]]
  // CHECK: %[[TMP_98:.*]] = stablehlo.constant dense<557.53534081772773>
  // CHECK: %[[TMP_99:.*]] = stablehlo.add %[[TMP_97]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.divide %[[TMP_71]], %[[TMP_99]]
  // CHECK: %[[TMP_103:.*]] = stablehlo.constant dense<0.56418958354775506>
  // CHECK: %[[TMP_105:.*]] = stablehlo.multiply %[[TMP_103]], %[[TMP_42]]
  // CHECK: %[[TMP_106:.*]] = stablehlo.constant dense<1.275366707599781>
  // CHECK: %[[TMP_107:.*]] = stablehlo.add %[[TMP_105]], %[[TMP_106]]
  // CHECK: %[[TMP_108:.*]] = stablehlo.multiply %[[TMP_107]], %[[TMP_42]]
  // CHECK: %[[TMP_109:.*]] = stablehlo.constant dense<5.0190504225118051>
  // CHECK: %[[TMP_110:.*]] = stablehlo.add %[[TMP_108]], %[[TMP_109]]
  // CHECK: %[[TMP_111:.*]] = stablehlo.multiply %[[TMP_110]], %[[TMP_42]]
  // CHECK: %[[TMP_112:.*]] = stablehlo.constant dense<6.160210979930536>
  // CHECK: %[[TMP_113:.*]] = stablehlo.add %[[TMP_111]], %[[TMP_112]]
  // CHECK: %[[TMP_114:.*]] = stablehlo.multiply %[[TMP_113]], %[[TMP_42]]
  // CHECK: %[[TMP_115:.*]] = stablehlo.constant dense<7.4097426995044895>
  // CHECK: %[[TMP_116:.*]] = stablehlo.add %[[TMP_114]], %[[TMP_115]]
  // CHECK: %[[TMP_117:.*]] = stablehlo.multiply %[[TMP_116]], %[[TMP_42]]
  // CHECK: %[[TMP_118:.*]] = stablehlo.constant dense<2.9788666537210022>
  // CHECK: %[[TMP_119:.*]] = stablehlo.add %[[TMP_117]], %[[TMP_118]]
  // CHECK: %[[TMP_120:.*]] = stablehlo.multiply %[[TMP_41]], %[[TMP_119]]
  // CHECK: %[[TMP_123:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_125:.*]] = stablehlo.multiply %[[TMP_123]], %[[TMP_42]]
  // CHECK: %[[TMP_126:.*]] = stablehlo.constant dense<2.2605286322011726>
  // CHECK: %[[TMP_127:.*]] = stablehlo.add %[[TMP_125]], %[[TMP_126]]
  // CHECK: %[[TMP_128:.*]] = stablehlo.multiply %[[TMP_127]], %[[TMP_42]]
  // CHECK: %[[TMP_129:.*]] = stablehlo.constant dense<9.3960352493800147>
  // CHECK: %[[TMP_130:.*]] = stablehlo.add %[[TMP_128]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = stablehlo.multiply %[[TMP_130]], %[[TMP_42]]
  // CHECK: %[[TMP_132:.*]] = stablehlo.constant dense<12.048953980809666>
  // CHECK: %[[TMP_133:.*]] = stablehlo.add %[[TMP_131]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = stablehlo.multiply %[[TMP_133]], %[[TMP_42]]
  // CHECK: %[[TMP_135:.*]] = stablehlo.constant dense<17.081445074756591>
  // CHECK: %[[TMP_136:.*]] = stablehlo.add %[[TMP_134]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = stablehlo.multiply %[[TMP_136]], %[[TMP_42]]
  // CHECK: %[[TMP_138:.*]] = stablehlo.constant dense<9.6089680906328585>
  // CHECK: %[[TMP_139:.*]] = stablehlo.add %[[TMP_137]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = stablehlo.multiply %[[TMP_139]], %[[TMP_42]]
  // CHECK: %[[TMP_141:.*]] = stablehlo.constant dense<3.3690764510008151>
  // CHECK: %[[TMP_142:.*]] = stablehlo.add %[[TMP_140]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = stablehlo.divide %[[TMP_120]], %[[TMP_142]]
  // CHECK: %[[TMP_144:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_145:.*]] = stablehlo.compare LT, %[[TMP_42]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = stablehlo.select %[[TMP_145]], %[[TMP_100]], %[[TMP_143]]
  // CHECK: %[[TMP_147:.*]] = stablehlo.constant dense<-709.78271289338397>
  // CHECK: %[[TMP_148:.*]] = stablehlo.compare LT, %[[TMP_40]], %[[TMP_147]]
  // CHECK: %[[TMP_149:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_150:.*]] = stablehlo.select %[[TMP_148]], %[[TMP_149]], %[[TMP_146]]
  // CHECK: %[[TMP_152:.*]] = stablehlo.compare LT, %[[ARG]], %[[TMP_149]]
  // CHECK: %[[TMP_153:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_154:.*]] = stablehlo.subtract %[[TMP_153]], %[[TMP_150]]
  // CHECK: %[[TMP_155:.*]] = stablehlo.select %[[TMP_152]], %[[TMP_154]], %[[TMP_150]]
  // CHECK: %[[TMP_156:.*]] = stablehlo.subtract %[[TMP_38]], %[[TMP_155]]
  // CHECK: %[[TMP_157:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_159:.*]] = stablehlo.compare LT, %[[TMP_157]], %[[TMP_38]]
  // CHECK: %[[RESULT:.*]] = stablehlo.select %[[TMP_159]], %[[TMP_37]], %[[TMP_156]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erf_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erf_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK-DAG: %[[TMP_0:.*]] = stablehlo.constant dense<-4.000000e+00>
  // CHECK-DAG: %[[TMP_1:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_2:.*]] = stablehlo.clamp %[[TMP_0]], %[[ARG]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.multiply %[[TMP_2]], %[[TMP_2]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.constant dense<-2.72614237E-10>
  // CHECK: %[[TMP_8:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_3]]
  // CHECK: %[[TMP_9:.*]] = stablehlo.constant dense<2.77068146E-8>
  // CHECK: %[[TMP_10:.*]] = stablehlo.add %[[TMP_8]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.multiply %[[TMP_10]], %[[TMP_3]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.constant dense<-2.10102394E-6>
  // CHECK: %[[TMP_13:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = stablehlo.multiply %[[TMP_13]], %[[TMP_3]]
  // CHECK: %[[TMP_15:.*]] = stablehlo.constant dense<-5.69250624E-5>
  // CHECK: %[[TMP_16:.*]] = stablehlo.add %[[TMP_14]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.multiply %[[TMP_16]], %[[TMP_3]]
  // CHECK: %[[TMP_18:.*]] = stablehlo.constant dense<-7.34990637E-4>
  // CHECK: %[[TMP_19:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = stablehlo.multiply %[[TMP_19]], %[[TMP_3]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.constant dense<-2.954600e-03>
  // CHECK: %[[TMP_22:.*]] = stablehlo.add %[[TMP_20]], %[[TMP_21]]
  // CHECK: %[[TMP_23:.*]] = stablehlo.multiply %[[TMP_22]], %[[TMP_3]]
  // CHECK: %[[TMP_24:.*]] = stablehlo.constant dense<-0.0160960332>
  // CHECK: %[[TMP_25:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_24]]
  // CHECK: %[[TMP_28:.*]] = stablehlo.constant dense<-1.45660715E-5>
  // CHECK: %[[TMP_30:.*]] = stablehlo.multiply %[[TMP_28]], %[[TMP_3]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.constant dense<-2.13374049E-4>
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_30]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = stablehlo.multiply %[[TMP_32]], %[[TMP_3]]
  // CHECK: %[[TMP_34:.*]] = stablehlo.constant dense<-0.00168282702>
  // CHECK: %[[TMP_35:.*]] = stablehlo.add %[[TMP_33]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.multiply %[[TMP_35]], %[[TMP_3]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.constant dense<-0.00737332925>
  // CHECK: %[[TMP_38:.*]] = stablehlo.add %[[TMP_36]], %[[TMP_37]]
  // CHECK: %[[TMP_39:.*]] = stablehlo.multiply %[[TMP_38]], %[[TMP_3]]
  // CHECK: %[[TMP_40:.*]] = stablehlo.constant dense<-0.0142647391>
  // CHECK: %[[TMP_41:.*]] = stablehlo.add %[[TMP_39]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = stablehlo.multiply %[[TMP_2]], %[[TMP_25]]
  // CHECK: %[[TMP_43:.*]] = stablehlo.divide %[[TMP_42]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_44:.*]] = stablehlo.constant dense<-1.000000e+00>
  // CHECK-DAG: %[[TMP_45:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[RESULT:.*]] = stablehlo.clamp %[[TMP_44]], %[[TMP_43]], %[[TMP_45]]
  %1 = "chlo.erf"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erf_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erf_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erf_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erf_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // CHECK: stablehlo.convert %[[ARG]] : (tensor<bf16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<bf16>
  %1 = "chlo.erf"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @acosh
// CHECK-SAME:   %[[TMP_arg0:.*]]: tensor<f16>) -> tensor<f16>
func.func @acosh(%arg: tensor<f16>) -> tensor<f16> {
  // CHECK:   %[[TMP_0:.*]] = stablehlo.constant dense<6.550400e+04> : tensor<f16>
  // CHECK:   %[[TMP_1:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f16>
  // CHECK:   %[[TMP_2:.*]] = stablehlo.divide %[[TMP_0]], %[[TMP_1]] : tensor<f16>
  // CHECK:   %[[TMP_3:.*]] = stablehlo.compare  GE, %[[TMP_arg0]], %[[TMP_2]] : (tensor<f16>, tensor<f16>) -> tensor<i1>
  // CHECK:   %[[TMP_4:.*]] = stablehlo.log %[[TMP_1]] : tensor<f16>
  // CHECK:   %[[TMP_5:.*]] = stablehlo.log %[[TMP_arg0]] : tensor<f16>
  // CHECK:   %[[TMP_6:.*]] = stablehlo.add %[[TMP_4]], %[[TMP_5]] : tensor<f16>
  // CHECK:   %[[TMP_7:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f16>
  // CHECK:   %[[TMP_8:.*]] = stablehlo.subtract %[[TMP_arg0]], %[[TMP_7]] : tensor<f16>
  // CHECK:   %[[TMP_9:.*]] = stablehlo.sqrt %[[TMP_8]] : tensor<f16>
  // CHECK:   %[[TMP_10:.*]] = stablehlo.add %[[TMP_arg0]], %[[TMP_7]] : tensor<f16>
  // CHECK:   %[[TMP_11:.*]] = stablehlo.sqrt %[[TMP_10]] : tensor<f16>
  // CHECK:   %[[TMP_12:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_9]] : tensor<f16>
  // CHECK:   %[[TMP_13:.*]] = stablehlo.multiply %[[TMP_9]], %[[TMP_12]] : tensor<f16>
  // CHECK:   %[[TMP_14:.*]] = stablehlo.log_plus_one %[[TMP_13]] : tensor<f16>
  // CHECK:   %[[TMP_15:.*]] = stablehlo.select %[[TMP_3]], %[[TMP_6]], %[[TMP_14]] : tensor<i1>, tensor<f16>
  // CHECK:   return %[[TMP_15]] : tensor<f16>
  %1 = "chlo.acosh"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL:   func.func @acosh_complex_f32(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.abs %[[VAL_1]] : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.abs %[[VAL_3]] : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.maximum %[[VAL_2]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.sqrt %[[VAL_6]] : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.divide %[[VAL_7]], %[[VAL_8]] : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.compare  GE, %[[VAL_5]], %[[VAL_9]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.compare  LE, %[[VAL_2]], %[[VAL_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.add %[[VAL_2]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.abs %[[VAL_14]] : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.maximum %[[VAL_15]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.minimum %[[VAL_15]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.compare  EQ, %[[VAL_16]], %[[VAL_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_19:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.sqrt %[[VAL_19]] : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.multiply %[[VAL_20]], %[[VAL_16]] : tensor<f32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.divide %[[VAL_17]], %[[VAL_16]] : tensor<f32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_22]] : tensor<f32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_23]] : tensor<f32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.sqrt %[[VAL_24]] : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.compare  EQ, %[[VAL_25]], %[[VAL_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_27:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_28:.*]] = stablehlo.compare  GT, %[[VAL_23]], %[[VAL_27]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_29:.*]] = stablehlo.and %[[VAL_26]], %[[VAL_28]] : tensor<i1>
// CHECK:           %[[VAL_30:.*]] = stablehlo.multiply %[[VAL_16]], %[[VAL_23]] : tensor<f32>
// CHECK:           %[[VAL_31:.*]] = stablehlo.divide %[[VAL_30]], %[[VAL_19]] : tensor<f32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.add %[[VAL_16]], %[[VAL_31]] : tensor<f32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.multiply %[[VAL_16]], %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.select %[[VAL_29]], %[[VAL_32]], %[[VAL_33]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_35:.*]] = stablehlo.select %[[VAL_18]], %[[VAL_21]], %[[VAL_34]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.subtract %[[VAL_2]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_37:.*]] = stablehlo.abs %[[VAL_36]] : tensor<f32>
// CHECK:           %[[VAL_38:.*]] = stablehlo.maximum %[[VAL_37]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_39:.*]] = stablehlo.minimum %[[VAL_37]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_40:.*]] = stablehlo.compare  EQ, %[[VAL_38]], %[[VAL_39]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_41:.*]] = stablehlo.multiply %[[VAL_20]], %[[VAL_38]] : tensor<f32>
// CHECK:           %[[VAL_42:.*]] = stablehlo.divide %[[VAL_39]], %[[VAL_38]] : tensor<f32>
// CHECK:           %[[VAL_43:.*]] = stablehlo.multiply %[[VAL_42]], %[[VAL_42]] : tensor<f32>
// CHECK:           %[[VAL_44:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_43]] : tensor<f32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.sqrt %[[VAL_44]] : tensor<f32>
// CHECK:           %[[VAL_46:.*]] = stablehlo.compare  EQ, %[[VAL_45]], %[[VAL_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_47:.*]] = stablehlo.compare  GT, %[[VAL_43]], %[[VAL_27]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_48:.*]] = stablehlo.and %[[VAL_46]], %[[VAL_47]] : tensor<i1>
// CHECK:           %[[VAL_49:.*]] = stablehlo.multiply %[[VAL_38]], %[[VAL_43]] : tensor<f32>
// CHECK:           %[[VAL_50:.*]] = stablehlo.divide %[[VAL_49]], %[[VAL_19]] : tensor<f32>
// CHECK:           %[[VAL_51:.*]] = stablehlo.add %[[VAL_38]], %[[VAL_50]] : tensor<f32>
// CHECK:           %[[VAL_52:.*]] = stablehlo.multiply %[[VAL_38]], %[[VAL_45]] : tensor<f32>
// CHECK:           %[[VAL_53:.*]] = stablehlo.select %[[VAL_48]], %[[VAL_51]], %[[VAL_52]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_54:.*]] = stablehlo.select %[[VAL_40]], %[[VAL_41]], %[[VAL_53]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_55:.*]] = stablehlo.add %[[VAL_35]], %[[VAL_54]] : tensor<f32>
// CHECK:           %[[VAL_56:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_55]] : tensor<f32>
// CHECK:           %[[VAL_57:.*]] = stablehlo.add %[[VAL_56]], %[[VAL_2]] : tensor<f32>
// CHECK:           %[[VAL_58:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_57]] : tensor<f32>
// CHECK:           %[[VAL_59:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_60:.*]] = stablehlo.add %[[VAL_35]], %[[VAL_14]] : tensor<f32>
// CHECK:           %[[VAL_61:.*]] = stablehlo.divide %[[VAL_59]], %[[VAL_60]] : tensor<f32>
// CHECK:           %[[VAL_62:.*]] = stablehlo.subtract %[[VAL_54]], %[[VAL_36]] : tensor<f32>
// CHECK:           %[[VAL_63:.*]] = stablehlo.add %[[VAL_61]], %[[VAL_62]] : tensor<f32>
// CHECK:           %[[VAL_64:.*]] = stablehlo.multiply %[[VAL_58]], %[[VAL_63]] : tensor<f32>
// CHECK:           %[[VAL_65:.*]] = stablehlo.sqrt %[[VAL_64]] : tensor<f32>
// CHECK:           %[[VAL_66:.*]] = stablehlo.divide %[[VAL_58]], %[[VAL_60]] : tensor<f32>
// CHECK:           %[[VAL_67:.*]] = stablehlo.add %[[VAL_54]], %[[VAL_36]] : tensor<f32>
// CHECK:           %[[VAL_68:.*]] = stablehlo.divide %[[VAL_58]], %[[VAL_67]] : tensor<f32>
// CHECK:           %[[VAL_69:.*]] = stablehlo.add %[[VAL_66]], %[[VAL_68]] : tensor<f32>
// CHECK:           %[[VAL_70:.*]] = stablehlo.sqrt %[[VAL_69]] : tensor<f32>
// CHECK:           %[[VAL_71:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_70]] : tensor<f32>
// CHECK:           %[[VAL_72:.*]] = stablehlo.select %[[VAL_12]], %[[VAL_65]], %[[VAL_71]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_73:.*]] = stablehlo.select %[[VAL_10]], %[[VAL_4]], %[[VAL_72]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_74:.*]] = stablehlo.constant dense<9.99999995E+11> : tensor<f32>
// CHECK:           %[[VAL_75:.*]] = stablehlo.multiply %[[VAL_9]], %[[VAL_74]] : tensor<f32>
// CHECK:           %[[VAL_76:.*]] = stablehlo.compare  LT, %[[VAL_2]], %[[VAL_75]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_77:.*]] = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
// CHECK:           %[[VAL_78:.*]] = stablehlo.multiply %[[VAL_9]], %[[VAL_77]] : tensor<f32>
// CHECK:           %[[VAL_79:.*]] = stablehlo.constant dense<1.000000e+02> : tensor<f32>
// CHECK:           %[[VAL_80:.*]] = stablehlo.multiply %[[VAL_9]], %[[VAL_79]] : tensor<f32>
// CHECK:           %[[VAL_81:.*]] = stablehlo.select %[[VAL_76]], %[[VAL_78]], %[[VAL_80]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_82:.*]] = stablehlo.compare  GE, %[[VAL_4]], %[[VAL_81]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_83:.*]] = stablehlo.select %[[VAL_82]], %[[VAL_4]], %[[VAL_2]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_84:.*]] = stablehlo.select %[[VAL_82]], %[[VAL_81]], %[[VAL_9]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_85:.*]] = stablehlo.compare  GE, %[[VAL_83]], %[[VAL_84]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_86:.*]] = stablehlo.log %[[VAL_19]] : tensor<f32>
// CHECK:           %[[VAL_87:.*]] = stablehlo.log %[[VAL_83]] : tensor<f32>
// CHECK:           %[[VAL_88:.*]] = stablehlo.add %[[VAL_86]], %[[VAL_87]] : tensor<f32>
// CHECK:           %[[VAL_89:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           %[[VAL_90:.*]] = stablehlo.compare  EQ, %[[VAL_4]], %[[VAL_89]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_91:.*]] = stablehlo.not %[[VAL_90]] : tensor<i1>
// CHECK:           %[[VAL_92:.*]] = stablehlo.and %[[VAL_82]], %[[VAL_91]] : tensor<i1>
// CHECK:           %[[VAL_93:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_94:.*]] = stablehlo.select %[[VAL_92]], %[[VAL_93]], %[[VAL_27]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_95:.*]] = stablehlo.multiply %[[VAL_94]], %[[VAL_94]] : tensor<f32>
// CHECK:           %[[VAL_96:.*]] = stablehlo.log_plus_one %[[VAL_95]] : tensor<f32>
// CHECK:           %[[VAL_97:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_96]] : tensor<f32>
// CHECK:           %[[VAL_98:.*]] = stablehlo.add %[[VAL_88]], %[[VAL_97]] : tensor<f32>
// CHECK:           %[[VAL_99:.*]] = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
// CHECK:           %[[VAL_100:.*]] = stablehlo.sqrt %[[VAL_99]] : tensor<f32>
// CHECK:           %[[VAL_101:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_102:.*]] = stablehlo.multiply %[[VAL_100]], %[[VAL_101]] : tensor<f32>
// CHECK:           %[[VAL_103:.*]] = stablehlo.compare  LT, %[[VAL_4]], %[[VAL_102]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_104:.*]] = stablehlo.compare  LT, %[[VAL_2]], %[[VAL_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_105:.*]] = stablehlo.and %[[VAL_103]], %[[VAL_104]] : tensor<i1>
// CHECK:           %[[VAL_106:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_36]] : tensor<f32>
// CHECK:           %[[VAL_107:.*]] = stablehlo.add %[[VAL_56]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_108:.*]] = stablehlo.divide %[[VAL_106]], %[[VAL_107]] : tensor<f32>
// CHECK:           %[[VAL_109:.*]] = stablehlo.negate %[[VAL_108]] : tensor<f32>
// CHECK:           %[[VAL_110:.*]] = stablehlo.compare  GE, %[[VAL_2]], %[[VAL_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_111:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_59]] : tensor<f32>
// CHECK:           %[[VAL_112:.*]] = stablehlo.divide %[[VAL_111]], %[[VAL_60]] : tensor<f32>
// CHECK:           %[[VAL_113:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_67]] : tensor<f32>
// CHECK:           %[[VAL_114:.*]] = stablehlo.add %[[VAL_112]], %[[VAL_113]] : tensor<f32>
// CHECK:           %[[VAL_115:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK:           %[[VAL_116:.*]] = stablehlo.compare  LE, %[[VAL_56]], %[[VAL_115]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_117:.*]] = stablehlo.divide %[[VAL_111]], %[[VAL_62]] : tensor<f32>
// CHECK:           %[[VAL_118:.*]] = stablehlo.add %[[VAL_112]], %[[VAL_117]] : tensor<f32>
// CHECK:           %[[VAL_119:.*]] = stablehlo.subtract %[[VAL_56]], %[[VAL_11]] : tensor<f32>
// CHECK:           %[[VAL_120:.*]] = stablehlo.select %[[VAL_116]], %[[VAL_118]], %[[VAL_119]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_121:.*]] = stablehlo.select %[[VAL_110]], %[[VAL_114]], %[[VAL_120]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_122:.*]] = stablehlo.select %[[VAL_105]], %[[VAL_109]], %[[VAL_121]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_123:.*]] = stablehlo.multiply %[[VAL_122]], %[[VAL_107]] : tensor<f32>
// CHECK:           %[[VAL_124:.*]] = stablehlo.sqrt %[[VAL_123]] : tensor<f32>
// CHECK:           %[[VAL_125:.*]] = stablehlo.divide %[[VAL_4]], %[[VAL_124]] : tensor<f32>
// CHECK:           %[[VAL_126:.*]] = stablehlo.add %[[VAL_122]], %[[VAL_124]] : tensor<f32>
// CHECK:           %[[VAL_127:.*]] = stablehlo.log_plus_one %[[VAL_126]] : tensor<f32>
// CHECK:           %[[VAL_128:.*]] = stablehlo.select %[[VAL_105]], %[[VAL_125]], %[[VAL_127]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_129:.*]] = stablehlo.select %[[VAL_85]], %[[VAL_98]], %[[VAL_128]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_130:.*]] = stablehlo.complex %[[VAL_73]], %[[VAL_129]] : tensor<complex<f32>>
// CHECK:           %[[VAL_131:.*]] = stablehlo.imag %[[VAL_130]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_132:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_133:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_134:.*]] = stablehlo.compare  LT, %[[VAL_132]], %[[VAL_133]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_135:.*]] = stablehlo.real %[[VAL_130]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_136:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_137:.*]] = stablehlo.atan2 %[[VAL_135]], %[[VAL_136]] : tensor<f32>
// CHECK:           %[[VAL_138:.*]] = stablehlo.negate %[[VAL_137]] : tensor<f32>
// CHECK:           %[[VAL_139:.*]] = stablehlo.select %[[VAL_134]], %[[VAL_138]], %[[VAL_137]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_140:.*]] = stablehlo.complex %[[VAL_131]], %[[VAL_139]] : tensor<complex<f32>>
// CHECK:           return %[[VAL_140]] : tensor<complex<f32>>
// CHECK:         }
func.func @acosh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.acosh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @erfc_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erfc_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK-NEXT: %[[TMP_0:.*]] = stablehlo.multiply %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[TMP_1:.*]] = stablehlo.negate %[[TMP_0]]
  // CHECK-NEXT: %[[TMP_2:.*]] = stablehlo.exponential %[[TMP_1]]
  // CHECK-NEXT: %[[TMP_3:.*]] = stablehlo.abs %[[ARG]]
  // CHECK-NEXT: %[[TMP_6:.*]] = stablehlo.constant dense<2.4619698147353052E-10>
  // CHECK-NEXT: %[[TMP_8:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_9:.*]] = stablehlo.constant dense<0.56418956483106886>
  // CHECK-NEXT: %[[TMP_10:.*]] = stablehlo.add %[[TMP_8]], %[[TMP_9]]
  // CHECK-NEXT: %[[TMP_11:.*]] = stablehlo.multiply %[[TMP_10]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_12:.*]] = stablehlo.constant dense<7.4632105644226989>
  // CHECK-NEXT: %[[TMP_13:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_12]]
  // CHECK-NEXT: %[[TMP_14:.*]] = stablehlo.multiply %[[TMP_13]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_15:.*]] = stablehlo.constant dense<48.637197098568137>
  // CHECK-NEXT: %[[TMP_16:.*]] = stablehlo.add %[[TMP_14]], %[[TMP_15]]
  // CHECK-NEXT: %[[TMP_17:.*]] = stablehlo.multiply %[[TMP_16]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_18:.*]] = stablehlo.constant dense<196.5208329560771>
  // CHECK-NEXT: %[[TMP_19:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_18]]
  // CHECK-NEXT: %[[TMP_20:.*]] = stablehlo.multiply %[[TMP_19]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_21:.*]] = stablehlo.constant dense<526.44519499547732>
  // CHECK-NEXT: %[[TMP_22:.*]] = stablehlo.add %[[TMP_20]], %[[TMP_21]]
  // CHECK-NEXT: %[[TMP_23:.*]] = stablehlo.multiply %[[TMP_22]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_24:.*]] = stablehlo.constant dense<934.52852717195765>
  // CHECK-NEXT: %[[TMP_25:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_24]]
  // CHECK-NEXT: %[[TMP_26:.*]] = stablehlo.multiply %[[TMP_25]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_27:.*]] = stablehlo.constant dense<1027.5518868951572>
  // CHECK-NEXT: %[[TMP_28:.*]] = stablehlo.add %[[TMP_26]], %[[TMP_27]]
  // CHECK-NEXT: %[[TMP_29:.*]] = stablehlo.multiply %[[TMP_28]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_30:.*]] = stablehlo.constant dense<557.53533536939938>
  // CHECK-NEXT: %[[TMP_31:.*]] = stablehlo.add %[[TMP_29]], %[[TMP_30]]
  // CHECK-NEXT: %[[TMP_32:.*]] = stablehlo.multiply %[[TMP_2]], %[[TMP_31]]
  // CHECK-NEXT: %[[TMP_35:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_37:.*]] = stablehlo.multiply %[[TMP_35]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_38:.*]] = stablehlo.constant dense<13.228195115474499>
  // CHECK-NEXT: %[[TMP_39:.*]] = stablehlo.add %[[TMP_37]], %[[TMP_38]]
  // CHECK-NEXT: %[[TMP_40:.*]] = stablehlo.multiply %[[TMP_39]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_41:.*]] = stablehlo.constant dense<86.707214088598973>
  // CHECK-NEXT: %[[TMP_42:.*]] = stablehlo.add %[[TMP_40]], %[[TMP_41]]
  // CHECK-NEXT: %[[TMP_43:.*]] = stablehlo.multiply %[[TMP_42]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_44:.*]] = stablehlo.constant dense<354.93777888781989>
  // CHECK-NEXT: %[[TMP_45:.*]] = stablehlo.add %[[TMP_43]], %[[TMP_44]]
  // CHECK-NEXT: %[[TMP_46:.*]] = stablehlo.multiply %[[TMP_45]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_47:.*]] = stablehlo.constant dense<975.70850174320549>
  // CHECK-NEXT: %[[TMP_48:.*]] = stablehlo.add %[[TMP_46]], %[[TMP_47]]
  // CHECK-NEXT: %[[TMP_49:.*]] = stablehlo.multiply %[[TMP_48]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_50:.*]] = stablehlo.constant dense<1823.9091668790973>
  // CHECK-NEXT: %[[TMP_51:.*]] = stablehlo.add %[[TMP_49]], %[[TMP_50]]
  // CHECK-NEXT: %[[TMP_52:.*]] = stablehlo.multiply %[[TMP_51]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_53:.*]] = stablehlo.constant dense<2246.3376081871097>
  // CHECK-NEXT: %[[TMP_54:.*]] = stablehlo.add %[[TMP_52]], %[[TMP_53]]
  // CHECK-NEXT: %[[TMP_55:.*]] = stablehlo.multiply %[[TMP_54]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_56:.*]] = stablehlo.constant dense<1656.6630919416134>
  // CHECK-NEXT: %[[TMP_57:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_56]]
  // CHECK-NEXT: %[[TMP_58:.*]] = stablehlo.multiply %[[TMP_57]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_59:.*]] = stablehlo.constant dense<557.53534081772773>
  // CHECK-NEXT: %[[TMP_60:.*]] = stablehlo.add %[[TMP_58]], %[[TMP_59]]
  // CHECK-NEXT: %[[TMP_61:.*]] = stablehlo.divide %[[TMP_32]], %[[TMP_60]]
  // CHECK-NEXT: %[[TMP_64:.*]] = stablehlo.constant dense<0.56418958354775506>
  // CHECK-NEXT: %[[TMP_66:.*]] = stablehlo.multiply %[[TMP_64]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_67:.*]] = stablehlo.constant dense<1.275366707599781>
  // CHECK-NEXT: %[[TMP_68:.*]] = stablehlo.add %[[TMP_66]], %[[TMP_67]]
  // CHECK-NEXT: %[[TMP_69:.*]] = stablehlo.multiply %[[TMP_68]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_70:.*]] = stablehlo.constant dense<5.0190504225118051>
  // CHECK-NEXT: %[[TMP_71:.*]] = stablehlo.add %[[TMP_69]], %[[TMP_70]]
  // CHECK-NEXT: %[[TMP_72:.*]] = stablehlo.multiply %[[TMP_71]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_73:.*]] = stablehlo.constant dense<6.160210979930536>
  // CHECK-NEXT: %[[TMP_74:.*]] = stablehlo.add %[[TMP_72]], %[[TMP_73]]
  // CHECK-NEXT: %[[TMP_75:.*]] = stablehlo.multiply %[[TMP_74]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_76:.*]] = stablehlo.constant dense<7.4097426995044895>
  // CHECK-NEXT: %[[TMP_77:.*]] = stablehlo.add %[[TMP_75]], %[[TMP_76]]
  // CHECK-NEXT: %[[TMP_78:.*]] = stablehlo.multiply %[[TMP_77]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_79:.*]] = stablehlo.constant dense<2.9788666537210022>
  // CHECK-NEXT: %[[TMP_80:.*]] = stablehlo.add %[[TMP_78]], %[[TMP_79]]
  // CHECK-NEXT: %[[TMP_81:.*]] = stablehlo.multiply %[[TMP_2]], %[[TMP_80]]
  // CHECK-NEXT: %[[TMP_84:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_86:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_87:.*]] = stablehlo.constant dense<2.2605286322011726>
  // CHECK-NEXT: %[[TMP_88:.*]] = stablehlo.add %[[TMP_86]], %[[TMP_87]]
  // CHECK-NEXT: %[[TMP_89:.*]] = stablehlo.multiply %[[TMP_88]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_90:.*]] = stablehlo.constant dense<9.3960352493800147>
  // CHECK-NEXT: %[[TMP_91:.*]] = stablehlo.add %[[TMP_89]], %[[TMP_90]]
  // CHECK-NEXT: %[[TMP_92:.*]] = stablehlo.multiply %[[TMP_91]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_93:.*]] = stablehlo.constant dense<12.048953980809666>
  // CHECK-NEXT: %[[TMP_94:.*]] = stablehlo.add %[[TMP_92]], %[[TMP_93]]
  // CHECK-NEXT: %[[TMP_95:.*]] = stablehlo.multiply %[[TMP_94]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_96:.*]] = stablehlo.constant dense<17.081445074756591>
  // CHECK-NEXT: %[[TMP_97:.*]] = stablehlo.add %[[TMP_95]], %[[TMP_96]]
  // CHECK-NEXT: %[[TMP_98:.*]] = stablehlo.multiply %[[TMP_97]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_99:.*]] = stablehlo.constant dense<9.6089680906328585>
  // CHECK-NEXT: %[[TMP_100:.*]] = stablehlo.add %[[TMP_98]], %[[TMP_99]]
  // CHECK-NEXT: %[[TMP_101:.*]] = stablehlo.multiply %[[TMP_100]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_102:.*]] = stablehlo.constant dense<3.3690764510008151>
  // CHECK-NEXT: %[[TMP_103:.*]] = stablehlo.add %[[TMP_101]], %[[TMP_102]]
  // CHECK-NEXT: %[[TMP_104:.*]] = stablehlo.divide %[[TMP_81]], %[[TMP_103]]
  // CHECK-NEXT: %[[TMP_105:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT: %[[TMP_106:.*]] = stablehlo.compare LT, %[[TMP_3]], %[[TMP_105]]
  // CHECK-NEXT: %[[TMP_107:.*]] = stablehlo.select %[[TMP_106]], %[[TMP_61]], %[[TMP_104]]
  // CHECK-NEXT: %[[TMP_108:.*]] = stablehlo.constant dense<-709.78271289338397>
  // CHECK-NEXT: %[[TMP_109:.*]] = stablehlo.compare LT, %[[TMP_1]], %[[TMP_108]]
  // CHECK-NEXT: %[[TMP_110:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_111:.*]] = stablehlo.select %[[TMP_109]], %[[TMP_110]], %[[TMP_107]]
  // CHECK-NEXT: %[[TMP_113:.*]] = stablehlo.compare LT, %[[ARG]], %[[TMP_110]]
  // CHECK-NEXT: %[[TMP_114:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK-NEXT: %[[TMP_115:.*]] = stablehlo.subtract %[[TMP_114]], %[[TMP_111]]
  // CHECK-NEXT: %[[TMP_116:.*]] = stablehlo.select %[[TMP_113]], %[[TMP_115]], %[[TMP_111]]
  // CHECK-NEXT: %[[TMP_117:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_118:.*]] = stablehlo.multiply %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[TMP_121:.*]] = stablehlo.constant dense<9.6049737398705161>
  // CHECK-NEXT: %[[TMP_123:.*]] = stablehlo.multiply %[[TMP_121]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_124:.*]] = stablehlo.constant dense<90.026019720384269>
  // CHECK-NEXT: %[[TMP_125:.*]] = stablehlo.add %[[TMP_123]], %[[TMP_124]]
  // CHECK-NEXT: %[[TMP_126:.*]] = stablehlo.multiply %[[TMP_125]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_127:.*]] = stablehlo.constant dense<2232.0053459468431>
  // CHECK-NEXT: %[[TMP_128:.*]] = stablehlo.add %[[TMP_126]], %[[TMP_127]]
  // CHECK-NEXT: %[[TMP_129:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_130:.*]] = stablehlo.constant dense<7003.3251411280507>
  // CHECK-NEXT: %[[TMP_131:.*]] = stablehlo.add %[[TMP_129]], %[[TMP_130]]
  // CHECK-NEXT: %[[TMP_132:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_133:.*]] = stablehlo.constant dense<55592.301301039493>
  // CHECK-NEXT: %[[TMP_134:.*]] = stablehlo.add %[[TMP_132]], %[[TMP_133]]
  // CHECK-NEXT: %[[TMP_135:.*]] = stablehlo.multiply %[[ARG]], %[[TMP_134]]
  // CHECK-NEXT: %[[TMP_138:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_140:.*]] = stablehlo.multiply %[[TMP_138]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_141:.*]] = stablehlo.constant dense<33.561714164750313>
  // CHECK-NEXT: %[[TMP_142:.*]] = stablehlo.add %[[TMP_140]], %[[TMP_141]]
  // CHECK-NEXT: %[[TMP_143:.*]] = stablehlo.multiply %[[TMP_142]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_144:.*]] = stablehlo.constant dense<521.35794978015269>
  // CHECK-NEXT: %[[TMP_145:.*]] = stablehlo.add %[[TMP_143]], %[[TMP_144]]
  // CHECK-NEXT: %[[TMP_146:.*]] = stablehlo.multiply %[[TMP_145]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_147:.*]] = stablehlo.constant dense<4594.3238297098014>
  // CHECK-NEXT: %[[TMP_148:.*]] = stablehlo.add %[[TMP_146]], %[[TMP_147]]
  // CHECK-NEXT: %[[TMP_149:.*]] = stablehlo.multiply %[[TMP_148]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_150:.*]] = stablehlo.constant dense<22629.000061389095>
  // CHECK-NEXT: %[[TMP_151:.*]] = stablehlo.add %[[TMP_149]], %[[TMP_150]]
  // CHECK-NEXT: %[[TMP_152:.*]] = stablehlo.multiply %[[TMP_151]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_153:.*]] = stablehlo.constant dense<49267.394260863592>
  // CHECK-NEXT: %[[TMP_154:.*]] = stablehlo.add %[[TMP_152]], %[[TMP_153]]
  // CHECK-NEXT: %[[TMP_155:.*]] = stablehlo.divide %[[TMP_135]], %[[TMP_154]]
  // CHECK-NEXT: %[[TMP_156:.*]] = stablehlo.subtract %[[TMP_117]], %[[TMP_155]]
  // CHECK-NEXT: %[[TMP_157:.*]] = stablehlo.abs %[[ARG]]
  // CHECK-NEXT: %[[TMP_159:.*]] = stablehlo.compare LT, %[[TMP_157]], %[[TMP_117]]
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.select %[[TMP_159]], %[[TMP_156]], %[[TMP_116]]
  // CHECK-NEXT: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erfc_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erfc_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = stablehlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_1:.*]] = stablehlo.negate %[[TMP_0]]
  // CHECK: %[[TMP_2:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = stablehlo.divide %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.exponential %[[TMP_1]]
  // CHECK: %[[TMP_7:.*]] = stablehlo.divide %[[TMP_3]], %[[TMP_2]]
  // CHECK: %[[TMP_8:.*]] = stablehlo.multiply %[[TMP_5]], %[[TMP_7]]
  // CHECK: %[[TMP_9:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_10:.*]] = stablehlo.compare LT, %[[TMP_2]], %[[TMP_9]]
  // CHECK: %[[TMP_13:.*]] = stablehlo.constant dense<2.326820e-02>
  // CHECK: %[[TMP_15:.*]] = stablehlo.multiply %[[TMP_13]], %[[TMP_4]]
  // CHECK: %[[TMP_16:.*]] = stablehlo.constant dense<-0.138703942>
  // CHECK: %[[TMP_17:.*]] = stablehlo.add %[[TMP_15]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = stablehlo.multiply %[[TMP_17]], %[[TMP_4]]
  // CHECK: %[[TMP_19:.*]] = stablehlo.constant dense<0.368742466>
  // CHECK: %[[TMP_20:.*]] = stablehlo.add %[[TMP_18]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.multiply %[[TMP_20]], %[[TMP_4]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.constant dense<-0.582473278>
  // CHECK: %[[TMP_23:.*]] = stablehlo.add %[[TMP_21]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = stablehlo.multiply %[[TMP_23]], %[[TMP_4]]
  // CHECK: %[[TMP_25:.*]] = stablehlo.constant dense<0.621000468>
  // CHECK: %[[TMP_26:.*]] = stablehlo.add %[[TMP_24]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.multiply %[[TMP_26]], %[[TMP_4]]
  // CHECK: %[[TMP_28:.*]] = stablehlo.constant dense<-0.494451523>
  // CHECK: %[[TMP_29:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = stablehlo.multiply %[[TMP_29]], %[[TMP_4]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.constant dense<3.404880e-01>
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_30]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = stablehlo.multiply %[[TMP_32]], %[[TMP_4]]
  // CHECK: %[[TMP_34:.*]] = stablehlo.constant dense<-0.274112701>
  // CHECK: %[[TMP_35:.*]] = stablehlo.add %[[TMP_33]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.multiply %[[TMP_35]], %[[TMP_4]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.constant dense<0.563825965>
  // CHECK: %[[TMP_38:.*]] = stablehlo.add %[[TMP_36]], %[[TMP_37]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.constant dense<-10.477664>
  // CHECK: %[[TMP_43:.*]] = stablehlo.multiply %[[TMP_41]], %[[TMP_4]]
  // CHECK: %[[TMP_44:.*]] = stablehlo.constant dense<1.297720e+01>
  // CHECK: %[[TMP_45:.*]] = stablehlo.add %[[TMP_43]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.multiply %[[TMP_45]], %[[TMP_4]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.constant dense<-7.49551868>
  // CHECK: %[[TMP_48:.*]] = stablehlo.add %[[TMP_46]], %[[TMP_47]]
  // CHECK: %[[TMP_49:.*]] = stablehlo.multiply %[[TMP_48]], %[[TMP_4]]
  // CHECK: %[[TMP_50:.*]] = stablehlo.constant dense<2.92101908>
  // CHECK: %[[TMP_51:.*]] = stablehlo.add %[[TMP_49]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.multiply %[[TMP_51]], %[[TMP_4]]
  // CHECK: %[[TMP_53:.*]] = stablehlo.constant dense<-1.01526523>
  // CHECK: %[[TMP_54:.*]] = stablehlo.add %[[TMP_52]], %[[TMP_53]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.multiply %[[TMP_54]], %[[TMP_4]]
  // CHECK: %[[TMP_56:.*]] = stablehlo.constant dense<0.42184633>
  // CHECK: %[[TMP_57:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = stablehlo.multiply %[[TMP_57]], %[[TMP_4]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.constant dense<-0.282076746>
  // CHECK: %[[TMP_60:.*]] = stablehlo.add %[[TMP_58]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.multiply %[[TMP_60]], %[[TMP_4]]
  // CHECK: %[[TMP_62:.*]] = stablehlo.constant dense<0.564189494>
  // CHECK: %[[TMP_63:.*]] = stablehlo.add %[[TMP_61]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = stablehlo.select %[[TMP_10]], %[[TMP_38]], %[[TMP_63]]
  // CHECK: %[[TMP_65:.*]] = stablehlo.multiply %[[TMP_8]], %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.constant dense<-88.7228394>
  // CHECK: %[[TMP_67:.*]] = stablehlo.compare LT, %[[TMP_1]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_69:.*]] = stablehlo.select %[[TMP_67]], %[[TMP_68]], %[[TMP_65]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.compare LT, %[[ARG]], %[[TMP_68]]
  // CHECK: %[[TMP_73:.*]] = stablehlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_74:.*]] = stablehlo.select %[[TMP_71]], %[[TMP_73]], %[[TMP_69]]
  // CHECK: %[[TMP_75:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_76:.*]] = stablehlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.constant dense<7.85386146E-5>
  // CHECK: %[[TMP_81:.*]] = stablehlo.multiply %[[TMP_79]], %[[TMP_76]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.constant dense<-8.0101937E-4>
  // CHECK: %[[TMP_83:.*]] = stablehlo.add %[[TMP_81]], %[[TMP_82]]
  // CHECK: %[[TMP_84:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_76]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.constant dense<0.00518832775>
  // CHECK: %[[TMP_86:.*]] = stablehlo.add %[[TMP_84]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = stablehlo.multiply %[[TMP_86]], %[[TMP_76]]
  // CHECK: %[[TMP_88:.*]] = stablehlo.constant dense<-0.0268538129>
  // CHECK: %[[TMP_89:.*]] = stablehlo.add %[[TMP_87]], %[[TMP_88]]
  // CHECK: %[[TMP_90:.*]] = stablehlo.multiply %[[TMP_89]], %[[TMP_76]]
  // CHECK: %[[TMP_91:.*]] = stablehlo.constant dense<0.112835854>
  // CHECK: %[[TMP_92:.*]] = stablehlo.add %[[TMP_90]], %[[TMP_91]]
  // CHECK: %[[TMP_93:.*]] = stablehlo.multiply %[[TMP_92]], %[[TMP_76]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.constant dense<-0.37612626>
  // CHECK: %[[TMP_95:.*]] = stablehlo.add %[[TMP_93]], %[[TMP_94]]
  // CHECK: %[[TMP_96:.*]] = stablehlo.multiply %[[TMP_95]], %[[TMP_76]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.constant dense<1.12837911>
  // CHECK: %[[TMP_98:.*]] = stablehlo.add %[[TMP_96]], %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = stablehlo.multiply %[[ARG]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.subtract %[[TMP_75]], %[[TMP_99]]
  // CHECK: %[[TMP_101:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_103:.*]] = stablehlo.compare LT, %[[TMP_101]], %[[TMP_75]]
  // CHECK: %[[RESULT:.*]] = stablehlo.select %[[TMP_103]], %[[TMP_100]], %[[TMP_74]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erfc_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erfc_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erfc_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erfc_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // CHECK: stablehlo.convert %[[ARG]] : (tensor<bf16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<bf16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @is_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[ABS:.*]] = stablehlo.abs %arg0 : tensor<f32>
  // CHECK: %[[POS_INF:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.compare EQ, %[[ABS]], %[[POS_INF]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @is_pos_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_pos_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[POS_INF:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.compare EQ, %[[ARG]], %[[POS_INF]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_pos_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @is_neg_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_neg_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.compare EQ, %[[ARG]], %[[NEG_INF]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_neg_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @lgamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func.func @lgamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_1:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_9:.*]] = stablehlo.compare LT, %[[ARG]], %[[TMP_1]]
  // CHECK: %[[TMP_10:.*]] = stablehlo.negate %[[ARG]]
  // CHECK: %[[TMP_2:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_11:.*]] = stablehlo.subtract %[[ARG]], %[[TMP_2]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.select %[[TMP_9]], %[[TMP_10]], %[[TMP_11]]
  // CHECK-DAG: %[[TMP_8:.*]] = stablehlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_13:.*]] = stablehlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_14:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = stablehlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.add %[[TMP_8]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = stablehlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_19:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = stablehlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_24:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = stablehlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = stablehlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_29:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = stablehlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_34:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = stablehlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_39:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = stablehlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = stablehlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_53:.*]] = stablehlo.add %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_7:.*]] = stablehlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_54:.*]] = stablehlo.divide %[[TMP_12]], %[[TMP_6]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.log_plus_one %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = stablehlo.add %[[TMP_7]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = stablehlo.divide %[[TMP_53]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_1]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.subtract %[[TMP_58]], %[[TMP_57]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.multiply %[[TMP_59]], %[[TMP_56]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.log %[[TMP_52]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.constant dense<0.91893853320467266>
  // CHECK: %[[TMP_62:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_60]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.add %[[TMP_62]], %[[TMP_61]]
  // CHECK: %[[TMP_64:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_65:.*]] = stablehlo.floor %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.subtract %[[TMP_64]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.compare LT, %[[TMP_1]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.subtract %[[TMP_2]], %[[TMP_66]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.select %[[TMP_67]], %[[TMP_68]], %[[TMP_66]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_70:.*]] = stablehlo.multiply %[[TMP_3]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.sine %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.log %[[TMP_71]]
  // CHECK: %[[TMP_4:.*]] = stablehlo.constant dense<1.1447298858494002>
  // CHECK: %[[TMP_75:.*]] = stablehlo.subtract %[[TMP_4]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.subtract %[[TMP_75]], %[[TMP_63]]
  // CHECK: %[[TMP_73:.*]] = stablehlo.is_finite %[[TMP_72]]
  // CHECK: %[[TMP_74:.*]] = stablehlo.negate %[[TMP_72]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.select %[[TMP_73]], %[[TMP_76]], %[[TMP_74]]
  // CHECK: %[[TMP_78:.*]] = stablehlo.select %[[TMP_9]], %[[TMP_77]], %[[TMP_63]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_80:.*]] = stablehlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_81:.*]] = stablehlo.compare EQ, %[[TMP_79]], %[[TMP_80]]
  // CHECK: %[[TMP_0:.*]] = stablehlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_82:.*]] = stablehlo.select %[[TMP_81]], %[[TMP_0]], %[[TMP_78]]
  // CHECK: return %[[TMP_82]]
  %1 = chlo.lgamma %arg : tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @lgamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @lgamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_1:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_9:.*]] = stablehlo.compare LT, %[[ARG]], %[[TMP_1]]
  // CHECK: %[[TMP_10:.*]] = stablehlo.negate %[[ARG]]
  // CHECK: %[[TMP_2:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_11:.*]] = stablehlo.subtract %[[ARG]], %[[TMP_2]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.select %[[TMP_9]], %[[TMP_10]], %[[TMP_11]]
  // CHECK-DAG: %[[TMP_8:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_13:.*]] = stablehlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_14:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = stablehlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.add %[[TMP_8]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = stablehlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_19:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = stablehlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_24:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = stablehlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = stablehlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_29:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = stablehlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_34:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = stablehlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_39:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = stablehlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = stablehlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_53:.*]] = stablehlo.add %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_7:.*]] = stablehlo.constant dense<2.01490307>
  // CHECK: %[[TMP_54:.*]] = stablehlo.divide %[[TMP_12]], %[[TMP_6]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.log_plus_one %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = stablehlo.add %[[TMP_7]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = stablehlo.divide %[[TMP_53]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_1]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.subtract %[[TMP_58]], %[[TMP_57]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.multiply %[[TMP_59]], %[[TMP_56]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.log %[[TMP_52]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.constant dense<0.918938517>
  // CHECK: %[[TMP_62:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_60]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.add %[[TMP_62]], %[[TMP_61]]
  // CHECK: %[[TMP_64:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_65:.*]] = stablehlo.floor %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.subtract %[[TMP_64]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.compare LT, %[[TMP_1]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.subtract %[[TMP_2]], %[[TMP_66]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.select %[[TMP_67]], %[[TMP_68]], %[[TMP_66]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.constant dense<3.14159274>
  // CHECK: %[[TMP_70:.*]] = stablehlo.multiply %[[TMP_3]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.sine %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.log %[[TMP_71]]
  // CHECK: %[[TMP_4:.*]] = stablehlo.constant dense<1.14472985>
  // CHECK: %[[TMP_75:.*]] = stablehlo.subtract %[[TMP_4]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.subtract %[[TMP_75]], %[[TMP_63]]
  // CHECK: %[[TMP_73:.*]] = stablehlo.is_finite %[[TMP_72]]
  // CHECK: %[[TMP_74:.*]] = stablehlo.negate %[[TMP_72]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.select %[[TMP_73]], %[[TMP_76]], %[[TMP_74]]
  // CHECK: %[[TMP_78:.*]] = stablehlo.select %[[TMP_9]], %[[TMP_77]], %[[TMP_63]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.abs %[[ARG]]
  // CHECK: %[[TMP_80:.*]] = stablehlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_81:.*]] = stablehlo.compare EQ, %[[TMP_79]], %[[TMP_80]]
  // CHECK: %[[TMP_0:.*]] = stablehlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_82:.*]] = stablehlo.select %[[TMP_81]], %[[TMP_0]], %[[TMP_78]]
  // CHECK: return %[[TMP_82]]
  %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @lgamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @lgamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.lgamma %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @digamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func.func @digamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_1:.*]] = stablehlo.compare LT, %arg0, %[[TMP_0]]
  // CHECK: %[[TMP_2:.*]] = stablehlo.negate %arg0
  // CHECK: %[[TMP_3:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = stablehlo.subtract %arg0, %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.select %[[TMP_1]], %[[TMP_2]], %[[TMP_4]]
  // CHECK-DAG: %[[TMP_6:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_7:.*]] = stablehlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_8:.*]] = stablehlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_9:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.multiply %[[TMP_10]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.divide %[[TMP_8]], %[[TMP_11]]
  // CHECK: %[[TMP_13:.*]] = stablehlo.subtract %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = stablehlo.divide %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_7]], %[[TMP_14]]
  // CHECK-DAG: %[[TMP_16:.*]] = stablehlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_17:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_18:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = stablehlo.multiply %[[TMP_18]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = stablehlo.divide %[[TMP_16]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.subtract %[[TMP_13]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.divide %[[TMP_16]], %[[TMP_18]]
  // CHECK: %[[TMP_23:.*]] = stablehlo.add %[[TMP_15]], %[[TMP_22]]
  // CHECK-DAG: %[[TMP_24:.*]] = stablehlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_25:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_26:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.multiply %[[TMP_26]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = stablehlo.divide %[[TMP_24]], %[[TMP_27]]
  // CHECK: %[[TMP_29:.*]] = stablehlo.subtract %[[TMP_21]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = stablehlo.divide %[[TMP_24]], %[[TMP_26]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_30]]
  // CHECK-DAG: %[[TMP_32:.*]] = stablehlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_33:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_34:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_33]]
  // CHECK: %[[TMP_35:.*]] = stablehlo.multiply %[[TMP_34]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.divide %[[TMP_32]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.subtract %[[TMP_29]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = stablehlo.divide %[[TMP_32]], %[[TMP_34]]
  // CHECK: %[[TMP_39:.*]] = stablehlo.add %[[TMP_31]], %[[TMP_38]]
  // CHECK-DAG: %[[TMP_40:.*]] = stablehlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_41:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = stablehlo.multiply %[[TMP_42]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = stablehlo.divide %[[TMP_40]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = stablehlo.subtract %[[TMP_37]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.divide %[[TMP_40]], %[[TMP_42]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_39]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = stablehlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_49:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_50:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.multiply %[[TMP_50]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = stablehlo.subtract %[[TMP_45]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_54]]
  // CHECK-DAG: %[[TMP_56:.*]] = stablehlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_57:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.multiply %[[TMP_58]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.divide %[[TMP_56]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.subtract %[[TMP_53]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = stablehlo.divide %[[TMP_56]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_62]]
  // CHECK-DAG: %[[TMP_64:.*]] = stablehlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_65:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_66:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.multiply %[[TMP_66]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.divide %[[TMP_64]], %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.subtract %[[TMP_61]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = stablehlo.divide %[[TMP_64]], %[[TMP_66]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.add %[[TMP_63]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_73:.*]] = stablehlo.add %[[TMP_72]], %[[TMP_5]]
  // CHECK: %[[TMP_74:.*]] = stablehlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_75:.*]] = stablehlo.divide %[[TMP_5]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.log_plus_one %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.add %[[TMP_74]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = stablehlo.divide %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_80:.*]] = stablehlo.divide %[[TMP_79]], %[[TMP_73]]
  // CHECK: %[[TMP_81:.*]] = stablehlo.add %[[TMP_77]], %[[TMP_78]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.subtract %[[TMP_81]], %[[TMP_80]]
  // CHECK: %[[TMP_83:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_84:.*]] = stablehlo.add %arg0, %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.floor %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = stablehlo.abs %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = stablehlo.add %arg0, %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = stablehlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_89:.*]] = stablehlo.multiply %[[TMP_88]], %[[TMP_87]]
  // CHECK: %[[TMP_90:.*]] = stablehlo.cosine %[[TMP_89]]
  // CHECK: %[[TMP_92:.*]] = stablehlo.sine %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = stablehlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_93:.*]] = stablehlo.divide %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.subtract %[[TMP_82]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = stablehlo.select %[[TMP_1]], %[[TMP_94]], %[[TMP_82]]
  // CHECK: %[[TMP_96:.*]] = stablehlo.compare LE, %arg0, %[[TMP_6]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.floor %arg0
  // CHECK: %[[TMP_98:.*]] = stablehlo.compare EQ, %arg0, %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = stablehlo.and %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[RES:.*]] = stablehlo.select %[[TMP_99]], %[[TMP_100]], %[[TMP_95]]
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @digamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @digamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_1:.*]] = stablehlo.compare LT, %arg0, %[[TMP_0]]
  // CHECK: %[[TMP_2:.*]] = stablehlo.negate %arg0
  // CHECK: %[[TMP_3:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = stablehlo.subtract %arg0, %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.select %[[TMP_1]], %[[TMP_2]], %[[TMP_4]]
  // CHECK-DAG: %[[TMP_6:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_7:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_8:.*]] = stablehlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_9:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.multiply %[[TMP_10]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.divide %[[TMP_8]], %[[TMP_11]]
  // CHECK: %[[TMP_13:.*]] = stablehlo.subtract %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = stablehlo.divide %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_7]], %[[TMP_14]]
  // CHECK-DAG: %[[TMP_16:.*]] = stablehlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_17:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_18:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = stablehlo.multiply %[[TMP_18]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = stablehlo.divide %[[TMP_16]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.subtract %[[TMP_13]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.divide %[[TMP_16]], %[[TMP_18]]
  // CHECK: %[[TMP_23:.*]] = stablehlo.add %[[TMP_15]], %[[TMP_22]]
  // CHECK-DAG: %[[TMP_24:.*]] = stablehlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_25:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_26:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.multiply %[[TMP_26]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = stablehlo.divide %[[TMP_24]], %[[TMP_27]]
  // CHECK: %[[TMP_29:.*]] = stablehlo.subtract %[[TMP_21]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = stablehlo.divide %[[TMP_24]], %[[TMP_26]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_30]]
  // CHECK-DAG: %[[TMP_32:.*]] = stablehlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_33:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_34:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_33]]
  // CHECK: %[[TMP_35:.*]] = stablehlo.multiply %[[TMP_34]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.divide %[[TMP_32]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.subtract %[[TMP_29]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = stablehlo.divide %[[TMP_32]], %[[TMP_34]]
  // CHECK: %[[TMP_39:.*]] = stablehlo.add %[[TMP_31]], %[[TMP_38]]
  // CHECK-DAG: %[[TMP_40:.*]] = stablehlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_41:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = stablehlo.multiply %[[TMP_42]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = stablehlo.divide %[[TMP_40]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = stablehlo.subtract %[[TMP_37]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.divide %[[TMP_40]], %[[TMP_42]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_39]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = stablehlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_49:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_50:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.multiply %[[TMP_50]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = stablehlo.subtract %[[TMP_45]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_54]]
  // CHECK-DAG: %[[TMP_56:.*]] = stablehlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_57:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.multiply %[[TMP_58]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.divide %[[TMP_56]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.subtract %[[TMP_53]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = stablehlo.divide %[[TMP_56]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_62]]
  // CHECK-DAG: %[[TMP_64:.*]] = stablehlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_65:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_66:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.multiply %[[TMP_66]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.divide %[[TMP_64]], %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.subtract %[[TMP_61]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = stablehlo.divide %[[TMP_64]], %[[TMP_66]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.add %[[TMP_63]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_73:.*]] = stablehlo.add %[[TMP_72]], %[[TMP_5]]
  // CHECK: %[[TMP_74:.*]] = stablehlo.constant dense<2.01490307>
  // CHECK: %[[TMP_75:.*]] = stablehlo.divide %[[TMP_5]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.log_plus_one %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.add %[[TMP_74]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = stablehlo.divide %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_80:.*]] = stablehlo.divide %[[TMP_79]], %[[TMP_73]]
  // CHECK: %[[TMP_81:.*]] = stablehlo.add %[[TMP_77]], %[[TMP_78]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.subtract %[[TMP_81]], %[[TMP_80]]
  // CHECK: %[[TMP_83:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_84:.*]] = stablehlo.add %arg0, %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.floor %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = stablehlo.abs %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = stablehlo.add %arg0, %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = stablehlo.constant dense<3.14159274>
  // CHECK: %[[TMP_89:.*]] = stablehlo.multiply %[[TMP_88]], %[[TMP_87]]
  // CHECK: %[[TMP_90:.*]] = stablehlo.cosine %[[TMP_89]]
  // CHECK: %[[TMP_92:.*]] = stablehlo.sine %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = stablehlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_93:.*]] = stablehlo.divide %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.subtract %[[TMP_82]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = stablehlo.select %[[TMP_1]], %[[TMP_94]], %[[TMP_82]]
  // CHECK: %[[TMP_96:.*]] = stablehlo.compare LE, %arg0, %[[TMP_6]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.floor %arg0
  // CHECK: %[[TMP_98:.*]] = stablehlo.compare EQ, %arg0, %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = stablehlo.and %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.constant dense<0x7FC00000>
  // CHECK: %[[RES:.*]] = stablehlo.select %[[TMP_99]], %[[TMP_100]], %[[TMP_95]]
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @digamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @digamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @zeta_f16
// CHECK-SAME:  (%[[X:.*]]: tensor<f16>, %[[Q:.*]]: tensor<f16>) -> tensor<f16>
func.func @zeta_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[TMP_0:.*]] = stablehlo.convert %[[X]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[TMP_1:.*]] = stablehlo.convert %[[Q]] : (tensor<f16>) -> tensor<f32>
  // CHECK-DAG: %[[TMP_2:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_3:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = stablehlo.negate %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.power %[[TMP_1]], %[[TMP_4]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.add %[[TMP_1]], %[[TMP_3]]
  // CHECK: %[[TMP_7:.*]] = stablehlo.power %[[TMP_6]], %[[TMP_4]]
  // CHECK: %[[TMP_8:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_7]]
  // CHECK: %[[TMP_9:.*]] = stablehlo.add %[[TMP_6]], %[[TMP_3]]
  // CHECK: %[[TMP_10:.*]] = stablehlo.power %[[TMP_9]], %[[TMP_4]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.add %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = stablehlo.add %[[TMP_9]], %[[TMP_3]]
  // CHECK: %[[TMP_13:.*]] = stablehlo.power %[[TMP_12]], %[[TMP_4]]
  // CHECK: %[[TMP_14:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_13]]
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_3]]
  // CHECK: %[[TMP_16:.*]] = stablehlo.power %[[TMP_15]], %[[TMP_4]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.add %[[TMP_14]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = stablehlo.add %[[TMP_15]], %[[TMP_3]]
  // CHECK: %[[TMP_19:.*]] = stablehlo.power %[[TMP_18]], %[[TMP_4]]
  // CHECK: %[[TMP_20:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.add %[[TMP_18]], %[[TMP_3]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.power %[[TMP_21]], %[[TMP_4]]
  // CHECK: %[[TMP_23:.*]] = stablehlo.add %[[TMP_20]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = stablehlo.add %[[TMP_21]], %[[TMP_3]]
  // CHECK: %[[TMP_25:.*]] = stablehlo.power %[[TMP_24]], %[[TMP_4]]
  // CHECK: %[[TMP_26:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.add %[[TMP_24]], %[[TMP_3]]
  // CHECK: %[[TMP_28:.*]] = stablehlo.power %[[TMP_27]], %[[TMP_4]]
  // CHECK: %[[TMP_29:.*]] = stablehlo.add %[[TMP_26]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_3]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.power %[[TMP_30]], %[[TMP_4]]
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_29]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = stablehlo.add %[[TMP_30]], %[[TMP_3]]
  // CHECK: %[[TMP_34:.*]] = stablehlo.power %[[TMP_33]], %[[TMP_4]]
  // CHECK: %[[TMP_35:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_36:.*]] = stablehlo.multiply %[[TMP_34]], %[[TMP_33]]
  // CHECK-DAG: %[[TMP_37:.*]] = stablehlo.subtract %[[TMP_0]], %[[TMP_35]]
  // CHECK: %[[TMP_38:.*]] = stablehlo.divide %[[TMP_36]], %[[TMP_37]]
  // CHECK: %[[TMP_39:.*]] = stablehlo.multiply %[[TMP_33]], %[[TMP_33]]
  // CHECK: %[[TMP_40:.*]] = stablehlo.divide %[[TMP_3]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = stablehlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_44:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = stablehlo.multiply %[[TMP_42]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.constant dense<-1.39544646E-19>
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_2]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_47]]
  // CHECK: %[[TMP_49:.*]] = stablehlo.multiply %[[TMP_45]], %[[TMP_48]]
  // CHECK: %[[TMP_50:.*]] = stablehlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_51:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_53:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = stablehlo.multiply %[[TMP_51]], %[[TMP_53]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.constant dense<5.50900303E-18>
  // CHECK: %[[TMP_56:.*]] = stablehlo.add %[[TMP_49]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = stablehlo.multiply %[[TMP_54]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_60:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_62:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_61]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.multiply %[[TMP_60]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = stablehlo.constant dense<-2.17486866E-16>
  // CHECK: %[[TMP_65:.*]] = stablehlo.add %[[TMP_58]], %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.multiply %[[TMP_63]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_69:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = stablehlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_71:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.multiply %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_73:.*]] = stablehlo.constant dense<8.58606213E-15>
  // CHECK: %[[TMP_74:.*]] = stablehlo.add %[[TMP_67]], %[[TMP_73]]
  // CHECK: %[[TMP_75:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.multiply %[[TMP_72]], %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_78:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_77]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_80:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_79]]
  // CHECK: %[[TMP_81:.*]] = stablehlo.multiply %[[TMP_78]], %[[TMP_80]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.constant dense<-3.3896803E-13>
  // CHECK: %[[TMP_83:.*]] = stablehlo.add %[[TMP_76]], %[[TMP_82]]
  // CHECK: %[[TMP_84:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.multiply %[[TMP_81]], %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = stablehlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_87:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = stablehlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_89:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_88]]
  // CHECK: %[[TMP_90:.*]] = stablehlo.multiply %[[TMP_87]], %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = stablehlo.constant dense<1.33825364E-11>
  // CHECK: %[[TMP_92:.*]] = stablehlo.add %[[TMP_85]], %[[TMP_91]]
  // CHECK: %[[TMP_93:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.multiply %[[TMP_90]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = stablehlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_96:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_98:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = stablehlo.multiply %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.constant dense<-5.28419031E-10>
  // CHECK: %[[TMP_101:.*]] = stablehlo.add %[[TMP_94]], %[[TMP_100]]
  // CHECK: %[[TMP_102:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = stablehlo.multiply %[[TMP_99]], %[[TMP_102]]
  // CHECK: %[[TMP_104:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_105:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_107:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_106]]
  // CHECK: %[[TMP_108:.*]] = stablehlo.multiply %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = stablehlo.constant dense<2.08767563E-8>
  // CHECK: %[[TMP_110:.*]] = stablehlo.add %[[TMP_103]], %[[TMP_109]]
  // CHECK: %[[TMP_111:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = stablehlo.multiply %[[TMP_108]], %[[TMP_111]]
  // CHECK: %[[TMP_113:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_114:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_116:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_115]]
  // CHECK: %[[TMP_117:.*]] = stablehlo.multiply %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = stablehlo.constant dense<-8.26719599E-7>
  // CHECK: %[[TMP_119:.*]] = stablehlo.add %[[TMP_112]], %[[TMP_118]]
  // CHECK: %[[TMP_120:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = stablehlo.multiply %[[TMP_117]], %[[TMP_120]]
  // CHECK: %[[TMP_122:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_123:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_122]]
  // CHECK: %[[TMP_124:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_125:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_124]]
  // CHECK: %[[TMP_126:.*]] = stablehlo.multiply %[[TMP_123]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = stablehlo.constant dense<3.30687835E-5>
  // CHECK: %[[TMP_128:.*]] = stablehlo.add %[[TMP_121]], %[[TMP_127]]
  // CHECK: %[[TMP_129:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_128]]
  // CHECK: %[[TMP_130:.*]] = stablehlo.multiply %[[TMP_126]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_132:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_131]]
  // CHECK: %[[TMP_133:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_134:.*]] = stablehlo.add %[[TMP_0]], %[[TMP_133]]
  // CHECK: %[[TMP_135:.*]] = stablehlo.multiply %[[TMP_132]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = stablehlo.constant dense<-0.00138888892>
  // CHECK: %[[TMP_137:.*]] = stablehlo.add %[[TMP_130]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = stablehlo.multiply %[[TMP_40]], %[[TMP_137]]
  // CHECK: %[[TMP_139:.*]] = stablehlo.multiply %[[TMP_135]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_141:.*]] = stablehlo.divide %[[TMP_0]], %[[TMP_33]]
  // CHECK: %[[TMP_142:.*]] = stablehlo.constant dense<0.0833333358>
  // CHECK: %[[TMP_143:.*]] = stablehlo.add %[[TMP_142]], %[[TMP_139]]
  // CHECK: %[[TMP_144:.*]] = stablehlo.multiply %[[TMP_141]], %[[TMP_143]]
  // CHECK: %[[TMP_145:.*]] = stablehlo.add %[[TMP_140]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = stablehlo.multiply %[[TMP_34]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = stablehlo.add %[[TMP_32]], %[[TMP_38]]
  // CHECK: %[[TMP_148:.*]] = stablehlo.add %[[TMP_147]], %[[TMP_146]]
  // CHECK: %[[TMP_149:.*]] = stablehlo.abs %[[TMP_34]]
  // CHECK: %[[TMP_150:.*]] = stablehlo.abs %[[TMP_32]]
  // CHECK: %[[TMP_151:.*]] = stablehlo.constant dense<1.401300e-45>
  // CHECK: %[[TMP_152:.*]] = stablehlo.multiply %[[TMP_150]], %[[TMP_151]]
  // CHECK: %[[TMP_153:.*]] = stablehlo.compare LT, %[[TMP_149]], %[[TMP_152]]
  // CHECK: %[[TMP_154:.*]] = stablehlo.select %[[TMP_153]], %[[TMP_32]], %[[TMP_148]]
  // CHECK: %[[TMP_155:.*]] = stablehlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_156:.*]] = stablehlo.compare LT, %[[TMP_0]], %[[TMP_35]]
  // CHECK: %[[TMP_157:.*]] = stablehlo.select %[[TMP_156]], %[[TMP_155]], %[[TMP_154]]
  // CHECK: %[[TMP_158:.*]] = stablehlo.compare LE, %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_159:.*]] = stablehlo.floor %[[TMP_0]]
  // CHECK: %[[TMP_160:.*]] = stablehlo.compare NE, %[[TMP_0]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = stablehlo.and %[[TMP_158]], %[[TMP_160]] : tensor<i1>
  // CHECK: %[[TMP_162:.*]] = stablehlo.select %[[TMP_161]], %[[TMP_155]], %[[TMP_157]]
  // CHECK: %[[TMP_163:.*]] = stablehlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_164:.*]] = stablehlo.floor %[[TMP_1]]
  // CHECK: %[[TMP_165:.*]] = stablehlo.compare EQ, %[[TMP_1]], %[[TMP_164]]
  // CHECK: %[[TMP_166:.*]] = stablehlo.and %[[TMP_158]], %[[TMP_165]] : tensor<i1>
  // CHECK: %[[TMP_167:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_168:.*]] = stablehlo.floor %[[TMP_0]]
  // CHECK: %[[TMP_169:.*]] = stablehlo.compare EQ, %[[TMP_0]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = stablehlo.remainder %[[TMP_0]], %[[TMP_167]]
  // CHECK: %[[TMP_171:.*]] = stablehlo.compare EQ, %[[TMP_170]], %[[TMP_2]]
  // CHECK: %[[TMP_172:.*]] = stablehlo.and %[[TMP_169]], %[[TMP_171]] : tensor<i1>
  // CHECK: %[[TMP_173:.*]] = stablehlo.select %[[TMP_172]], %[[TMP_163]], %[[TMP_155]]
  // CHECK: %[[TMP_174:.*]] = stablehlo.select %[[TMP_166]], %[[TMP_173]], %[[TMP_162]]
  // CHECK: %[[TMP_175:.*]] = stablehlo.compare EQ, %[[TMP_0]], %[[TMP_3]]
  // CHECK: %[[TMP_176:.*]] = stablehlo.select %[[TMP_175]], %[[TMP_163]], %[[TMP_174]]
  // CHECK: %[[TMP_177:.*]] = stablehlo.convert %[[TMP_176]] : (tensor<f32>) -> tensor<f16>
  %0 = chlo.zeta %arg0, %arg1 : tensor<f16>, tensor<f16> -> tensor<f16>
  func.return %0 : tensor<f16>
}

// -----

// CHECK-LABEL: @polygamma_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>)
func.func @polygamma_f32(%lhs : tensor<f32>, %rhs : tensor<f32>) -> tensor<f32> {
  // CHECK-DAG: %[[TMP_0:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_1:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_2:.*]] = stablehlo.remainder %[[ARG0]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.multiply %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = stablehlo.subtract %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.add %[[ARG0]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_7:.*]] = stablehlo.compare LT, %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = stablehlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_9:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = stablehlo.subtract %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.select %[[TMP_7]], %[[TMP_8]], %[[TMP_10]]
  // CHECK-DAG: %[[TMP_12:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_13:.*]] = stablehlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_14:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = stablehlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = stablehlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_19:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = stablehlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_24:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = stablehlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = stablehlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_29:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = stablehlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_34:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = stablehlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_39:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = stablehlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = stablehlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_54:.*]] = stablehlo.add %[[TMP_53]], %[[TMP_11]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.constant dense<2.01490307>
  // CHECK: %[[TMP_56:.*]] = stablehlo.divide %[[TMP_11]], %[[TMP_53]]
  // CHECK: %[[TMP_57:.*]] = stablehlo.log_plus_one %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.divide %[[TMP_54]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_6]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.subtract %[[TMP_60]], %[[TMP_59]]
  // CHECK: %[[TMP_62:.*]] = stablehlo.multiply %[[TMP_61]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.log %[[TMP_52]]
  // CHECK: %[[TMP_64:.*]] = stablehlo.constant dense<0.918938517>
  // CHECK: %[[TMP_65:.*]] = stablehlo.add %[[TMP_64]], %[[TMP_62]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.add %[[TMP_65]], %[[TMP_63]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.floor %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.subtract %[[TMP_67]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = stablehlo.compare LT, %[[TMP_6]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.select %[[TMP_70]], %[[TMP_71]], %[[TMP_69]]
  // CHECK: %[[TMP_73:.*]] = stablehlo.constant dense<3.14159274>
  // CHECK: %[[TMP_74:.*]] = stablehlo.multiply %[[TMP_73]], %[[TMP_72]]
  // CHECK: %[[TMP_75:.*]] = stablehlo.sine %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.log %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.constant dense<1.14472985>
  // CHECK: %[[TMP_78:.*]] = stablehlo.subtract %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.subtract %[[TMP_78]], %[[TMP_66]]
  // CHECK: %[[TMP_80:.*]] = stablehlo.is_finite %[[TMP_76]]
  // CHECK: %[[TMP_81:.*]] = stablehlo.negate %[[TMP_76]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.select %[[TMP_80]], %[[TMP_79]], %[[TMP_81]]
  // CHECK: %[[TMP_83:.*]] = stablehlo.select %[[TMP_7]], %[[TMP_82]], %[[TMP_66]]
  // CHECK: %[[TMP_84:.*]] = stablehlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_86:.*]] = stablehlo.compare EQ, %[[TMP_84]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = stablehlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_88:.*]] = stablehlo.select %[[TMP_86]], %[[TMP_87]], %[[TMP_83]]
  // CHECK: %[[TMP_89:.*]] = stablehlo.exponential %[[TMP_88]]
  // CHECK-DAG: %[[TMP_90:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_91:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_92:.*]] = stablehlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_93:.*]] = stablehlo.power %[[ARG1]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.add %[[ARG1]], %[[TMP_91]]
  // CHECK: %[[TMP_95:.*]] = stablehlo.power %[[TMP_94]], %[[TMP_92]]
  // CHECK: %[[TMP_96:.*]] = stablehlo.add %[[TMP_93]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.add %[[TMP_94]], %[[TMP_91]]
  // CHECK: %[[TMP_98:.*]] = stablehlo.power %[[TMP_97]], %[[TMP_92]]
  // CHECK: %[[TMP_99:.*]] = stablehlo.add %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.add %[[TMP_97]], %[[TMP_91]]
  // CHECK: %[[TMP_101:.*]] = stablehlo.power %[[TMP_100]], %[[TMP_92]]
  // CHECK: %[[TMP_102:.*]] = stablehlo.add %[[TMP_99]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = stablehlo.add %[[TMP_100]], %[[TMP_91]]
  // CHECK: %[[TMP_104:.*]] = stablehlo.power %[[TMP_103]], %[[TMP_92]]
  // CHECK: %[[TMP_105:.*]] = stablehlo.add %[[TMP_102]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = stablehlo.add %[[TMP_103]], %[[TMP_91]]
  // CHECK: %[[TMP_107:.*]] = stablehlo.power %[[TMP_106]], %[[TMP_92]]
  // CHECK: %[[TMP_108:.*]] = stablehlo.add %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = stablehlo.add %[[TMP_106]], %[[TMP_91]]
  // CHECK: %[[TMP_110:.*]] = stablehlo.power %[[TMP_109]], %[[TMP_92]]
  // CHECK: %[[TMP_111:.*]] = stablehlo.add %[[TMP_108]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = stablehlo.add %[[TMP_109]], %[[TMP_91]]
  // CHECK: %[[TMP_113:.*]] = stablehlo.power %[[TMP_112]], %[[TMP_92]]
  // CHECK: %[[TMP_114:.*]] = stablehlo.add %[[TMP_111]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = stablehlo.add %[[TMP_112]], %[[TMP_91]]
  // CHECK: %[[TMP_116:.*]] = stablehlo.power %[[TMP_115]], %[[TMP_92]]
  // CHECK: %[[TMP_117:.*]] = stablehlo.add %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = stablehlo.add %[[TMP_115]], %[[TMP_91]]
  // CHECK: %[[TMP_119:.*]] = stablehlo.power %[[TMP_118]], %[[TMP_92]]
  // CHECK: %[[TMP_120:.*]] = stablehlo.add %[[TMP_117]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = stablehlo.add %[[TMP_118]], %[[TMP_91]]
  // CHECK: %[[TMP_122:.*]] = stablehlo.power %[[TMP_121]], %[[TMP_92]]
  // CHECK: %[[TMP_123:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_124:.*]] = stablehlo.multiply %[[TMP_122]], %[[TMP_121]]
  // CHECK-DAG: %[[TMP_125:.*]] = stablehlo.subtract %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_126:.*]] = stablehlo.divide %[[TMP_124]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = stablehlo.multiply %[[TMP_121]], %[[TMP_121]]
  // CHECK: %[[TMP_128:.*]] = stablehlo.divide %[[TMP_91]], %[[TMP_127]]
  // CHECK: %[[TMP_129:.*]] = stablehlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_130:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = stablehlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_132:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_131]]
  // CHECK: %[[TMP_133:.*]] = stablehlo.multiply %[[TMP_130]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = stablehlo.constant
  // CHECK: %[[TMP_135:.*]] = stablehlo.add %[[TMP_90]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = stablehlo.multiply %[[TMP_133]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = stablehlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_139:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = stablehlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_141:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_140]]
  // CHECK: %[[TMP_142:.*]] = stablehlo.multiply %[[TMP_139]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = stablehlo.constant
  // CHECK: %[[TMP_144:.*]] = stablehlo.add %[[TMP_137]], %[[TMP_143]]
  // CHECK: %[[TMP_145:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = stablehlo.multiply %[[TMP_142]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = stablehlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_148:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_147]]
  // CHECK: %[[TMP_149:.*]] = stablehlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_150:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_149]]
  // CHECK: %[[TMP_151:.*]] = stablehlo.multiply %[[TMP_148]], %[[TMP_150]]
  // CHECK: %[[TMP_152:.*]] = stablehlo.constant
  // CHECK: %[[TMP_153:.*]] = stablehlo.add %[[TMP_146]], %[[TMP_152]]
  // CHECK: %[[TMP_154:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_153]]
  // CHECK: %[[TMP_155:.*]] = stablehlo.multiply %[[TMP_151]], %[[TMP_154]]
  // CHECK: %[[TMP_156:.*]] = stablehlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_157:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_156]]
  // CHECK: %[[TMP_158:.*]] = stablehlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_159:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_158]]
  // CHECK: %[[TMP_160:.*]] = stablehlo.multiply %[[TMP_157]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = stablehlo.constant
  // CHECK: %[[TMP_162:.*]] = stablehlo.add %[[TMP_155]], %[[TMP_161]]
  // CHECK: %[[TMP_163:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_162]]
  // CHECK: %[[TMP_164:.*]] = stablehlo.multiply %[[TMP_160]], %[[TMP_163]]
  // CHECK: %[[TMP_165:.*]] = stablehlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_166:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_165]]
  // CHECK: %[[TMP_167:.*]] = stablehlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_168:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_167]]
  // CHECK: %[[TMP_169:.*]] = stablehlo.multiply %[[TMP_166]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = stablehlo.constant
  // CHECK: %[[TMP_171:.*]] = stablehlo.add %[[TMP_164]], %[[TMP_170]]
  // CHECK: %[[TMP_172:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_171]]
  // CHECK: %[[TMP_173:.*]] = stablehlo.multiply %[[TMP_169]], %[[TMP_172]]
  // CHECK: %[[TMP_174:.*]] = stablehlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_175:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_174]]
  // CHECK: %[[TMP_176:.*]] = stablehlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_177:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_176]]
  // CHECK: %[[TMP_178:.*]] = stablehlo.multiply %[[TMP_175]], %[[TMP_177]]
  // CHECK: %[[TMP_179:.*]] = stablehlo.constant
  // CHECK: %[[TMP_180:.*]] = stablehlo.add %[[TMP_173]], %[[TMP_179]]
  // CHECK: %[[TMP_181:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_180]]
  // CHECK: %[[TMP_182:.*]] = stablehlo.multiply %[[TMP_178]], %[[TMP_181]]
  // CHECK: %[[TMP_183:.*]] = stablehlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_184:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_183]]
  // CHECK: %[[TMP_185:.*]] = stablehlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_186:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_185]]
  // CHECK: %[[TMP_187:.*]] = stablehlo.multiply %[[TMP_184]], %[[TMP_186]]
  // CHECK: %[[TMP_188:.*]] = stablehlo.constant
  // CHECK: %[[TMP_189:.*]] = stablehlo.add %[[TMP_182]], %[[TMP_188]]
  // CHECK: %[[TMP_190:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_189]]
  // CHECK: %[[TMP_191:.*]] = stablehlo.multiply %[[TMP_187]], %[[TMP_190]]
  // CHECK: %[[TMP_192:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_193:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_192]]
  // CHECK: %[[TMP_194:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_195:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_194]]
  // CHECK: %[[TMP_196:.*]] = stablehlo.multiply %[[TMP_193]], %[[TMP_195]]
  // CHECK: %[[TMP_197:.*]] = stablehlo.constant
  // CHECK: %[[TMP_198:.*]] = stablehlo.add %[[TMP_191]], %[[TMP_197]]
  // CHECK: %[[TMP_199:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_198]]
  // CHECK: %[[TMP_200:.*]] = stablehlo.multiply %[[TMP_196]], %[[TMP_199]]
  // CHECK: %[[TMP_201:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_202:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_201]]
  // CHECK: %[[TMP_203:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_204:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_203]]
  // CHECK: %[[TMP_205:.*]] = stablehlo.multiply %[[TMP_202]], %[[TMP_204]]
  // CHECK: %[[TMP_206:.*]] = stablehlo.constant
  // CHECK: %[[TMP_207:.*]] = stablehlo.add %[[TMP_200]], %[[TMP_206]]
  // CHECK: %[[TMP_208:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_207]]
  // CHECK: %[[TMP_209:.*]] = stablehlo.multiply %[[TMP_205]], %[[TMP_208]]
  // CHECK: %[[TMP_210:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_211:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_210]]
  // CHECK: %[[TMP_212:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_213:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_212]]
  // CHECK: %[[TMP_214:.*]] = stablehlo.multiply %[[TMP_211]], %[[TMP_213]]
  // CHECK: %[[TMP_215:.*]] = stablehlo.constant
  // CHECK: %[[TMP_216:.*]] = stablehlo.add %[[TMP_209]], %[[TMP_215]]
  // CHECK: %[[TMP_217:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_216]]
  // CHECK: %[[TMP_218:.*]] = stablehlo.multiply %[[TMP_214]], %[[TMP_217]]
  // CHECK: %[[TMP_219:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_220:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_219]]
  // CHECK: %[[TMP_221:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_222:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_221]]
  // CHECK: %[[TMP_223:.*]] = stablehlo.multiply %[[TMP_220]], %[[TMP_222]]
  // CHECK: %[[TMP_224:.*]] = stablehlo.constant
  // CHECK: %[[TMP_225:.*]] = stablehlo.add %[[TMP_218]], %[[TMP_224]]
  // CHECK: %[[TMP_226:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_225]]
  // CHECK: %[[TMP_227:.*]] = stablehlo.multiply %[[TMP_223]], %[[TMP_226]]
  // CHECK: %[[TMP_228:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_229:.*]] = stablehlo.divide %[[TMP_5]], %[[TMP_121]]
  // CHECK: %[[TMP_230:.*]] = stablehlo.constant
  // CHECK: %[[TMP_231:.*]] = stablehlo.add %[[TMP_230]], %[[TMP_227]]
  // CHECK: %[[TMP_232:.*]] = stablehlo.multiply %[[TMP_229]], %[[TMP_231]]
  // CHECK: %[[TMP_233:.*]] = stablehlo.add %[[TMP_228]], %[[TMP_232]]
  // CHECK: %[[TMP_234:.*]] = stablehlo.multiply %[[TMP_122]], %[[TMP_233]]
  // CHECK: %[[TMP_235:.*]] = stablehlo.add %[[TMP_120]], %[[TMP_126]]
  // CHECK: %[[TMP_236:.*]] = stablehlo.add %[[TMP_235]], %[[TMP_234]]
  // CHECK: %[[TMP_237:.*]] = stablehlo.abs %[[TMP_122]]
  // CHECK: %[[TMP_238:.*]] = stablehlo.abs %[[TMP_120]]
  // CHECK: %[[TMP_239:.*]] = stablehlo.constant
  // CHECK: %[[TMP_240:.*]] = stablehlo.multiply %[[TMP_238]], %[[TMP_239]]
  // CHECK: %[[TMP_241:.*]] = stablehlo.compare LT, %[[TMP_237]], %[[TMP_240]]
  // CHECK: %[[TMP_242:.*]] = stablehlo.select %[[TMP_241]], %[[TMP_120]], %[[TMP_236]]
  // CHECK: %[[TMP_243:.*]] = stablehlo.constant
  // CHECK: %[[TMP_244:.*]] = stablehlo.compare LT, %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_245:.*]] = stablehlo.select %[[TMP_244]], %[[TMP_243]], %[[TMP_242]]
  // CHECK: %[[TMP_246:.*]] = stablehlo.compare LE, %[[ARG1]], %[[TMP_90]]
  // CHECK: %[[TMP_247:.*]] = stablehlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_248:.*]] = stablehlo.compare NE, %[[TMP_5]], %[[TMP_247]]
  // CHECK: %[[TMP_249:.*]] = stablehlo.and %[[TMP_246]], %[[TMP_248]]
  // CHECK: %[[TMP_250:.*]] = stablehlo.select %[[TMP_249]], %[[TMP_243]], %[[TMP_245]]
  // CHECK: %[[TMP_251:.*]] = stablehlo.constant
  // CHECK: %[[TMP_252:.*]] = stablehlo.floor %[[ARG1]]
  // CHECK: %[[TMP_253:.*]] = stablehlo.compare EQ, %[[ARG1]], %[[TMP_252]]
  // CHECK: %[[TMP_254:.*]] = stablehlo.and %[[TMP_246]], %[[TMP_253]]
  // CHECK: %[[TMP_255:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_256:.*]] = stablehlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_257:.*]] = stablehlo.compare EQ, %[[TMP_5]], %[[TMP_256]]
  // CHECK: %[[TMP_258:.*]] = stablehlo.remainder %[[TMP_5]], %[[TMP_255]]
  // CHECK: %[[TMP_259:.*]] = stablehlo.compare EQ, %[[TMP_258]], %[[TMP_90]]
  // CHECK: %[[TMP_260:.*]] = stablehlo.and %[[TMP_257]], %[[TMP_259]]
  // CHECK: %[[TMP_261:.*]] = stablehlo.select %[[TMP_260]], %[[TMP_251]], %[[TMP_243]]
  // CHECK: %[[TMP_262:.*]] = stablehlo.select %[[TMP_254]], %[[TMP_261]], %[[TMP_250]]
  // CHECK: %[[TMP_263:.*]] = stablehlo.compare EQ, %[[TMP_5]], %[[TMP_91]]
  // CHECK: %[[TMP_264:.*]] = stablehlo.select %[[TMP_263]], %[[TMP_251]], %[[TMP_262]]
  // CHECK: %[[TMP_265:.*]] = stablehlo.multiply %[[TMP_4]], %[[TMP_89]]
  // CHECK: %[[TMP_266:.*]] = stablehlo.multiply %[[TMP_265]], %[[TMP_264]]
  // CHECK: %[[TMP_267:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_268:.*]] = stablehlo.compare EQ, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_269:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_270:.*]] = stablehlo.compare LT, %[[ARG1]], %[[TMP_269]]
  // CHECK: %[[TMP_271:.*]] = stablehlo.negate %[[ARG1]]
  // CHECK: %[[TMP_272:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_273:.*]] = stablehlo.subtract %[[ARG1]], %[[TMP_272]]
  // CHECK: %[[TMP_274:.*]] = stablehlo.select %[[TMP_270]], %[[TMP_271]], %[[TMP_273]]
  // CHECK-DAG: %[[TMP_275:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_276:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_277:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_278:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_279:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_278]]
  // CHECK: %[[TMP_280:.*]] = stablehlo.multiply %[[TMP_279]], %[[TMP_279]]
  // CHECK: %[[TMP_281:.*]] = stablehlo.divide %[[TMP_277]], %[[TMP_280]]
  // CHECK: %[[TMP_282:.*]] = stablehlo.subtract %[[TMP_275]], %[[TMP_281]]
  // CHECK: %[[TMP_283:.*]] = stablehlo.divide %[[TMP_277]], %[[TMP_279]]
  // CHECK: %[[TMP_284:.*]] = stablehlo.add %[[TMP_276]], %[[TMP_283]]
  // CHECK-DAG: %[[TMP_285:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_286:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_287:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_286]]
  // CHECK: %[[TMP_288:.*]] = stablehlo.multiply %[[TMP_287]], %[[TMP_287]]
  // CHECK: %[[TMP_289:.*]] = stablehlo.divide %[[TMP_285]], %[[TMP_288]]
  // CHECK: %[[TMP_290:.*]] = stablehlo.subtract %[[TMP_282]], %[[TMP_289]]
  // CHECK: %[[TMP_291:.*]] = stablehlo.divide %[[TMP_285]], %[[TMP_287]]
  // CHECK: %[[TMP_292:.*]] = stablehlo.add %[[TMP_284]], %[[TMP_291]]
  // CHECK-DAG: %[[TMP_293:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_294:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_295:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_294]]
  // CHECK: %[[TMP_296:.*]] = stablehlo.multiply %[[TMP_295]], %[[TMP_295]]
  // CHECK: %[[TMP_297:.*]] = stablehlo.divide %[[TMP_293]], %[[TMP_296]]
  // CHECK: %[[TMP_298:.*]] = stablehlo.subtract %[[TMP_290]], %[[TMP_297]]
  // CHECK: %[[TMP_299:.*]] = stablehlo.divide %[[TMP_293]], %[[TMP_295]]
  // CHECK: %[[TMP_300:.*]] = stablehlo.add %[[TMP_292]], %[[TMP_299]]
  // CHECK-DAG: %[[TMP_301:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_302:.*]] = stablehlo.constant
  // CHECK: %[[TMP_303:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_302]]
  // CHECK: %[[TMP_304:.*]] = stablehlo.multiply %[[TMP_303]], %[[TMP_303]]
  // CHECK: %[[TMP_305:.*]] = stablehlo.divide %[[TMP_301]], %[[TMP_304]]
  // CHECK: %[[TMP_306:.*]] = stablehlo.subtract %[[TMP_298]], %[[TMP_305]]
  // CHECK: %[[TMP_307:.*]] = stablehlo.divide %[[TMP_301]], %[[TMP_303]]
  // CHECK: %[[TMP_308:.*]] = stablehlo.add %[[TMP_300]], %[[TMP_307]]
  // CHECK-DAG: %[[TMP_309:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_310:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_311:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_310]]
  // CHECK: %[[TMP_312:.*]] = stablehlo.multiply %[[TMP_311]], %[[TMP_311]]
  // CHECK: %[[TMP_313:.*]] = stablehlo.divide %[[TMP_309]], %[[TMP_312]]
  // CHECK: %[[TMP_314:.*]] = stablehlo.subtract %[[TMP_306]], %[[TMP_313]]
  // CHECK: %[[TMP_315:.*]] = stablehlo.divide %[[TMP_309]], %[[TMP_311]]
  // CHECK: %[[TMP_316:.*]] = stablehlo.add %[[TMP_308]], %[[TMP_315]]
  // CHECK-DAG: %[[TMP_317:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_318:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_319:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_318]]
  // CHECK: %[[TMP_320:.*]] = stablehlo.multiply %[[TMP_319]], %[[TMP_319]]
  // CHECK: %[[TMP_321:.*]] = stablehlo.divide %[[TMP_317]], %[[TMP_320]]
  // CHECK: %[[TMP_322:.*]] = stablehlo.subtract %[[TMP_314]], %[[TMP_321]]
  // CHECK: %[[TMP_323:.*]] = stablehlo.divide %[[TMP_317]], %[[TMP_319]]
  // CHECK: %[[TMP_324:.*]] = stablehlo.add %[[TMP_316]], %[[TMP_323]]
  // CHECK-DAG: %[[TMP_325:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_326:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_327:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_326]]
  // CHECK: %[[TMP_328:.*]] = stablehlo.multiply %[[TMP_327]], %[[TMP_327]]
  // CHECK: %[[TMP_329:.*]] = stablehlo.divide %[[TMP_325]], %[[TMP_328]]
  // CHECK: %[[TMP_330:.*]] = stablehlo.subtract %[[TMP_322]], %[[TMP_329]]
  // CHECK: %[[TMP_331:.*]] = stablehlo.divide %[[TMP_325]], %[[TMP_327]]
  // CHECK: %[[TMP_332:.*]] = stablehlo.add %[[TMP_324]], %[[TMP_331]]
  // CHECK-DAG: %[[TMP_333:.*]] = stablehlo.constant
  // CHECK-DAG: %[[TMP_334:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_335:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_334]]
  // CHECK: %[[TMP_336:.*]] = stablehlo.multiply %[[TMP_335]], %[[TMP_335]]
  // CHECK: %[[TMP_337:.*]] = stablehlo.divide %[[TMP_333]], %[[TMP_336]]
  // CHECK: %[[TMP_338:.*]] = stablehlo.subtract %[[TMP_330]], %[[TMP_337]]
  // CHECK: %[[TMP_339:.*]] = stablehlo.divide %[[TMP_333]], %[[TMP_335]]
  // CHECK: %[[TMP_340:.*]] = stablehlo.add %[[TMP_332]], %[[TMP_339]]
  // CHECK: %[[TMP_341:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_342:.*]] = stablehlo.add %[[TMP_341]], %[[TMP_274]]
  // CHECK: %[[TMP_343:.*]] = stablehlo.constant
  // CHECK: %[[TMP_344:.*]] = stablehlo.divide %[[TMP_274]], %[[TMP_341]]
  // CHECK: %[[TMP_345:.*]] = stablehlo.log_plus_one %[[TMP_344]]
  // CHECK: %[[TMP_346:.*]] = stablehlo.add %[[TMP_343]], %[[TMP_345]]
  // CHECK: %[[TMP_347:.*]] = stablehlo.divide %[[TMP_338]], %[[TMP_340]]
  // CHECK: %[[TMP_348:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_349:.*]] = stablehlo.divide %[[TMP_348]], %[[TMP_342]]
  // CHECK: %[[TMP_350:.*]] = stablehlo.add %[[TMP_346]], %[[TMP_347]]
  // CHECK: %[[TMP_351:.*]] = stablehlo.subtract %[[TMP_350]], %[[TMP_349]]
  // CHECK: %[[TMP_352:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_353:.*]] = stablehlo.add %[[ARG1]], %[[TMP_352]]
  // CHECK: %[[TMP_354:.*]] = stablehlo.floor %[[TMP_353]]
  // CHECK: %[[TMP_355:.*]] = stablehlo.abs %[[TMP_354]]
  // CHECK: %[[TMP_356:.*]] = stablehlo.add %[[ARG1]], %[[TMP_355]]
  // CHECK: %[[TMP_357:.*]] = stablehlo.constant
  // CHECK: %[[TMP_358:.*]] = stablehlo.multiply %[[TMP_357]], %[[TMP_356]]
  // CHECK: %[[TMP_359:.*]] = stablehlo.cosine %[[TMP_358]]
  // CHECK: %[[TMP_360:.*]] = stablehlo.sine %[[TMP_358]]
  // CHECK: %[[TMP_361:.*]] = stablehlo.multiply %[[TMP_357]], %[[TMP_359]]
  // CHECK: %[[TMP_362:.*]] = stablehlo.divide %[[TMP_361]], %[[TMP_360]]
  // CHECK: %[[TMP_363:.*]] = stablehlo.subtract %[[TMP_351]], %[[TMP_362]]
  // CHECK: %[[TMP_364:.*]] = stablehlo.select %[[TMP_270]], %[[TMP_363]], %[[TMP_351]]
  // CHECK: %[[TMP_365:.*]] = stablehlo.compare LE, %[[ARG1]], %[[TMP_275]]
  // CHECK: %[[TMP_366:.*]] = stablehlo.floor %[[ARG1]]
  // CHECK: %[[TMP_367:.*]] = stablehlo.compare EQ, %[[ARG1]], %[[TMP_366]]
  // CHECK: %[[TMP_368:.*]] = stablehlo.and %[[TMP_365]], %[[TMP_367]]
  // CHECK: %[[TMP_369:.*]] = stablehlo.constant
  // CHECK: %[[TMP_370:.*]] = stablehlo.select %[[TMP_368]], %[[TMP_369]], %[[TMP_364]]
  // CHECK: %[[TMP_371:.*]] = stablehlo.select %[[TMP_268]], %[[TMP_370]], %[[TMP_266]]
  // CHECK: %[[TMP_372:.*]] = stablehlo.floor %[[ARG0]]
  // CHECK: %[[TMP_373:.*]] = stablehlo.compare NE, %[[ARG0]], %[[TMP_372]]
  // CHECK: %[[TMP_374:.*]] = stablehlo.compare LT, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_375:.*]] = stablehlo.or %[[TMP_373]], %[[TMP_374]]
  // CHECK: %[[TMP_376:.*]] = stablehlo.constant
  // CHECK: %[[TMP_377:.*]] = stablehlo.select %[[TMP_375]], %[[TMP_376]], %[[TMP_371]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f32>, tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @polygamma_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f64>, %[[ARG1:.*]]: tensor<f64>)
func.func @polygamma_f64(%lhs : tensor<f64>, %rhs : tensor<f64>) -> tensor<f64> {
  // CHECK-DAG: %[[TMP_0:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_1:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_2:.*]] = stablehlo.remainder %[[ARG0]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = stablehlo.multiply %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = stablehlo.subtract %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = stablehlo.add %[[ARG0]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_7:.*]] = stablehlo.compare LT, %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = stablehlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_9:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = stablehlo.subtract %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = stablehlo.select %[[TMP_7]], %[[TMP_8]], %[[TMP_10]]
  // CHECK-DAG: %[[TMP_12:.*]] = stablehlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_13:.*]] = stablehlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_14:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = stablehlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = stablehlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_19:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = stablehlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = stablehlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = stablehlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_24:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = stablehlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = stablehlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = stablehlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_29:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = stablehlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = stablehlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_34:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = stablehlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = stablehlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = stablehlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_39:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = stablehlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = stablehlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = stablehlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = stablehlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = stablehlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = stablehlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = stablehlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_54:.*]] = stablehlo.add %[[TMP_53]], %[[TMP_11]]
  // CHECK: %[[TMP_55:.*]] = stablehlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_56:.*]] = stablehlo.divide %[[TMP_11]], %[[TMP_53]]
  // CHECK: %[[TMP_57:.*]] = stablehlo.log_plus_one %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = stablehlo.divide %[[TMP_54]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_6]]
  // CHECK: %[[TMP_61:.*]] = stablehlo.subtract %[[TMP_60]], %[[TMP_59]]
  // CHECK: %[[TMP_62:.*]] = stablehlo.multiply %[[TMP_61]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = stablehlo.log %[[TMP_52]]
  // CHECK: %[[TMP_64:.*]] = stablehlo.constant dense<0.91893853320467266>
  // CHECK: %[[TMP_65:.*]] = stablehlo.add %[[TMP_64]], %[[TMP_62]]
  // CHECK: %[[TMP_66:.*]] = stablehlo.add %[[TMP_65]], %[[TMP_63]]
  // CHECK: %[[TMP_67:.*]] = stablehlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_68:.*]] = stablehlo.floor %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = stablehlo.subtract %[[TMP_67]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = stablehlo.compare LT, %[[TMP_6]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = stablehlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_72:.*]] = stablehlo.select %[[TMP_70]], %[[TMP_71]], %[[TMP_69]]
  // CHECK: %[[TMP_73:.*]] = stablehlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_74:.*]] = stablehlo.multiply %[[TMP_73]], %[[TMP_72]]
  // CHECK: %[[TMP_75:.*]] = stablehlo.sine %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = stablehlo.log %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = stablehlo.constant dense<1.1447298858494002>
  // CHECK: %[[TMP_78:.*]] = stablehlo.subtract %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = stablehlo.subtract %[[TMP_78]], %[[TMP_66]]
  // CHECK: %[[TMP_80:.*]] = stablehlo.is_finite %[[TMP_76]]
  // CHECK: %[[TMP_81:.*]] = stablehlo.negate %[[TMP_76]]
  // CHECK: %[[TMP_82:.*]] = stablehlo.select %[[TMP_80]], %[[TMP_79]], %[[TMP_81]]
  // CHECK: %[[TMP_83:.*]] = stablehlo.select %[[TMP_7]], %[[TMP_82]], %[[TMP_66]]
  // CHECK: %[[TMP_84:.*]] = stablehlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_85:.*]] = stablehlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_86:.*]] = stablehlo.compare EQ, %[[TMP_84]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = stablehlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_88:.*]] = stablehlo.select %[[TMP_86]], %[[TMP_87]], %[[TMP_83]]
  // CHECK: %[[TMP_89:.*]] = stablehlo.exponential %[[TMP_88]]
  // CHECK-DAG: %[[TMP_90:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_91:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_92:.*]] = stablehlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_93:.*]] = stablehlo.power %[[ARG1]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = stablehlo.add %[[ARG1]], %[[TMP_91]]
  // CHECK: %[[TMP_95:.*]] = stablehlo.power %[[TMP_94]], %[[TMP_92]]
  // CHECK: %[[TMP_96:.*]] = stablehlo.add %[[TMP_93]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = stablehlo.add %[[TMP_94]], %[[TMP_91]]
  // CHECK: %[[TMP_98:.*]] = stablehlo.power %[[TMP_97]], %[[TMP_92]]
  // CHECK: %[[TMP_99:.*]] = stablehlo.add %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = stablehlo.add %[[TMP_97]], %[[TMP_91]]
  // CHECK: %[[TMP_101:.*]] = stablehlo.power %[[TMP_100]], %[[TMP_92]]
  // CHECK: %[[TMP_102:.*]] = stablehlo.add %[[TMP_99]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = stablehlo.add %[[TMP_100]], %[[TMP_91]]
  // CHECK: %[[TMP_104:.*]] = stablehlo.power %[[TMP_103]], %[[TMP_92]]
  // CHECK: %[[TMP_105:.*]] = stablehlo.add %[[TMP_102]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = stablehlo.add %[[TMP_103]], %[[TMP_91]]
  // CHECK: %[[TMP_107:.*]] = stablehlo.power %[[TMP_106]], %[[TMP_92]]
  // CHECK: %[[TMP_108:.*]] = stablehlo.add %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = stablehlo.add %[[TMP_106]], %[[TMP_91]]
  // CHECK: %[[TMP_110:.*]] = stablehlo.power %[[TMP_109]], %[[TMP_92]]
  // CHECK: %[[TMP_111:.*]] = stablehlo.add %[[TMP_108]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = stablehlo.add %[[TMP_109]], %[[TMP_91]]
  // CHECK: %[[TMP_113:.*]] = stablehlo.power %[[TMP_112]], %[[TMP_92]]
  // CHECK: %[[TMP_114:.*]] = stablehlo.add %[[TMP_111]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = stablehlo.add %[[TMP_112]], %[[TMP_91]]
  // CHECK: %[[TMP_116:.*]] = stablehlo.power %[[TMP_115]], %[[TMP_92]]
  // CHECK: %[[TMP_117:.*]] = stablehlo.add %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = stablehlo.add %[[TMP_115]], %[[TMP_91]]
  // CHECK: %[[TMP_119:.*]] = stablehlo.power %[[TMP_118]], %[[TMP_92]]
  // CHECK: %[[TMP_120:.*]] = stablehlo.add %[[TMP_117]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = stablehlo.add %[[TMP_118]], %[[TMP_91]]
  // CHECK: %[[TMP_122:.*]] = stablehlo.power %[[TMP_121]], %[[TMP_92]]
  // CHECK: %[[TMP_123:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_124:.*]] = stablehlo.multiply %[[TMP_122]], %[[TMP_121]]
  // CHECK-DAG: %[[TMP_125:.*]] = stablehlo.subtract %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_126:.*]] = stablehlo.divide %[[TMP_124]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = stablehlo.multiply %[[TMP_121]], %[[TMP_121]]
  // CHECK: %[[TMP_128:.*]] = stablehlo.divide %[[TMP_91]], %[[TMP_127]]
  // CHECK: %[[TMP_129:.*]] = stablehlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_130:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = stablehlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_132:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_131]]
  // CHECK: %[[TMP_133:.*]] = stablehlo.multiply %[[TMP_130]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = stablehlo.constant dense<-1.3954464685812522E-19>
  // CHECK: %[[TMP_135:.*]] = stablehlo.add %[[TMP_90]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = stablehlo.multiply %[[TMP_133]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = stablehlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_139:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = stablehlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_141:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_140]]
  // CHECK: %[[TMP_142:.*]] = stablehlo.multiply %[[TMP_139]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = stablehlo.constant dense<5.5090028283602295E-18>
  // CHECK: %[[TMP_144:.*]] = stablehlo.add %[[TMP_137]], %[[TMP_143]]
  // CHECK: %[[TMP_145:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = stablehlo.multiply %[[TMP_142]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = stablehlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_148:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_147]]
  // CHECK: %[[TMP_149:.*]] = stablehlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_150:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_149]]
  // CHECK: %[[TMP_151:.*]] = stablehlo.multiply %[[TMP_148]], %[[TMP_150]]
  // CHECK: %[[TMP_152:.*]] = stablehlo.constant dense<-2.1748686985580617E-16>
  // CHECK: %[[TMP_153:.*]] = stablehlo.add %[[TMP_146]], %[[TMP_152]]
  // CHECK: %[[TMP_154:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_153]]
  // CHECK: %[[TMP_155:.*]] = stablehlo.multiply %[[TMP_151]], %[[TMP_154]]
  // CHECK: %[[TMP_156:.*]] = stablehlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_157:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_156]]
  // CHECK: %[[TMP_158:.*]] = stablehlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_159:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_158]]
  // CHECK: %[[TMP_160:.*]] = stablehlo.multiply %[[TMP_157]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = stablehlo.constant dense<8.5860620562778452E-15>
  // CHECK: %[[TMP_162:.*]] = stablehlo.add %[[TMP_155]], %[[TMP_161]]
  // CHECK: %[[TMP_163:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_162]]
  // CHECK: %[[TMP_164:.*]] = stablehlo.multiply %[[TMP_160]], %[[TMP_163]]
  // CHECK: %[[TMP_165:.*]] = stablehlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_166:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_165]]
  // CHECK: %[[TMP_167:.*]] = stablehlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_168:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_167]]
  // CHECK: %[[TMP_169:.*]] = stablehlo.multiply %[[TMP_166]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = stablehlo.constant dense<-3.3896802963225832E-13>
  // CHECK: %[[TMP_171:.*]] = stablehlo.add %[[TMP_164]], %[[TMP_170]]
  // CHECK: %[[TMP_172:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_171]]
  // CHECK: %[[TMP_173:.*]] = stablehlo.multiply %[[TMP_169]], %[[TMP_172]]
  // CHECK: %[[TMP_174:.*]] = stablehlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_175:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_174]]
  // CHECK: %[[TMP_176:.*]] = stablehlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_177:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_176]]
  // CHECK: %[[TMP_178:.*]] = stablehlo.multiply %[[TMP_175]], %[[TMP_177]]
  // CHECK: %[[TMP_179:.*]] = stablehlo.constant dense<1.3382536530684679E-11>
  // CHECK: %[[TMP_180:.*]] = stablehlo.add %[[TMP_173]], %[[TMP_179]]
  // CHECK: %[[TMP_181:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_180]]
  // CHECK: %[[TMP_182:.*]] = stablehlo.multiply %[[TMP_178]], %[[TMP_181]]
  // CHECK: %[[TMP_183:.*]] = stablehlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_184:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_183]]
  // CHECK: %[[TMP_185:.*]] = stablehlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_186:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_185]]
  // CHECK: %[[TMP_187:.*]] = stablehlo.multiply %[[TMP_184]], %[[TMP_186]]
  // CHECK: %[[TMP_188:.*]] = stablehlo.constant dense<-5.2841901386874932E-10>
  // CHECK: %[[TMP_189:.*]] = stablehlo.add %[[TMP_182]], %[[TMP_188]]
  // CHECK: %[[TMP_190:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_189]]
  // CHECK: %[[TMP_191:.*]] = stablehlo.multiply %[[TMP_187]], %[[TMP_190]]
  // CHECK: %[[TMP_192:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_193:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_192]]
  // CHECK: %[[TMP_194:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_195:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_194]]
  // CHECK: %[[TMP_196:.*]] = stablehlo.multiply %[[TMP_193]], %[[TMP_195]]
  // CHECK: %[[TMP_197:.*]] = stablehlo.constant dense<2.08767569878681E-8>
  // CHECK: %[[TMP_198:.*]] = stablehlo.add %[[TMP_191]], %[[TMP_197]]
  // CHECK: %[[TMP_199:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_198]]
  // CHECK: %[[TMP_200:.*]] = stablehlo.multiply %[[TMP_196]], %[[TMP_199]]
  // CHECK: %[[TMP_201:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_202:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_201]]
  // CHECK: %[[TMP_203:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_204:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_203]]
  // CHECK: %[[TMP_205:.*]] = stablehlo.multiply %[[TMP_202]], %[[TMP_204]]
  // CHECK: %[[TMP_206:.*]] = stablehlo.constant dense<-8.2671957671957675E-7>
  // CHECK: %[[TMP_207:.*]] = stablehlo.add %[[TMP_200]], %[[TMP_206]]
  // CHECK: %[[TMP_208:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_207]]
  // CHECK: %[[TMP_209:.*]] = stablehlo.multiply %[[TMP_205]], %[[TMP_208]]
  // CHECK: %[[TMP_210:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_211:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_210]]
  // CHECK: %[[TMP_212:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_213:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_212]]
  // CHECK: %[[TMP_214:.*]] = stablehlo.multiply %[[TMP_211]], %[[TMP_213]]
  // CHECK: %[[TMP_215:.*]] = stablehlo.constant dense<3.3068783068783071E-5>
  // CHECK: %[[TMP_216:.*]] = stablehlo.add %[[TMP_209]], %[[TMP_215]]
  // CHECK: %[[TMP_217:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_216]]
  // CHECK: %[[TMP_218:.*]] = stablehlo.multiply %[[TMP_214]], %[[TMP_217]]
  // CHECK: %[[TMP_219:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_220:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_219]]
  // CHECK: %[[TMP_221:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_222:.*]] = stablehlo.add %[[TMP_5]], %[[TMP_221]]
  // CHECK: %[[TMP_223:.*]] = stablehlo.multiply %[[TMP_220]], %[[TMP_222]]
  // CHECK: %[[TMP_224:.*]] = stablehlo.constant dense<-0.0013888888888888889>
  // CHECK: %[[TMP_225:.*]] = stablehlo.add %[[TMP_218]], %[[TMP_224]]
  // CHECK: %[[TMP_226:.*]] = stablehlo.multiply %[[TMP_128]], %[[TMP_225]]
  // CHECK: %[[TMP_227:.*]] = stablehlo.multiply %[[TMP_223]], %[[TMP_226]]
  // CHECK: %[[TMP_228:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_229:.*]] = stablehlo.divide %[[TMP_5]], %[[TMP_121]]
  // CHECK: %[[TMP_230:.*]] = stablehlo.constant dense<0.083333333333333329>
  // CHECK: %[[TMP_231:.*]] = stablehlo.add %[[TMP_230]], %[[TMP_227]]
  // CHECK: %[[TMP_232:.*]] = stablehlo.multiply %[[TMP_229]], %[[TMP_231]]
  // CHECK: %[[TMP_233:.*]] = stablehlo.add %[[TMP_228]], %[[TMP_232]]
  // CHECK: %[[TMP_234:.*]] = stablehlo.multiply %[[TMP_122]], %[[TMP_233]]
  // CHECK: %[[TMP_235:.*]] = stablehlo.add %[[TMP_120]], %[[TMP_126]]
  // CHECK: %[[TMP_236:.*]] = stablehlo.add %[[TMP_235]], %[[TMP_234]]
  // CHECK: %[[TMP_237:.*]] = stablehlo.abs %[[TMP_122]]
  // CHECK: %[[TMP_238:.*]] = stablehlo.abs %[[TMP_120]]
  // CHECK: %[[TMP_239:.*]] = stablehlo.constant dense<4.940660e-324>
  // CHECK: %[[TMP_240:.*]] = stablehlo.multiply %[[TMP_238]], %[[TMP_239]]
  // CHECK: %[[TMP_241:.*]] = stablehlo.compare LT, %[[TMP_237]], %[[TMP_240]]
  // CHECK: %[[TMP_242:.*]] = stablehlo.select %[[TMP_241]], %[[TMP_120]], %[[TMP_236]]
  // CHECK: %[[TMP_243:.*]] = stablehlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_244:.*]] = stablehlo.compare LT, %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_245:.*]] = stablehlo.select %[[TMP_244]], %[[TMP_243]], %[[TMP_242]]
  // CHECK: %[[TMP_246:.*]] = stablehlo.compare LE, %[[ARG1]], %[[TMP_90]]
  // CHECK: %[[TMP_247:.*]] = stablehlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_248:.*]] = stablehlo.compare NE, %[[TMP_5]], %[[TMP_247]]
  // CHECK: %[[TMP_249:.*]] = stablehlo.and %[[TMP_246]], %[[TMP_248]]
  // CHECK: %[[TMP_250:.*]] = stablehlo.select %[[TMP_249]], %[[TMP_243]], %[[TMP_245]]
  // CHECK: %[[TMP_251:.*]] = stablehlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_252:.*]] = stablehlo.floor %[[ARG1]]
  // CHECK: %[[TMP_253:.*]] = stablehlo.compare EQ, %[[ARG1]], %[[TMP_252]]
  // CHECK: %[[TMP_254:.*]] = stablehlo.and %[[TMP_246]], %[[TMP_253]]
  // CHECK: %[[TMP_255:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_256:.*]] = stablehlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_257:.*]] = stablehlo.compare EQ, %[[TMP_5]], %[[TMP_256]]
  // CHECK: %[[TMP_258:.*]] = stablehlo.remainder %[[TMP_5]], %[[TMP_255]]
  // CHECK: %[[TMP_259:.*]] = stablehlo.compare EQ, %[[TMP_258]], %[[TMP_90]]
  // CHECK: %[[TMP_260:.*]] = stablehlo.and %[[TMP_257]], %[[TMP_259]]
  // CHECK: %[[TMP_261:.*]] = stablehlo.select %[[TMP_260]], %[[TMP_251]], %[[TMP_243]]
  // CHECK: %[[TMP_262:.*]] = stablehlo.select %[[TMP_254]], %[[TMP_261]], %[[TMP_250]]
  // CHECK: %[[TMP_263:.*]] = stablehlo.compare EQ, %[[TMP_5]], %[[TMP_91]]
  // CHECK: %[[TMP_264:.*]] = stablehlo.select %[[TMP_263]], %[[TMP_251]], %[[TMP_262]]
  // CHECK: %[[TMP_265:.*]] = stablehlo.multiply %[[TMP_4]], %[[TMP_89]]
  // CHECK: %[[TMP_266:.*]] = stablehlo.multiply %[[TMP_265]], %[[TMP_264]]
  // CHECK: %[[TMP_267:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_268:.*]] = stablehlo.compare EQ, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_269:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_270:.*]] = stablehlo.compare LT, %[[ARG1]], %[[TMP_269]]
  // CHECK: %[[TMP_271:.*]] = stablehlo.negate %[[ARG1]]
  // CHECK: %[[TMP_272:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_273:.*]] = stablehlo.subtract %[[ARG1]], %[[TMP_272]]
  // CHECK: %[[TMP_274:.*]] = stablehlo.select %[[TMP_270]], %[[TMP_271]], %[[TMP_273]]
  // CHECK-DAG: %[[TMP_275:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_276:.*]] = stablehlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_277:.*]] = stablehlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_278:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_279:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_278]]
  // CHECK: %[[TMP_280:.*]] = stablehlo.multiply %[[TMP_279]], %[[TMP_279]]
  // CHECK: %[[TMP_281:.*]] = stablehlo.divide %[[TMP_277]], %[[TMP_280]]
  // CHECK: %[[TMP_282:.*]] = stablehlo.subtract %[[TMP_275]], %[[TMP_281]]
  // CHECK: %[[TMP_283:.*]] = stablehlo.divide %[[TMP_277]], %[[TMP_279]]
  // CHECK: %[[TMP_284:.*]] = stablehlo.add %[[TMP_276]], %[[TMP_283]]
  // CHECK-DAG: %[[TMP_285:.*]] = stablehlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_286:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_287:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_286]]
  // CHECK: %[[TMP_288:.*]] = stablehlo.multiply %[[TMP_287]], %[[TMP_287]]
  // CHECK: %[[TMP_289:.*]] = stablehlo.divide %[[TMP_285]], %[[TMP_288]]
  // CHECK: %[[TMP_290:.*]] = stablehlo.subtract %[[TMP_282]], %[[TMP_289]]
  // CHECK: %[[TMP_291:.*]] = stablehlo.divide %[[TMP_285]], %[[TMP_287]]
  // CHECK: %[[TMP_292:.*]] = stablehlo.add %[[TMP_284]], %[[TMP_291]]
  // CHECK-DAG: %[[TMP_293:.*]] = stablehlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_294:.*]] = stablehlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_295:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_294]]
  // CHECK: %[[TMP_296:.*]] = stablehlo.multiply %[[TMP_295]], %[[TMP_295]]
  // CHECK: %[[TMP_297:.*]] = stablehlo.divide %[[TMP_293]], %[[TMP_296]]
  // CHECK: %[[TMP_298:.*]] = stablehlo.subtract %[[TMP_290]], %[[TMP_297]]
  // CHECK: %[[TMP_299:.*]] = stablehlo.divide %[[TMP_293]], %[[TMP_295]]
  // CHECK: %[[TMP_300:.*]] = stablehlo.add %[[TMP_292]], %[[TMP_299]]
  // CHECK-DAG: %[[TMP_301:.*]] = stablehlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_302:.*]] = stablehlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_303:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_302]]
  // CHECK: %[[TMP_304:.*]] = stablehlo.multiply %[[TMP_303]], %[[TMP_303]]
  // CHECK: %[[TMP_305:.*]] = stablehlo.divide %[[TMP_301]], %[[TMP_304]]
  // CHECK: %[[TMP_306:.*]] = stablehlo.subtract %[[TMP_298]], %[[TMP_305]]
  // CHECK: %[[TMP_307:.*]] = stablehlo.divide %[[TMP_301]], %[[TMP_303]]
  // CHECK: %[[TMP_308:.*]] = stablehlo.add %[[TMP_300]], %[[TMP_307]]
  // CHECK-DAG: %[[TMP_309:.*]] = stablehlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_310:.*]] = stablehlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_311:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_310]]
  // CHECK: %[[TMP_312:.*]] = stablehlo.multiply %[[TMP_311]], %[[TMP_311]]
  // CHECK: %[[TMP_313:.*]] = stablehlo.divide %[[TMP_309]], %[[TMP_312]]
  // CHECK: %[[TMP_314:.*]] = stablehlo.subtract %[[TMP_306]], %[[TMP_313]]
  // CHECK: %[[TMP_315:.*]] = stablehlo.divide %[[TMP_309]], %[[TMP_311]]
  // CHECK: %[[TMP_316:.*]] = stablehlo.add %[[TMP_308]], %[[TMP_315]]
  // CHECK-DAG: %[[TMP_317:.*]] = stablehlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_318:.*]] = stablehlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_319:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_318]]
  // CHECK: %[[TMP_320:.*]] = stablehlo.multiply %[[TMP_319]], %[[TMP_319]]
  // CHECK: %[[TMP_321:.*]] = stablehlo.divide %[[TMP_317]], %[[TMP_320]]
  // CHECK: %[[TMP_322:.*]] = stablehlo.subtract %[[TMP_314]], %[[TMP_321]]
  // CHECK: %[[TMP_323:.*]] = stablehlo.divide %[[TMP_317]], %[[TMP_319]]
  // CHECK: %[[TMP_324:.*]] = stablehlo.add %[[TMP_316]], %[[TMP_323]]
  // CHECK-DAG: %[[TMP_325:.*]] = stablehlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_326:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_327:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_326]]
  // CHECK: %[[TMP_328:.*]] = stablehlo.multiply %[[TMP_327]], %[[TMP_327]]
  // CHECK: %[[TMP_329:.*]] = stablehlo.divide %[[TMP_325]], %[[TMP_328]]
  // CHECK: %[[TMP_330:.*]] = stablehlo.subtract %[[TMP_322]], %[[TMP_329]]
  // CHECK: %[[TMP_331:.*]] = stablehlo.divide %[[TMP_325]], %[[TMP_327]]
  // CHECK: %[[TMP_332:.*]] = stablehlo.add %[[TMP_324]], %[[TMP_331]]
  // CHECK-DAG: %[[TMP_333:.*]] = stablehlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_334:.*]] = stablehlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_335:.*]] = stablehlo.add %[[TMP_274]], %[[TMP_334]]
  // CHECK: %[[TMP_336:.*]] = stablehlo.multiply %[[TMP_335]], %[[TMP_335]]
  // CHECK: %[[TMP_337:.*]] = stablehlo.divide %[[TMP_333]], %[[TMP_336]]
  // CHECK: %[[TMP_338:.*]] = stablehlo.subtract %[[TMP_330]], %[[TMP_337]]
  // CHECK: %[[TMP_339:.*]] = stablehlo.divide %[[TMP_333]], %[[TMP_335]]
  // CHECK: %[[TMP_340:.*]] = stablehlo.add %[[TMP_332]], %[[TMP_339]]
  // CHECK: %[[TMP_341:.*]] = stablehlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_342:.*]] = stablehlo.add %[[TMP_341]], %[[TMP_274]]
  // CHECK: %[[TMP_343:.*]] = stablehlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_344:.*]] = stablehlo.divide %[[TMP_274]], %[[TMP_341]]
  // CHECK: %[[TMP_345:.*]] = stablehlo.log_plus_one %[[TMP_344]]
  // CHECK: %[[TMP_346:.*]] = stablehlo.add %[[TMP_343]], %[[TMP_345]]
  // CHECK: %[[TMP_347:.*]] = stablehlo.divide %[[TMP_338]], %[[TMP_340]]
  // CHECK: %[[TMP_348:.*]] = stablehlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_349:.*]] = stablehlo.divide %[[TMP_348]], %[[TMP_342]]
  // CHECK: %[[TMP_350:.*]] = stablehlo.add %[[TMP_346]], %[[TMP_347]]
  // CHECK: %[[TMP_351:.*]] = stablehlo.subtract %[[TMP_350]], %[[TMP_349]]
  // CHECK: %[[TMP_352:.*]] = stablehlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_353:.*]] = stablehlo.add %[[ARG1]], %[[TMP_352]]
  // CHECK: %[[TMP_354:.*]] = stablehlo.floor %[[TMP_353]]
  // CHECK: %[[TMP_355:.*]] = stablehlo.abs %[[TMP_354]]
  // CHECK: %[[TMP_356:.*]] = stablehlo.add %[[ARG1]], %[[TMP_355]]
  // CHECK: %[[TMP_357:.*]] = stablehlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_358:.*]] = stablehlo.multiply %[[TMP_357]], %[[TMP_356]]
  // CHECK: %[[TMP_359:.*]] = stablehlo.cosine %[[TMP_358]]
  // CHECK: %[[TMP_360:.*]] = stablehlo.sine %[[TMP_358]]
  // CHECK: %[[TMP_361:.*]] = stablehlo.multiply %[[TMP_357]], %[[TMP_359]]
  // CHECK: %[[TMP_362:.*]] = stablehlo.divide %[[TMP_361]], %[[TMP_360]]
  // CHECK: %[[TMP_363:.*]] = stablehlo.subtract %[[TMP_351]], %[[TMP_362]]
  // CHECK: %[[TMP_364:.*]] = stablehlo.select %[[TMP_270]], %[[TMP_363]], %[[TMP_351]]
  // CHECK: %[[TMP_365:.*]] = stablehlo.compare LE, %[[ARG1]], %[[TMP_275]]
  // CHECK: %[[TMP_366:.*]] = stablehlo.floor %[[ARG1]]
  // CHECK: %[[TMP_367:.*]] = stablehlo.compare EQ, %[[ARG1]], %[[TMP_366]]
  // CHECK: %[[TMP_368:.*]] = stablehlo.and %[[TMP_365]], %[[TMP_367]]
  // CHECK: %[[TMP_369:.*]] = stablehlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_370:.*]] = stablehlo.select %[[TMP_368]], %[[TMP_369]], %[[TMP_364]]
  // CHECK: %[[TMP_371:.*]] = stablehlo.select %[[TMP_268]], %[[TMP_370]], %[[TMP_266]]
  // CHECK: %[[TMP_372:.*]] = stablehlo.floor %[[ARG0]]
  // CHECK: %[[TMP_373:.*]] = stablehlo.compare NE, %[[ARG0]], %[[TMP_372]]
  // CHECK: %[[TMP_374:.*]] = stablehlo.compare LT, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_375:.*]] = stablehlo.or %[[TMP_373]], %[[TMP_374]]
  // CHECK: %[[TMP_376:.*]] = stablehlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_377:.*]] = stablehlo.select %[[TMP_375]], %[[TMP_376]], %[[TMP_371]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f64>, tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @polygamma_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>, %[[ARG1:.*]]: tensor<f16>)
func.func @polygamma_f16(%lhs : tensor<f16>, %rhs : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG0]] : (tensor<f16>) -> tensor<f32>
  // CHECK: stablehlo.convert %[[ARG1]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f16>, tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----


// CHECK-LABEL: @sinh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func.func @sinh_f32(%x : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[LOG_HALF:.*]] = stablehlo.log %[[HALF]] : tensor<f32>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = stablehlo.add %[[X]], %[[LOG_HALF]] : tensor<f32>
  // CHECK: %[[EXP_1:.*]] = stablehlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<f32>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = stablehlo.subtract %[[LOG_HALF]], %[[X]] : tensor<f32>
  // CHECK: %[[EXP_2:.*]] = stablehlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<f32>
  // CHECK: %[[LARGE_SINH_RESULT:.*]] = stablehlo.subtract %[[EXP_1]], %[[EXP_2]] : tensor<f32>
  // CHECK: %[[EXPM1:.*]] = stablehlo.exponential_minus_one %[[X]] : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[EXPM1_PLUS_ONE:.*]] = stablehlo.add %[[EXPM1]], %[[ONE]] : tensor<f32>
  // CHECK: %[[RATIO:.*]] = stablehlo.divide %[[EXPM1]], %[[EXPM1_PLUS_ONE]] : tensor<f32>
  // CHECK: %[[SUM:.*]] = stablehlo.add %[[EXPM1]], %[[RATIO]] : tensor<f32>
  // CHECK: %[[SMALL_SINH_RESULT:.*]] = stablehlo.multiply %[[HALF]], %[[SUM]] : tensor<f32>
  // CHECK: %[[ABS_X:.*]] = stablehlo.abs %[[X]] : tensor<f32>
  // CHECK: %[[ABS_X_LT_ONE:.*]] = stablehlo.compare LT, %[[ABS_X]], %[[ONE]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: %[[RESULT:.*]] = stablehlo.select %[[ABS_X_LT_ONE]], %[[SMALL_SINH_RESULT]], %[[LARGE_SINH_RESULT]] : tensor<i1>, tensor<f32>
  // CHECK: return %[[RESULT]] : tensor<f32>
  %1 = chlo.sinh %x : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @sinh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func.func @sinh_f16(%x : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG0]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.sinh %x : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @sinh_complex
// CHECK-SAME: (%[[X:.*]]: tensor<2xcomplex<f32>>)
func.func @sinh_complex(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: %[[HALF:.*]] = stablehlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_HALF:.*]] = stablehlo.log %[[HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = stablehlo.add %[[X]], %[[LOG_HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[EXP_1:.*]] = stablehlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = stablehlo.subtract %[[LOG_HALF]], %[[X]] : tensor<2xcomplex<f32>>
  // CHECK: %[[EXP_2:.*]] = stablehlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<2xcomplex<f32>>
  // CHECK: %[[RESULT:.*]] = stablehlo.subtract %[[EXP_1]], %[[EXP_2]] : tensor<2xcomplex<f32>>
  // CHECK: return %[[RESULT]] : tensor<2xcomplex<f32>>
  %1 = chlo.sinh %x : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @cosh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func.func @cosh_f32(%x : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[LOG_HALF:.*]] = stablehlo.log %[[HALF]] : tensor<f32>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = stablehlo.add %[[X]], %[[LOG_HALF]] : tensor<f32>
  // CHECK: %[[EXP_1:.*]] = stablehlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<f32>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = stablehlo.subtract %[[LOG_HALF]], %[[X]] : tensor<f32>
  // CHECK: %[[EXP_2:.*]] = stablehlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<f32>
  // CHECK: %[[RESULT:.*]] = stablehlo.add %[[EXP_1]], %[[EXP_2]] : tensor<f32>
  // CHECK: return %[[RESULT]] : tensor<f32>
  %1 = chlo.cosh %x : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @cosh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func.func @cosh_f16(%x : tensor<f16>) -> tensor<f16> {
  // CHECK: stablehlo.convert %[[ARG0]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.cosh %x : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @cosh_complex_f32
// CHECK-SAME: (%[[X:.*]]: tensor<complex<f32>>)
func.func @cosh_complex_f32(%x : tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: %[[HALF:.*]] = stablehlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<complex<f32>>
  // CHECK: %[[LOG_HALF:.*]] = stablehlo.log %[[HALF]] : tensor<complex<f32>>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = stablehlo.add %[[X]], %[[LOG_HALF]] : tensor<complex<f32>>
  // CHECK: %[[EXP_1:.*]] = stablehlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<complex<f32>>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = stablehlo.subtract %[[LOG_HALF]], %[[X]] : tensor<complex<f32>>
  // CHECK: %[[EXP_2:.*]] = stablehlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<complex<f32>>
  // CHECK: %[[RESULT:.*]] = stablehlo.add %[[EXP_1]], %[[EXP_2]] : tensor<complex<f32>>
  // CHECK: return %[[RESULT]] : tensor<complex<f32>>
  %1 = chlo.cosh %x : tensor<complex<f32>> -> tensor<complex<f32>>
  func.return %1 : tensor<complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @atanh_f32(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.abs %[[VAL_0]] : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.compare  GT, %[[VAL_1]], %[[VAL_2]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<0x7FC00000> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.log_plus_one %[[VAL_0]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.negate %[[VAL_0]] : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.log_plus_one %[[VAL_6]] : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.subtract %[[VAL_5]], %[[VAL_7]] : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.multiply %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.select %[[VAL_3]], %[[VAL_4]], %[[VAL_10]] : tensor<i1>, tensor<f32>
// CHECK:           return %[[VAL_11]] : tensor<f32>
// CHECK:         }
func.func @atanh_f32(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.atanh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL:    func.func @atanh_complex_f32(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.real %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.compare  GE, %[[VAL_1]], %[[VAL_2]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.select %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.abs %[[VAL_1]] : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.compare  GT, %[[VAL_9]], %[[VAL_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<9.00719925E+15> : tensor<f32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<9.99999968E+37> : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.compare  GT, %[[VAL_9]], %[[VAL_13]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<0x4B800001> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.constant dense<2.050000e+03> : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.select %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.select %[[VAL_11]], %[[VAL_12]], %[[VAL_17]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.multiply %[[VAL_18]], %[[VAL_18]] : tensor<f32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.compare  LT, %[[VAL_8]], %[[VAL_19]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_21:.*]] = stablehlo.imag %[[VAL_0]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.abs %[[VAL_21]] : tensor<f32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.compare  LT, %[[VAL_22]], %[[VAL_19]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_24:.*]] = stablehlo.and %[[VAL_20]], %[[VAL_23]] : tensor<i1>
// CHECK:           %[[VAL_25:.*]] = stablehlo.subtract %[[VAL_4]], %[[VAL_8]] : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.multiply %[[VAL_25]], %[[VAL_25]] : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.multiply %[[VAL_21]], %[[VAL_21]] : tensor<f32>
// CHECK:           %[[VAL_28:.*]] = stablehlo.add %[[VAL_26]], %[[VAL_27]] : tensor<f32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.divide %[[VAL_8]], %[[VAL_28]] : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = stablehlo.multiply %[[VAL_22]], %[[VAL_18]] : tensor<f32>
// CHECK:           %[[VAL_31:.*]] = stablehlo.compare  LT, %[[VAL_30]], %[[VAL_8]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_32:.*]] = stablehlo.divide %[[VAL_4]], %[[VAL_8]] : tensor<f32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.compare  EQ, %[[VAL_1]], %[[VAL_33]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_35:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.compare  EQ, %[[VAL_1]], %[[VAL_35]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_37:.*]] = stablehlo.or %[[VAL_34]], %[[VAL_36]] : tensor<i1>
// CHECK:           %[[VAL_38:.*]] = stablehlo.compare  EQ, %[[VAL_21]], %[[VAL_33]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_39:.*]] = stablehlo.compare  EQ, %[[VAL_21]], %[[VAL_35]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_40:.*]] = stablehlo.or %[[VAL_38]], %[[VAL_39]] : tensor<i1>
// CHECK:           %[[VAL_41:.*]] = stablehlo.or %[[VAL_37]], %[[VAL_40]] : tensor<i1>
// CHECK:           %[[VAL_42:.*]] = stablehlo.divide %[[VAL_8]], %[[VAL_21]] : tensor<f32>
// CHECK:           %[[VAL_43:.*]] = stablehlo.divide %[[VAL_21]], %[[VAL_8]] : tensor<f32>
// CHECK:           %[[VAL_44:.*]] = stablehlo.add %[[VAL_42]], %[[VAL_43]] : tensor<f32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.divide %[[VAL_4]], %[[VAL_44]] : tensor<f32>
// CHECK:           %[[VAL_46:.*]] = stablehlo.divide %[[VAL_45]], %[[VAL_21]] : tensor<f32>
// CHECK:           %[[VAL_47:.*]] = stablehlo.select %[[VAL_41]], %[[VAL_2]], %[[VAL_46]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_48:.*]] = stablehlo.select %[[VAL_31]], %[[VAL_32]], %[[VAL_47]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_49:.*]] = stablehlo.select %[[VAL_24]], %[[VAL_29]], %[[VAL_48]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_50:.*]] = stablehlo.multiply %[[VAL_7]], %[[VAL_49]] : tensor<f32>
// CHECK:           %[[VAL_51:.*]] = stablehlo.log_plus_one %[[VAL_50]] : tensor<f32>
// CHECK:           %[[VAL_52:.*]] = stablehlo.multiply %[[VAL_6]], %[[VAL_51]] : tensor<f32>
// CHECK:           %[[VAL_53:.*]] = stablehlo.constant dense<2.500000e-01> : tensor<f32>
// CHECK:           %[[VAL_54:.*]] = stablehlo.multiply %[[VAL_52]], %[[VAL_53]] : tensor<f32>
// CHECK:           %[[VAL_55:.*]] = stablehlo.add %[[VAL_21]], %[[VAL_21]] : tensor<f32>
// CHECK:           %[[VAL_56:.*]] = stablehlo.add %[[VAL_4]], %[[VAL_8]] : tensor<f32>
// CHECK:           %[[VAL_57:.*]] = stablehlo.multiply %[[VAL_25]], %[[VAL_56]] : tensor<f32>
// CHECK:           %[[VAL_58:.*]] = stablehlo.subtract %[[VAL_57]], %[[VAL_27]] : tensor<f32>
// CHECK:           %[[VAL_59:.*]] = stablehlo.atan2 %[[VAL_55]], %[[VAL_58]] : tensor<f32>
// CHECK:           %[[VAL_60:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_61:.*]] = stablehlo.compare  GE, %[[VAL_21]], %[[VAL_60]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:           %[[VAL_62:.*]] = stablehlo.select %[[VAL_61]], %[[VAL_4]], %[[VAL_5]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_63:.*]] = stablehlo.constant dense<3.14159274> : tensor<f32>
// CHECK:           %[[VAL_64:.*]] = stablehlo.multiply %[[VAL_62]], %[[VAL_63]] : tensor<f32>
// CHECK:           %[[VAL_65:.*]] = stablehlo.select %[[VAL_24]], %[[VAL_59]], %[[VAL_64]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_66:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_67:.*]] = stablehlo.multiply %[[VAL_65]], %[[VAL_66]] : tensor<f32>
// CHECK:           %[[VAL_68:.*]] = stablehlo.complex %[[VAL_54]], %[[VAL_67]] : tensor<complex<f32>>
// CHECK:           return %[[VAL_68]] : tensor<complex<f32>>
// CHECK:         }
func.func @atanh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.atanh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @next_after_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func.func @next_after_f32(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[X_AS_INT:.*]] = stablehlo.bitcast_convert %[[ARG0]] : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[Y_AS_INT:.*]] = stablehlo.bitcast_convert %[[ARG1]] : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[X_IS_NAN:.*]] = stablehlo.compare NE, %[[ARG0]], %[[ARG0]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[Y_IS_NAN:.*]] = stablehlo.compare NE, %[[ARG1]], %[[ARG1]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[INPUT_IS_NAN:.*]] = stablehlo.or %[[X_IS_NAN]], %[[Y_IS_NAN]] : tensor<2xi1>
  // CHECK: %[[NAN:.*]] = stablehlo.constant dense<0x7FC00000> : tensor<2xf32>
  // CHECK: %[[NAN_AS_INT:.*]] = stablehlo.bitcast_convert %[[NAN]] : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK-DAG: %[[SIGN_MASK:.*]] = stablehlo.constant dense<-2147483648> : tensor<2xi32>
  // CHECK-DAG: %[[NEGATED_SIGN_MASK:.*]] = stablehlo.constant dense<2147483647> : tensor<2xi32>
  // CHECK: %[[X_ABS:.*]] = stablehlo.and %[[X_AS_INT]], %[[NEGATED_SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[Y_ABS:.*]] = stablehlo.and %[[Y_AS_INT]], %[[NEGATED_SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[X_AND_Y_ARE_EQUAL:.*]] = stablehlo.compare EQ, %[[ARG0]], %[[ARG1]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK: %[[X_ABS_IS_ZERO:.*]] = stablehlo.compare EQ, %[[X_ABS]], %[[ZERO]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[Y_ABS_IS_ZERO:.*]] = stablehlo.compare EQ, %[[Y_ABS]], %[[ZERO]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[X_SIGN:.*]] = stablehlo.and %[[X_AS_INT]], %[[SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[Y_SIGN:.*]] = stablehlo.and %[[Y_AS_INT]], %[[SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<2xi32>
  // CHECK: %[[RESULT_FOR_X_ZERO_Y_NON_ZERO:.*]] = stablehlo.or %[[Y_SIGN]], %[[ONE]] : tensor<2xi32>
  // CHECK: %[[SIGNS_DISAGREE:.*]] = stablehlo.compare NE, %[[X_SIGN]], %[[Y_SIGN]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[X_MAGNITUDE_LARGER_THAN_Y:.*]] = stablehlo.compare GT, %[[X_ABS]], %[[Y_ABS]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[RESULT_HAS_SMALLER_MAGNITUDE:.*]] = stablehlo.or %[[X_MAGNITUDE_LARGER_THAN_Y]], %[[SIGNS_DISAGREE]] : tensor<2xi1>
  // CHECK: %[[MINUS_ONE:.*]] = stablehlo.constant dense<-1> : tensor<2xi32>
  // CHECK: %[[MAGNITUDE_ADJUSTMENT:.*]] = stablehlo.select %[[RESULT_HAS_SMALLER_MAGNITUDE]], %[[MINUS_ONE]], %[[ONE]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT0:.*]] = stablehlo.add %[[X_AS_INT]], %[[MAGNITUDE_ADJUSTMENT]] : tensor<2xi32>
  // CHECK: %[[RESULT1:.*]] = stablehlo.select %[[Y_ABS_IS_ZERO]], %[[Y_AS_INT]], %[[RESULT_FOR_X_ZERO_Y_NON_ZERO]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT2:.*]] = stablehlo.select %[[X_ABS_IS_ZERO]], %[[RESULT1]], %[[RESULT0]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT3:.*]] = stablehlo.select %[[X_AND_Y_ARE_EQUAL]], %[[Y_AS_INT]], %[[RESULT2]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT4:.*]] = stablehlo.select %[[INPUT_IS_NAN]], %[[NAN_AS_INT]], %[[RESULT3]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[FINAL_RESULT:.*]] = stablehlo.bitcast_convert %[[RESULT4]] : (tensor<2xi32>) -> tensor<2xf32>
  // CHECK: return %[[FINAL_RESULT]]
  %1 = chlo.broadcast_next_after %x, %y : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @tan_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @tan_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = stablehlo.tan %[[ARG]] : tensor<f32>
  // CHECK: return %[[TMP_0]] : tensor<f32>
  %1 = chlo.tan %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @tan_complexf32
// CHECK-SAME: %[[ARG0:.+]]: tensor<1xf32>, %[[ARG1:.+]]: tensor<1xf32>
func.func @tan_complexf32(%arg0 : tensor<1xf32>, %arg1 : tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
  // CHECK: %[[TMP_0:.*]] = stablehlo.complex %[[ARG0]], %[[ARG1]] : tensor<1xcomplex<f32>>
  // CHECK: %[[TMP_1:.*]] = stablehlo.tan %[[TMP_0]] : tensor<1xcomplex<f32>>
  // CHECK: %[[TMP_2:.*]] = stablehlo.real %[[TMP_1]] :  (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  // CHECK: %[[TMP_3:.*]] = stablehlo.imag %[[TMP_1]] : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  // CHECK: return %[[TMP_2]], %[[TMP_3]] : tensor<1xf32>, tensor<1xf32>
  %0 = stablehlo.complex %arg0, %arg1 : tensor<1xcomplex<f32>>
  %1 = chlo.tan %0 : tensor<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %2 = stablehlo.real %1 : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  %3 = stablehlo.imag %1 : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  func.return %2, %3 : tensor<1xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: @top_k
// CHECK-SAME: (%[[ARG:.*]]: tensor<16x16xf32>)
func.func @top_k(%arg : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  // CHECK:      %[[IOTA:.*]] = stablehlo.iota dim = 1 : tensor<16x16xi32>
  // CHECK-NEXT: %[[SORT:.*]]:2 = "stablehlo.sort"(%[[ARG]], %[[IOTA]]) <{dimension = 1 : i64, is_stable = true}> ({
  // CHECK-NEXT: ^{{.*}}(%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>, %{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>):
  // CHECK-NEXT:   %[[CMP:.*]] = stablehlo.compare GT, %[[LHS]], %[[RHS]], TOTALORDER
  // CHECK-NEXT:   stablehlo.return %[[CMP]]
  // CHECK-NEXT: }) : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  // CHECK-NEXT: %[[VAL:.*]] = stablehlo.slice %[[SORT]]#0 [0:16, 0:8] : (tensor<16x16xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %[[IDX:.*]] = stablehlo.slice %[[SORT]]#1 [0:16, 0:8] : (tensor<16x16xi32>) -> tensor<16x8xi32>
  // CHECK-NEXT: return %[[VAL]], %[[IDX]]
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  func.return %1#0, %1#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

// -----

// CHECK-LABEL: @dyn_top_k
// CHECK-SAME: ([[ARG:%.*]]: tensor<?x5x?xi1>
// CHECK-SAME: -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
func.func @dyn_top_k(%arg0: tensor<?x5x?xi1>) -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>) {
  // CHECK-NEXT: [[DIM_0_I32:%.*]] = stablehlo.get_dimension_size [[ARG]], dim = 0 : (tensor<?x5x?xi1>) -> tensor<i32>
  // CHECK-NEXT: [[DIM_0_I32x1:%.*]] = stablehlo.reshape [[DIM_0_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[DIM_1_I32:%.*]] = stablehlo.get_dimension_size [[ARG]], dim = 1 : (tensor<?x5x?xi1>) -> tensor<i32>
  // CHECK-NEXT: [[DIM_1_I32x1:%.*]] = stablehlo.reshape [[DIM_1_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[DIM_2_I32:%.*]] = stablehlo.get_dimension_size [[ARG]], dim = 2 : (tensor<?x5x?xi1>) -> tensor<i32>
  // CHECK-NEXT: [[DIM_2_I32x1:%.*]] = stablehlo.reshape [[DIM_2_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[IOTA_SHAPE:%.*]] = stablehlo.concatenate [[DIM_0_I32x1]], [[DIM_1_I32x1]], [[DIM_2_I32x1]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[K_I32:%.*]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-NEXT: [[K_I32x1:%.*]] = stablehlo.reshape [[K_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[RESULT_SHAPE:%.*]] = stablehlo.concatenate [[DIM_0_I32x1]], [[DIM_1_I32x1]], [[K_I32x1]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[IOTA:%.*]] = stablehlo.dynamic_iota [[IOTA_SHAPE]], dim = 2 : (tensor<3xi32>) -> tensor<?x5x?xi32>
  // CHECK-NEXT: [[SORT:%.*]]:2 = "stablehlo.sort"([[ARG]], [[IOTA]]) <{dimension = 2 : i64, is_stable = true}> ({
  // CHECK-NEXT: ^bb0([[ARG_1:%.*]]: tensor<i1>, [[ARG_2:%.*]]: tensor<i1>, [[ARG_3:%.*]]: tensor<i32>, [[ARG_4:%.*]]: tensor<i32>):
  // CHECK-NEXT:   [[CMP:%.*]] = stablehlo.compare  GT, [[ARG_1]], [[ARG_2]] : (tensor<i1>, tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return [[CMP]] : tensor<i1>
  // CHECK-NEXT: }) : (tensor<?x5x?xi1>, tensor<?x5x?xi32>) -> (tensor<?x5x?xi1>, tensor<?x5x?xi32>)
  // CHECK-NEXT: [[STARTS:%.*]] = stablehlo.constant dense<0> : tensor<3xi64>
  // CHECK-NEXT: [[LIMITS:%.*]] = stablehlo.convert [[RESULT_SHAPE]] : (tensor<3xi32>) -> tensor<3xi64>
  // CHECK-NEXT: [[STRIDES:%.*]] = stablehlo.constant dense<1> : tensor<3xi64>
  // CHECK-NEXT: [[VAL:%.*]] = stablehlo.real_dynamic_slice [[SORT]]#0, [[STARTS]], [[LIMITS]], [[STRIDES]] : (tensor<?x5x?xi1>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x5x2xi1>
  // CHECK-NEXT: [[IDX:%.*]] = stablehlo.real_dynamic_slice [[SORT]]#1, [[STARTS]], [[LIMITS]], [[STRIDES]] : (tensor<?x5x?xi32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x5x2xi32>
  // CHECK-NEXT: return [[VAL]], [[IDX]] : tensor<?x5x2xi1>, tensor<?x5x2xi32>
  %values, %indices = chlo.top_k(%arg0, k = 2) : tensor<?x5x?xi1> -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
  return %values, %indices : tensor<?x5x2xi1>, tensor<?x5x2xi32>
}

// -----

func.func @unranked_top_k(%arg : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>) {
  // expected-error@+1 {{failed to legalize operation 'chlo.top_k' that was explicitly marked illegal}}
  %1:2 = chlo.top_k(%arg, k=8) : tensor<*xf32> -> (tensor<*xf32>, tensor<*xi32>)
  func.return %1#0, %1#1 : tensor<*xf32>, tensor<*xi32>
}

// -----

// Verify bessel_i1e operator for f16, f32, f64 separately as they use
// different coefficients.

// CHECK-LABEL: @bessel_i1e_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf16>)
func.func @bessel_i1e_f16(%arg: tensor<16x16xf16>) -> tensor<16x16xf16> {
  // CHECK-NEXT:  %[[TMP_0:.*]] = stablehlo.convert %[[ARG0]] : (tensor<16x16xf16>) -> tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_1:.*]] = stablehlo.abs %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_2:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_3:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_4:.*]] = stablehlo.constant dense<3.200000e+01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_5:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_6:.*]] = stablehlo.multiply %[[TMP_2]], %[[TMP_1]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_7:.*]] = stablehlo.subtract %[[TMP_6]], %[[TMP_3]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_9:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_10:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_11:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_8]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_12:.*]] = stablehlo.subtract %[[TMP_11]], %[[TMP_9]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_13:.*]] = stablehlo.constant dense<9.38153732E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_14:.*]] = stablehlo.add %[[TMP_12]], %[[TMP_13]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_15:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_14]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_16:.*]] = stablehlo.subtract %[[TMP_15]], %[[TMP_8]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_17:.*]] = stablehlo.constant dense<-4.44505908E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_18:.*]] = stablehlo.add %[[TMP_16]], %[[TMP_17]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_19:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_18]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_20:.*]] = stablehlo.subtract %[[TMP_19]], %[[TMP_14]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_21:.*]] = stablehlo.constant dense<2.00329481E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_22:.*]] = stablehlo.add %[[TMP_20]], %[[TMP_21]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_23:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_22]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_24:.*]] = stablehlo.subtract %[[TMP_23]], %[[TMP_18]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_25:.*]] = stablehlo.constant dense<-8.568720e-07> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_26:.*]] = stablehlo.add %[[TMP_24]], %[[TMP_25]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_27:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_26]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_28:.*]] = stablehlo.subtract %[[TMP_27]], %[[TMP_22]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_29:.*]] = stablehlo.constant dense<3.47025139E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_30:.*]] = stablehlo.add %[[TMP_28]], %[[TMP_29]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_31:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_30]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_32:.*]] = stablehlo.subtract %[[TMP_31]], %[[TMP_26]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_33:.*]] = stablehlo.constant dense<-1.32731639E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_34:.*]] = stablehlo.add %[[TMP_32]], %[[TMP_33]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_35:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_34]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_36:.*]] = stablehlo.subtract %[[TMP_35]], %[[TMP_30]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_37:.*]] = stablehlo.constant dense<4.78156508E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_38:.*]] = stablehlo.add %[[TMP_36]], %[[TMP_37]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_39:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_38]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_40:.*]] = stablehlo.subtract %[[TMP_39]], %[[TMP_34]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_41:.*]] = stablehlo.constant dense<-1.61760821E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_42:.*]] = stablehlo.add %[[TMP_40]], %[[TMP_41]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_43:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_42]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_44:.*]] = stablehlo.subtract %[[TMP_43]], %[[TMP_38]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_45:.*]] = stablehlo.constant dense<5.122860e-04> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_46:.*]] = stablehlo.add %[[TMP_44]], %[[TMP_45]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_47:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_46]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_48:.*]] = stablehlo.subtract %[[TMP_47]], %[[TMP_42]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_49:.*]] = stablehlo.constant dense<-0.00151357241> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_50:.*]] = stablehlo.add %[[TMP_48]], %[[TMP_49]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_51:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_50]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_52:.*]] = stablehlo.subtract %[[TMP_51]], %[[TMP_46]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_53:.*]] = stablehlo.constant dense<0.0041564228> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_54:.*]] = stablehlo.add %[[TMP_52]], %[[TMP_53]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_55:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_54]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_56:.*]] = stablehlo.subtract %[[TMP_55]], %[[TMP_50]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_57:.*]] = stablehlo.constant dense<-0.0105640851> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_58:.*]] = stablehlo.add %[[TMP_56]], %[[TMP_57]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_59:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_58]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_60:.*]] = stablehlo.subtract %[[TMP_59]], %[[TMP_54]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_61:.*]] = stablehlo.constant dense<0.0247264486> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_62:.*]] = stablehlo.add %[[TMP_60]], %[[TMP_61]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_63:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_62]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_64:.*]] = stablehlo.subtract %[[TMP_63]], %[[TMP_58]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_65:.*]] = stablehlo.constant dense<-0.0529459827> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_66:.*]] = stablehlo.add %[[TMP_64]], %[[TMP_65]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_67:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_66]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_68:.*]] = stablehlo.subtract %[[TMP_67]], %[[TMP_62]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_69:.*]] = stablehlo.constant dense<0.102643661> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_70:.*]] = stablehlo.add %[[TMP_68]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_71:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_70]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_72:.*]] = stablehlo.subtract %[[TMP_71]], %[[TMP_66]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_73:.*]] = stablehlo.constant dense<-0.176416516> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_74:.*]] = stablehlo.add %[[TMP_72]], %[[TMP_73]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_75:.*]] = stablehlo.multiply %[[TMP_7]], %[[TMP_74]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_76:.*]] = stablehlo.subtract %[[TMP_75]], %[[TMP_70]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_77:.*]] = stablehlo.constant dense<0.252587199> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_78:.*]] = stablehlo.add %[[TMP_76]], %[[TMP_77]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_79:.*]] = stablehlo.subtract %[[TMP_78]], %[[TMP_70]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_80:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_81:.*]] = stablehlo.multiply %[[TMP_79]], %[[TMP_80]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_82:.*]] = stablehlo.multiply %[[TMP_1]], %[[TMP_81]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_83:.*]] = stablehlo.divide %[[TMP_4]], %[[TMP_1]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_84:.*]] = stablehlo.subtract %[[TMP_83]], %[[TMP_3]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_85:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_86:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_87:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_88:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_85]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_89:.*]] = stablehlo.subtract %[[TMP_88]], %[[TMP_86]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_90:.*]] = stablehlo.constant dense<-3.83538046E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_91:.*]] = stablehlo.add %[[TMP_89]], %[[TMP_90]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_92:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_91]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_93:.*]] = stablehlo.subtract %[[TMP_92]], %[[TMP_85]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_94:.*]] = stablehlo.constant dense<-2.63146891E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_95:.*]] = stablehlo.add %[[TMP_93]], %[[TMP_94]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_96:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_95]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_97:.*]] = stablehlo.subtract %[[TMP_96]], %[[TMP_91]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_98:.*]] = stablehlo.constant dense<-2.51223611E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_99:.*]] = stablehlo.add %[[TMP_97]], %[[TMP_98]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_100:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_99]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_101:.*]] = stablehlo.subtract %[[TMP_100]], %[[TMP_95]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_102:.*]] = stablehlo.constant dense<-3.88256467E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_103:.*]] = stablehlo.add %[[TMP_101]], %[[TMP_102]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_104:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_103]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_105:.*]] = stablehlo.subtract %[[TMP_104]], %[[TMP_99]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_106:.*]] = stablehlo.constant dense<-1.10588939E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_107:.*]] = stablehlo.add %[[TMP_105]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_108:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_107]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_109:.*]] = stablehlo.subtract %[[TMP_108]], %[[TMP_103]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_110:.*]] = stablehlo.constant dense<-0.00976109784> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_111:.*]] = stablehlo.add %[[TMP_109]], %[[TMP_110]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_112:.*]] = stablehlo.multiply %[[TMP_84]], %[[TMP_111]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_113:.*]] = stablehlo.subtract %[[TMP_112]], %[[TMP_107]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_114:.*]] = stablehlo.constant dense<0.778576254> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_115:.*]] = stablehlo.add %[[TMP_113]], %[[TMP_114]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_116:.*]] = stablehlo.subtract %[[TMP_115]], %[[TMP_107]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_117:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_118:.*]] = stablehlo.multiply %[[TMP_116]], %[[TMP_117]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_119:.*]] = stablehlo.sqrt %[[TMP_1]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_120:.*]] = stablehlo.divide %[[TMP_118]], %[[TMP_119]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_121:.*]] = stablehlo.compare LE, %[[TMP_1]], %[[TMP_5]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
  // CHECK-NEXT:  %[[TMP_122:.*]] = stablehlo.select %[[TMP_121]], %[[TMP_82]], %[[TMP_120]] : tensor<16x16xi1>, tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_123:.*]] = stablehlo.sign %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_124:.*]] = stablehlo.multiply %[[TMP_123]], %[[TMP_122]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_125:.*]] = stablehlo.convert %[[TMP_124]] : (tensor<16x16xf32>) -> tensor<16x16xf16>
  // CHECK-NEXT:  return %[[TMP_125]] : tensor<16x16xf16>
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf16> -> tensor<16x16xf16>
  func.return %0 : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: @bessel_i1e_f32
// CHECK-SAME:   (%[[ARG0:.*]]: tensor<16x16xf32>)
func.func @bessel_i1e_f32(%arg : tensor<16x16xf32>) -> tensor<16x16xf32> {
  // CHECK-NEXT:  %[[TMP_0:.*]] = stablehlo.abs %[[ARG0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_1:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_2:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_3:.*]] = stablehlo.constant dense<3.200000e+01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_4:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_5:.*]] = stablehlo.multiply %[[TMP_1]], %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_6:.*]] = stablehlo.subtract %[[TMP_5]], %[[TMP_2]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_7:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_9:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_10:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_7]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_11:.*]] = stablehlo.subtract %[[TMP_10]], %[[TMP_8]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_12:.*]] = stablehlo.constant dense<9.38153732E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_13:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_12]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_14:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_13]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_15:.*]] = stablehlo.subtract %[[TMP_14]], %[[TMP_7]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_16:.*]] = stablehlo.constant dense<-4.44505908E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_17:.*]] = stablehlo.add %[[TMP_15]], %[[TMP_16]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_18:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_17]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_19:.*]] = stablehlo.subtract %[[TMP_18]], %[[TMP_13]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_20:.*]] = stablehlo.constant dense<2.00329481E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_21:.*]] = stablehlo.add %[[TMP_19]], %[[TMP_20]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_22:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_21]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_23:.*]] = stablehlo.subtract %[[TMP_22]], %[[TMP_17]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_24:.*]] = stablehlo.constant dense<-8.568720e-07> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_25:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_24]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_26:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_25]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_27:.*]] = stablehlo.subtract %[[TMP_26]], %[[TMP_21]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_28:.*]] = stablehlo.constant dense<3.47025139E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_29:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_28]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_30:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_29]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_31:.*]] = stablehlo.subtract %[[TMP_30]], %[[TMP_25]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_32:.*]] = stablehlo.constant dense<-1.32731639E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_33:.*]] = stablehlo.add %[[TMP_31]], %[[TMP_32]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_34:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_33]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_35:.*]] = stablehlo.subtract %[[TMP_34]], %[[TMP_29]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_36:.*]] = stablehlo.constant dense<4.78156508E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_37:.*]] = stablehlo.add %[[TMP_35]], %[[TMP_36]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_38:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_37]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_39:.*]] = stablehlo.subtract %[[TMP_38]], %[[TMP_33]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_40:.*]] = stablehlo.constant dense<-1.61760821E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_41:.*]] = stablehlo.add %[[TMP_39]], %[[TMP_40]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_42:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_41]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_43:.*]] = stablehlo.subtract %[[TMP_42]], %[[TMP_37]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_44:.*]] = stablehlo.constant dense<5.122860e-04> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_45:.*]] = stablehlo.add %[[TMP_43]], %[[TMP_44]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_46:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_45]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_47:.*]] = stablehlo.subtract %[[TMP_46]], %[[TMP_41]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_48:.*]] = stablehlo.constant dense<-0.00151357241> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_49:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_48]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_50:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_49]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_51:.*]] = stablehlo.subtract %[[TMP_50]], %[[TMP_45]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_52:.*]] = stablehlo.constant dense<0.0041564228> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_53:.*]] = stablehlo.add %[[TMP_51]], %[[TMP_52]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_54:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_53]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_55:.*]] = stablehlo.subtract %[[TMP_54]], %[[TMP_49]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_56:.*]] = stablehlo.constant dense<-0.0105640851> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_57:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_56]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_58:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_57]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_59:.*]] = stablehlo.subtract %[[TMP_58]], %[[TMP_53]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_60:.*]] = stablehlo.constant dense<0.0247264486> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_61:.*]] = stablehlo.add %[[TMP_59]], %[[TMP_60]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_62:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_61]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_63:.*]] = stablehlo.subtract %[[TMP_62]], %[[TMP_57]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_64:.*]] = stablehlo.constant dense<-0.0529459827> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_65:.*]] = stablehlo.add %[[TMP_63]], %[[TMP_64]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_66:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_65]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_67:.*]] = stablehlo.subtract %[[TMP_66]], %[[TMP_61]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_68:.*]] = stablehlo.constant dense<0.102643661> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_69:.*]] = stablehlo.add %[[TMP_67]], %[[TMP_68]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_70:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_71:.*]] = stablehlo.subtract %[[TMP_70]], %[[TMP_65]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_72:.*]] = stablehlo.constant dense<-0.176416516> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_73:.*]] = stablehlo.add %[[TMP_71]], %[[TMP_72]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_74:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_73]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_75:.*]] = stablehlo.subtract %[[TMP_74]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_76:.*]] = stablehlo.constant dense<0.252587199> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_77:.*]] = stablehlo.add %[[TMP_75]], %[[TMP_76]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_78:.*]] = stablehlo.subtract %[[TMP_77]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_79:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_80:.*]] = stablehlo.multiply %[[TMP_78]], %[[TMP_79]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_81:.*]] = stablehlo.multiply %[[TMP_0]], %[[TMP_80]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_82:.*]] = stablehlo.divide %[[TMP_3]], %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_83:.*]] = stablehlo.subtract %[[TMP_82]], %[[TMP_2]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_84:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_85:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_86:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_87:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_84]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_88:.*]] = stablehlo.subtract %[[TMP_87]], %[[TMP_85]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_89:.*]] = stablehlo.constant dense<-3.83538046E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_90:.*]] = stablehlo.add %[[TMP_88]], %[[TMP_89]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_91:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_90]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_92:.*]] = stablehlo.subtract %[[TMP_91]], %[[TMP_84]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_93:.*]] = stablehlo.constant dense<-2.63146891E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_94:.*]] = stablehlo.add %[[TMP_92]], %[[TMP_93]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_95:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_94]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_96:.*]] = stablehlo.subtract %[[TMP_95]], %[[TMP_90]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_97:.*]] = stablehlo.constant dense<-2.51223611E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_98:.*]] = stablehlo.add %[[TMP_96]], %[[TMP_97]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_99:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_98]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_100:.*]] = stablehlo.subtract %[[TMP_99]], %[[TMP_94]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_101:.*]] = stablehlo.constant dense<-3.88256467E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_102:.*]] = stablehlo.add %[[TMP_100]], %[[TMP_101]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_103:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_102]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_104:.*]] = stablehlo.subtract %[[TMP_103]], %[[TMP_98]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_105:.*]] = stablehlo.constant dense<-1.10588939E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_106:.*]] = stablehlo.add %[[TMP_104]], %[[TMP_105]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_107:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_108:.*]] = stablehlo.subtract %[[TMP_107]], %[[TMP_102]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_109:.*]] = stablehlo.constant dense<-0.00976109784> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_110:.*]] = stablehlo.add %[[TMP_108]], %[[TMP_109]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_111:.*]] = stablehlo.multiply %[[TMP_83]], %[[TMP_110]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_112:.*]] = stablehlo.subtract %[[TMP_111]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_113:.*]] = stablehlo.constant dense<0.778576254> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_114:.*]] = stablehlo.add %[[TMP_112]], %[[TMP_113]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_115:.*]] = stablehlo.subtract %[[TMP_114]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_116:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_117:.*]] = stablehlo.multiply %[[TMP_115]], %[[TMP_116]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_118:.*]] = stablehlo.sqrt %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_119:.*]] = stablehlo.divide %[[TMP_117]], %[[TMP_118]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_120:.*]] = stablehlo.compare LE, %[[TMP_0]], %[[TMP_4]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
  // CHECK-NEXT:  %[[TMP_121:.*]] = stablehlo.select %[[TMP_120]], %[[TMP_81]], %[[TMP_119]] : tensor<16x16xi1>, tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_122:.*]] = stablehlo.sign %[[ARG0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_123:.*]] = stablehlo.multiply %[[TMP_122]], %[[TMP_121]] : tensor<16x16xf32>
  // CHECK-NEXT:  return %[[TMP_123]] : tensor<16x16xf32>
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: @bessel_i1e_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf64>)
func.func @bessel_i1e_f64(%arg : tensor<16x16xf64>) -> tensor<16x16xf64> {
  // CHECK-NEXT: %[[TMP_0:.*]] = stablehlo.abs %[[ARG0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_1:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_2:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_3:.*]] = stablehlo.constant dense<3.200000e+01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_4:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_5:.*]] = stablehlo.multiply %[[TMP_1]], %[[TMP_0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_6:.*]] = stablehlo.subtract %[[TMP_5]], %[[TMP_2]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_7:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_9:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_10:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_7]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_11:.*]] = stablehlo.subtract %[[TMP_10]], %[[TMP_8]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_12:.*]] = stablehlo.constant dense<2.7779141127610464E-18> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_13:.*]] = stablehlo.add %[[TMP_11]], %[[TMP_12]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_14:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_13]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_15:.*]] = stablehlo.subtract %[[TMP_14]], %[[TMP_7]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_16:.*]] = stablehlo.constant dense<-2.111421214358166E-17> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_17:.*]] = stablehlo.add %[[TMP_15]], %[[TMP_16]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_18:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_17]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_19:.*]] = stablehlo.subtract %[[TMP_18]], %[[TMP_13]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_20:.*]] = stablehlo.constant dense<1.5536319577362005E-16> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_21:.*]] = stablehlo.add %[[TMP_19]], %[[TMP_20]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_22:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_21]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_23:.*]] = stablehlo.subtract %[[TMP_22]], %[[TMP_17]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_24:.*]] = stablehlo.constant dense<-1.1055969477353862E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_25:.*]] = stablehlo.add %[[TMP_23]], %[[TMP_24]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_26:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_25]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_27:.*]] = stablehlo.subtract %[[TMP_26]], %[[TMP_21]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_28:.*]] = stablehlo.constant dense<7.6006842947354077E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_29:.*]] = stablehlo.add %[[TMP_27]], %[[TMP_28]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_30:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_29]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_31:.*]] = stablehlo.subtract %[[TMP_30]], %[[TMP_25]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_32:.*]] = stablehlo.constant dense<-5.0421855047279118E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_33:.*]] = stablehlo.add %[[TMP_31]], %[[TMP_32]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_34:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_33]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_35:.*]] = stablehlo.subtract %[[TMP_34]], %[[TMP_29]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_36:.*]] = stablehlo.constant dense<3.2237933659455748E-13> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_37:.*]] = stablehlo.add %[[TMP_35]], %[[TMP_36]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_38:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_37]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_39:.*]] = stablehlo.subtract %[[TMP_38]], %[[TMP_33]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_40:.*]] = stablehlo.constant dense<-1.9839743977649436E-12> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_41:.*]] = stablehlo.add %[[TMP_39]], %[[TMP_40]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_42:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_41]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_43:.*]] = stablehlo.subtract %[[TMP_42]], %[[TMP_37]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_44:.*]] = stablehlo.constant dense<1.1736186298890901E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_45:.*]] = stablehlo.add %[[TMP_43]], %[[TMP_44]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_46:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_45]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_47:.*]] = stablehlo.subtract %[[TMP_46]], %[[TMP_41]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_48:.*]] = stablehlo.constant dense<-6.6634897235020271E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_49:.*]] = stablehlo.add %[[TMP_47]], %[[TMP_48]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_50:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_49]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_51:.*]] = stablehlo.subtract %[[TMP_50]], %[[TMP_45]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_52:.*]] = stablehlo.constant dense<3.6255902815521172E-10> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_53:.*]] = stablehlo.add %[[TMP_51]], %[[TMP_52]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_54:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_53]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_55:.*]] = stablehlo.subtract %[[TMP_54]], %[[TMP_49]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_56:.*]] = stablehlo.constant dense<-1.8872497517228294E-9> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_57:.*]] = stablehlo.add %[[TMP_55]], %[[TMP_56]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_58:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_57]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_59:.*]] = stablehlo.subtract %[[TMP_58]], %[[TMP_53]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_60:.*]] = stablehlo.constant dense<9.3815373864957726E-9> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_61:.*]] = stablehlo.add %[[TMP_59]], %[[TMP_60]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_62:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_61]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_63:.*]] = stablehlo.subtract %[[TMP_62]], %[[TMP_57]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_64:.*]] = stablehlo.constant dense<-4.4450591287963281E-8> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_65:.*]] = stablehlo.add %[[TMP_63]], %[[TMP_64]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_66:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_65]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_67:.*]] = stablehlo.subtract %[[TMP_66]], %[[TMP_61]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_68:.*]] = stablehlo.constant dense<2.0032947535521353E-7> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_69:.*]] = stablehlo.add %[[TMP_67]], %[[TMP_68]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_70:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_69]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_71:.*]] = stablehlo.subtract %[[TMP_70]], %[[TMP_65]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_72:.*]] = stablehlo.constant dense<-8.5687202646954547E-7> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_73:.*]] = stablehlo.add %[[TMP_71]], %[[TMP_72]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_74:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_73]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_75:.*]] = stablehlo.subtract %[[TMP_74]], %[[TMP_69]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_76:.*]] = stablehlo.constant dense<3.4702513081376785E-6> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_77:.*]] = stablehlo.add %[[TMP_75]], %[[TMP_76]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_78:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_77]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_79:.*]] = stablehlo.subtract %[[TMP_78]], %[[TMP_73]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_80:.*]] = stablehlo.constant dense<-1.3273163656039436E-5> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_81:.*]] = stablehlo.add %[[TMP_79]], %[[TMP_80]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_82:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_81]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_83:.*]] = stablehlo.subtract %[[TMP_82]], %[[TMP_77]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_84:.*]] = stablehlo.constant dense<4.7815651075500542E-5> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_85:.*]] = stablehlo.add %[[TMP_83]], %[[TMP_84]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_86:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_85]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_87:.*]] = stablehlo.subtract %[[TMP_86]], %[[TMP_81]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_88:.*]] = stablehlo.constant dense<-1.6176081582589674E-4> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_89:.*]] = stablehlo.add %[[TMP_87]], %[[TMP_88]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_90:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_89]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_91:.*]] = stablehlo.subtract %[[TMP_90]], %[[TMP_85]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_92:.*]] = stablehlo.constant dense<5.1228595616857576E-4> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_93:.*]] = stablehlo.add %[[TMP_91]], %[[TMP_92]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_94:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_93]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_95:.*]] = stablehlo.subtract %[[TMP_94]], %[[TMP_89]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_96:.*]] = stablehlo.constant dense<-0.0015135724506312532> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_97:.*]] = stablehlo.add %[[TMP_95]], %[[TMP_96]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_98:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_97]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_99:.*]] = stablehlo.subtract %[[TMP_98]], %[[TMP_93]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_100:.*]] = stablehlo.constant dense<0.0041564229443128882> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_101:.*]] = stablehlo.add %[[TMP_99]], %[[TMP_100]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_102:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_101]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_103:.*]] = stablehlo.subtract %[[TMP_102]], %[[TMP_97]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_104:.*]] = stablehlo.constant dense<-0.010564084894626197> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_105:.*]] = stablehlo.add %[[TMP_103]], %[[TMP_104]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_106:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_105]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_107:.*]] = stablehlo.subtract %[[TMP_106]], %[[TMP_101]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_108:.*]] = stablehlo.constant dense<0.024726449030626516> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_109:.*]] = stablehlo.add %[[TMP_107]], %[[TMP_108]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_110:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_109]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_111:.*]] = stablehlo.subtract %[[TMP_110]], %[[TMP_105]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_112:.*]] = stablehlo.constant dense<-0.052945981208094989> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_113:.*]] = stablehlo.add %[[TMP_111]], %[[TMP_112]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_114:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_113]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_115:.*]] = stablehlo.subtract %[[TMP_114]], %[[TMP_109]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_116:.*]] = stablehlo.constant dense<0.10264365868984709> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_117:.*]] = stablehlo.add %[[TMP_115]], %[[TMP_116]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_118:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_117]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_119:.*]] = stablehlo.subtract %[[TMP_118]], %[[TMP_113]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_120:.*]] = stablehlo.constant dense<-0.17641651835783406> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_121:.*]] = stablehlo.add %[[TMP_119]], %[[TMP_120]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_122:.*]] = stablehlo.multiply %[[TMP_6]], %[[TMP_121]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_123:.*]] = stablehlo.subtract %[[TMP_122]], %[[TMP_117]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_124:.*]] = stablehlo.constant dense<0.25258718644363365> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_125:.*]] = stablehlo.add %[[TMP_123]], %[[TMP_124]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_126:.*]] = stablehlo.subtract %[[TMP_125]], %[[TMP_117]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_127:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_128:.*]] = stablehlo.multiply %[[TMP_126]], %[[TMP_127]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_129:.*]] = stablehlo.multiply %[[TMP_0]], %[[TMP_128]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_130:.*]] = stablehlo.divide %[[TMP_3]], %[[TMP_0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_131:.*]] = stablehlo.subtract %[[TMP_130]], %[[TMP_2]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_132:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_133:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_134:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_135:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_132]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_136:.*]] = stablehlo.subtract %[[TMP_135]], %[[TMP_133]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_137:.*]] = stablehlo.constant dense<7.5172963108421052E-18> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_138:.*]] = stablehlo.add %[[TMP_136]], %[[TMP_137]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_139:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_138]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_140:.*]] = stablehlo.subtract %[[TMP_139]], %[[TMP_132]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_141:.*]] = stablehlo.constant dense<4.4143483230717077E-18> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_142:.*]] = stablehlo.add %[[TMP_140]], %[[TMP_141]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_143:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_142]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_144:.*]] = stablehlo.subtract %[[TMP_143]], %[[TMP_138]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_145:.*]] = stablehlo.constant dense<-4.6503053684893586E-17> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_146:.*]] = stablehlo.add %[[TMP_144]], %[[TMP_145]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_147:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_146]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_148:.*]] = stablehlo.subtract %[[TMP_147]], %[[TMP_142]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_149:.*]] = stablehlo.constant dense<-3.2095259219934238E-17> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_150:.*]] = stablehlo.add %[[TMP_148]], %[[TMP_149]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_151:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_150]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_152:.*]] = stablehlo.subtract %[[TMP_151]], %[[TMP_146]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_153:.*]] = stablehlo.constant dense<2.9626289976459501E-16> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_154:.*]] = stablehlo.add %[[TMP_152]], %[[TMP_153]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_155:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_154]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_156:.*]] = stablehlo.subtract %[[TMP_155]], %[[TMP_150]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_157:.*]] = stablehlo.constant dense<3.3082023109209285E-16> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_158:.*]] = stablehlo.add %[[TMP_156]], %[[TMP_157]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_159:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_158]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_160:.*]] = stablehlo.subtract %[[TMP_159]], %[[TMP_154]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_161:.*]] = stablehlo.constant dense<-1.8803547755107825E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_162:.*]] = stablehlo.add %[[TMP_160]], %[[TMP_161]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_163:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_162]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_164:.*]] = stablehlo.subtract %[[TMP_163]], %[[TMP_158]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_165:.*]] = stablehlo.constant dense<-3.8144030724370075E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_166:.*]] = stablehlo.add %[[TMP_164]], %[[TMP_165]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_167:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_166]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_168:.*]] = stablehlo.subtract %[[TMP_167]], %[[TMP_162]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_169:.*]] = stablehlo.constant dense<1.0420276984128802E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_170:.*]] = stablehlo.add %[[TMP_168]], %[[TMP_169]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_171:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_170]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_172:.*]] = stablehlo.subtract %[[TMP_171]], %[[TMP_166]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_173:.*]] = stablehlo.constant dense<4.272440016711951E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_174:.*]] = stablehlo.add %[[TMP_172]], %[[TMP_173]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_175:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_174]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_176:.*]] = stablehlo.subtract %[[TMP_175]], %[[TMP_170]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_177:.*]] = stablehlo.constant dense<-2.1015418427726643E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_178:.*]] = stablehlo.add %[[TMP_176]], %[[TMP_177]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_179:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_178]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_180:.*]] = stablehlo.subtract %[[TMP_179]], %[[TMP_174]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_181:.*]] = stablehlo.constant dense<-4.0835511110921974E-13> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_182:.*]] = stablehlo.add %[[TMP_180]], %[[TMP_181]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_183:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_182]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_184:.*]] = stablehlo.subtract %[[TMP_183]], %[[TMP_178]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_185:.*]] = stablehlo.constant dense<-7.1985517762459084E-13> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_186:.*]] = stablehlo.add %[[TMP_184]], %[[TMP_185]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_187:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_186]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_188:.*]] = stablehlo.subtract %[[TMP_187]], %[[TMP_182]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_189:.*]] = stablehlo.constant dense<2.0356285441470896E-12> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_190:.*]] = stablehlo.add %[[TMP_188]], %[[TMP_189]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_191:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_190]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_192:.*]] = stablehlo.subtract %[[TMP_191]], %[[TMP_186]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_193:.*]] = stablehlo.constant dense<1.4125807436613782E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_194:.*]] = stablehlo.add %[[TMP_192]], %[[TMP_193]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_195:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_194]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_196:.*]] = stablehlo.subtract %[[TMP_195]], %[[TMP_190]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_197:.*]] = stablehlo.constant dense<3.2526035830154884E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_198:.*]] = stablehlo.add %[[TMP_196]], %[[TMP_197]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_199:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_198]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_200:.*]] = stablehlo.subtract %[[TMP_199]], %[[TMP_194]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_201:.*]] = stablehlo.constant dense<-1.8974958123505413E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_202:.*]] = stablehlo.add %[[TMP_200]], %[[TMP_201]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_203:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_202]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_204:.*]] = stablehlo.subtract %[[TMP_203]], %[[TMP_198]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_205:.*]] = stablehlo.constant dense<-5.5897434621965838E-10> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_206:.*]] = stablehlo.add %[[TMP_204]], %[[TMP_205]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_207:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_206]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_208:.*]] = stablehlo.subtract %[[TMP_207]], %[[TMP_202]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_209:.*]] = stablehlo.constant dense<-3.835380385964237E-9> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_210:.*]] = stablehlo.add %[[TMP_208]], %[[TMP_209]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_211:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_210]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_212:.*]] = stablehlo.subtract %[[TMP_211]], %[[TMP_206]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_213:.*]] = stablehlo.constant dense<-2.6314688468895196E-8> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_214:.*]] = stablehlo.add %[[TMP_212]], %[[TMP_213]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_215:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_214]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_216:.*]] = stablehlo.subtract %[[TMP_215]], %[[TMP_210]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_217:.*]] = stablehlo.constant dense<-2.5122362378702088E-7> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_218:.*]] = stablehlo.add %[[TMP_216]], %[[TMP_217]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_219:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_218]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_220:.*]] = stablehlo.subtract %[[TMP_219]], %[[TMP_214]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_221:.*]] = stablehlo.constant dense<-3.8825648088776906E-6> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_222:.*]] = stablehlo.add %[[TMP_220]], %[[TMP_221]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_223:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_222]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_224:.*]] = stablehlo.subtract %[[TMP_223]], %[[TMP_218]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_225:.*]] = stablehlo.constant dense<-1.1058893876262371E-4> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_226:.*]] = stablehlo.add %[[TMP_224]], %[[TMP_225]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_227:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_226]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_228:.*]] = stablehlo.subtract %[[TMP_227]], %[[TMP_222]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_229:.*]] = stablehlo.constant dense<-0.0097610974913614687> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_230:.*]] = stablehlo.add %[[TMP_228]], %[[TMP_229]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_231:.*]] = stablehlo.multiply %[[TMP_131]], %[[TMP_230]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_232:.*]] = stablehlo.subtract %[[TMP_231]], %[[TMP_226]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_233:.*]] = stablehlo.constant dense<0.7785762350182801> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_234:.*]] = stablehlo.add %[[TMP_232]], %[[TMP_233]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_235:.*]] = stablehlo.subtract %[[TMP_234]], %[[TMP_226]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_236:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_237:.*]] = stablehlo.multiply %[[TMP_235]], %[[TMP_236]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_238:.*]] = stablehlo.sqrt %[[TMP_0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_239:.*]] = stablehlo.divide %[[TMP_237]], %[[TMP_238]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_240:.*]] = stablehlo.compare LE, %[[TMP_0]], %[[TMP_4]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
  // CHECK-NEXT: %[[TMP_241:.*]] = stablehlo.select %[[TMP_240]], %[[TMP_129]], %[[TMP_239]] : tensor<16x16xi1>, tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_242:.*]] = stablehlo.sign %[[ARG0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_243:.*]] = stablehlo.multiply %[[TMP_242]], %[[TMP_241]] : tensor<16x16xf64>
  // CHECK-NEXT: return %[[TMP_243]] : tensor<16x16xf64>
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf64> -> tensor<16x16xf64>
  func.return %0 : tensor<16x16xf64>
}

// -----

// CHECK-LABEL: @erf_inv
// CHECK-SAME:  ([[ARG_0:%.*]]: tensor<16x16xf32>) {
// CHECK-DAG:     [[VAL_0:%.*]] = stablehlo.negate [[ARG_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_1:%.*]] = stablehlo.multiply [[ARG_0]], [[VAL_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_2:%.*]] = stablehlo.log_plus_one [[VAL_1]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_3:%.*]] = stablehlo.negate [[VAL_2]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_4:%.*]] = stablehlo.constant dense<5.000000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_5:%.*]] = stablehlo.compare  LT, [[VAL_3]], [[VAL_4]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_6:%.*]] = stablehlo.constant dense<2.500000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_7:%.*]] = stablehlo.subtract [[VAL_3]], [[VAL_6]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_8:%.*]] = stablehlo.sqrt [[VAL_3]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_9:%.*]] = stablehlo.constant dense<3.000000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_10:%.*]] = stablehlo.subtract [[VAL_8]], [[VAL_9]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_11:%.*]] = stablehlo.select [[VAL_5]], [[VAL_7]], [[VAL_10]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_12:%.*]] = stablehlo.constant dense<2.81022636E-8> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_13:%.*]] = stablehlo.constant dense<-2.00214257E-4> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_14:%.*]] = stablehlo.select [[VAL_5]], [[VAL_12]], [[VAL_13]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_15:%.*]] = stablehlo.constant dense<3.43273939E-7> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_16:%.*]] = stablehlo.constant dense<1.00950558E-4> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_17:%.*]] = stablehlo.select [[VAL_5]], [[VAL_15]], [[VAL_16]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_18:%.*]] = stablehlo.multiply [[VAL_14]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_19:%.*]] = stablehlo.add [[VAL_17]], [[VAL_18]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_20:%.*]] = stablehlo.constant dense<-3.5233877E-6> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_21:%.*]] = stablehlo.constant dense<0.00134934322> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_22:%.*]] = stablehlo.select [[VAL_5]], [[VAL_20]], [[VAL_21]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_23:%.*]] = stablehlo.multiply [[VAL_19]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_24:%.*]] = stablehlo.add [[VAL_22]], [[VAL_23]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_25:%.*]] = stablehlo.constant dense<-4.39150654E-6> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_26:%.*]] = stablehlo.constant dense<-0.00367342844> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_27:%.*]] = stablehlo.select [[VAL_5]], [[VAL_25]], [[VAL_26]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_28:%.*]] = stablehlo.multiply [[VAL_24]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_29:%.*]] = stablehlo.add [[VAL_27]], [[VAL_28]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_30:%.*]] = stablehlo.constant dense<2.1858087E-4> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_31:%.*]] = stablehlo.constant dense<0.00573950773> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_32:%.*]] = stablehlo.select [[VAL_5]], [[VAL_30]], [[VAL_31]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_33:%.*]] = stablehlo.multiply [[VAL_29]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_34:%.*]] = stablehlo.add [[VAL_32]], [[VAL_33]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_35:%.*]] = stablehlo.constant dense<-0.00125372503> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_36:%.*]] = stablehlo.constant dense<-0.0076224613> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_37:%.*]] = stablehlo.select [[VAL_5]], [[VAL_35]], [[VAL_36]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_38:%.*]] = stablehlo.multiply [[VAL_34]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_39:%.*]] = stablehlo.add [[VAL_37]], [[VAL_38]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_40:%.*]] = stablehlo.constant dense<-0.00417768164> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_41:%.*]] = stablehlo.constant dense<0.00943887047> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_42:%.*]] = stablehlo.select [[VAL_5]], [[VAL_40]], [[VAL_41]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_43:%.*]] = stablehlo.multiply [[VAL_39]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_44:%.*]] = stablehlo.add [[VAL_42]], [[VAL_43]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_45:%.*]] = stablehlo.constant dense<0.246640727> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_46:%.*]] = stablehlo.constant dense<1.00167406> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_47:%.*]] = stablehlo.select [[VAL_5]], [[VAL_45]], [[VAL_46]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_48:%.*]] = stablehlo.multiply [[VAL_44]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_49:%.*]] = stablehlo.add [[VAL_47]], [[VAL_48]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_50:%.*]] = stablehlo.constant dense<1.50140941> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_51:%.*]] = stablehlo.constant dense<2.83297682> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_52:%.*]] = stablehlo.select [[VAL_5]], [[VAL_50]], [[VAL_51]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_53:%.*]] = stablehlo.multiply [[VAL_49]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_54:%.*]] = stablehlo.add [[VAL_52]], [[VAL_53]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_55:%.*]] = stablehlo.multiply [[VAL_54]], [[ARG_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_56:%.*]] = stablehlo.abs [[ARG_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_57:%.*]] = stablehlo.constant dense<1.000000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_58:%.*]] = stablehlo.compare  EQ, [[VAL_56]], [[VAL_57]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_59:%.*]] = stablehlo.constant dense<0x7F800000> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_60:%.*]] = stablehlo.multiply [[ARG_0]], [[VAL_59]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_61:%.*]] = stablehlo.select [[VAL_58]], [[VAL_60]], [[VAL_55]] : tensor<16x16xi1>, tensor<16x16xf32>
func.func @erf_inv(%arg0 : tensor<16x16xf32>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  return
}

// -----

// CHECK-LABEL: @erf_inv_wide
// CHECK-SAME:  ([[ARG_0:%.*]]: tensor<16x16xf64>) {
// CHECK-DAG:     [[VAL_0:%.*]] = stablehlo.negate [[ARG_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_1:%.*]] = stablehlo.multiply [[ARG_0]], [[VAL_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_2:%.*]] = stablehlo.log_plus_one [[VAL_1]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_3:%.*]] = stablehlo.negate [[VAL_2]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_4:%.*]] = stablehlo.constant dense<6.250000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_5:%.*]] = stablehlo.compare  LT, [[VAL_3]], [[VAL_4]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_6:%.*]] = stablehlo.constant dense<1.600000e+01> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_7:%.*]] = stablehlo.compare  LT, [[VAL_3]], [[VAL_6]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_8:%.*]] = stablehlo.sqrt [[VAL_3]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_9:%.*]] = stablehlo.constant dense<3.125000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_10:%.*]] = stablehlo.subtract [[VAL_3]], [[VAL_9]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_11:%.*]] = stablehlo.constant dense<3.250000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_12:%.*]] = stablehlo.constant dense<5.000000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_13:%.*]] = stablehlo.select [[VAL_7]], [[VAL_11]], [[VAL_12]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_14:%.*]] = stablehlo.subtract [[VAL_8]], [[VAL_13]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_15:%.*]] = stablehlo.select [[VAL_5]], [[VAL_10]], [[VAL_14]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_16:%.*]] = stablehlo.constant dense<-3.6444120640178197E-21> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_17:%.*]] = stablehlo.constant dense<2.2137376921775787E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_18:%.*]] = stablehlo.select [[VAL_5]], [[VAL_16]], [[VAL_17]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_19:%.*]] = stablehlo.constant dense<-2.7109920616438573E-11> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_20:%.*]] = stablehlo.select [[VAL_7]], [[VAL_18]], [[VAL_19]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_21:%.*]] = stablehlo.constant dense<-1.6850591381820166E-19> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_22:%.*]] = stablehlo.constant dense<9.075656193888539E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_23:%.*]] = stablehlo.select [[VAL_5]], [[VAL_21]], [[VAL_22]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_24:%.*]] = stablehlo.constant dense<-2.5556418169965252E-10> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_25:%.*]] = stablehlo.select [[VAL_7]], [[VAL_23]], [[VAL_24]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_26:%.*]] = stablehlo.multiply [[VAL_20]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_27:%.*]] = stablehlo.add [[VAL_25]], [[VAL_26]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_28:%.*]] = stablehlo.constant dense<1.28584807152564E-18> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_29:%.*]] = stablehlo.constant dense<-2.7517406297064545E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_30:%.*]] = stablehlo.select [[VAL_5]], [[VAL_28]], [[VAL_29]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_31:%.*]] = stablehlo.constant dense<1.5076572693500548E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_32:%.*]] = stablehlo.select [[VAL_7]], [[VAL_30]], [[VAL_31]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_33:%.*]] = stablehlo.multiply [[VAL_27]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_34:%.*]] = stablehlo.add [[VAL_32]], [[VAL_33]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_35:%.*]] = stablehlo.constant dense<1.1157877678025181E-17> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_36:%.*]] = stablehlo.constant dense<1.8239629214389228E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_37:%.*]] = stablehlo.select [[VAL_5]], [[VAL_35]], [[VAL_36]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_38:%.*]] = stablehlo.constant dense<-3.789465440126737E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_39:%.*]] = stablehlo.select [[VAL_7]], [[VAL_37]], [[VAL_38]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_40:%.*]] = stablehlo.multiply [[VAL_34]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_41:%.*]] = stablehlo.add [[VAL_39]], [[VAL_40]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_42:%.*]] = stablehlo.constant dense<-1.3331716628546209E-16> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_43:%.*]] = stablehlo.constant dense<1.5027403968909828E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_44:%.*]] = stablehlo.select [[VAL_5]], [[VAL_42]], [[VAL_43]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_45:%.*]] = stablehlo.constant dense<7.6157012080783394E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_46:%.*]] = stablehlo.select [[VAL_7]], [[VAL_44]], [[VAL_45]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_47:%.*]] = stablehlo.multiply [[VAL_41]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_48:%.*]] = stablehlo.add [[VAL_46]], [[VAL_47]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_49:%.*]] = stablehlo.constant dense<2.0972767875968562E-17> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_50:%.*]] = stablehlo.constant dense<-4.013867526981546E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_51:%.*]] = stablehlo.select [[VAL_5]], [[VAL_49]], [[VAL_50]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_52:%.*]] = stablehlo.constant dense<-1.496002662714924E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_53:%.*]] = stablehlo.select [[VAL_7]], [[VAL_51]], [[VAL_52]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_54:%.*]] = stablehlo.multiply [[VAL_48]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_55:%.*]] = stablehlo.add [[VAL_53]], [[VAL_54]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_56:%.*]] = stablehlo.constant dense<6.6376381343583238E-15> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_57:%.*]] = stablehlo.constant dense<2.9234449089955446E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_58:%.*]] = stablehlo.select [[VAL_5]], [[VAL_56]], [[VAL_57]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_59:%.*]] = stablehlo.constant dense<2.9147953450901081E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_60:%.*]] = stablehlo.select [[VAL_7]], [[VAL_58]], [[VAL_59]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_61:%.*]] = stablehlo.multiply [[VAL_55]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_62:%.*]] = stablehlo.add [[VAL_60]], [[VAL_61]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_63:%.*]] = stablehlo.constant dense<-4.0545662729752069E-14> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_64:%.*]] = stablehlo.constant dense<1.2475304481671779E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_65:%.*]] = stablehlo.select [[VAL_5]], [[VAL_63]], [[VAL_64]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_66:%.*]] = stablehlo.constant dense<-6.7711997758452339E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_67:%.*]] = stablehlo.select [[VAL_7]], [[VAL_65]], [[VAL_66]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_68:%.*]] = stablehlo.multiply [[VAL_62]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_69:%.*]] = stablehlo.add [[VAL_67]], [[VAL_68]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_70:%.*]] = stablehlo.constant dense<-8.1519341976054721E-14> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_71:%.*]] = stablehlo.constant dense<-4.7318229009055734E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_72:%.*]] = stablehlo.select [[VAL_5]], [[VAL_70]], [[VAL_71]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_73:%.*]] = stablehlo.constant dense<2.2900482228026655E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_74:%.*]] = stablehlo.select [[VAL_7]], [[VAL_72]], [[VAL_73]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_75:%.*]] = stablehlo.multiply [[VAL_69]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_76:%.*]] = stablehlo.add [[VAL_74]], [[VAL_75]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_77:%.*]] = stablehlo.constant dense<2.6335093153082323E-12> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_78:%.*]] = stablehlo.constant dense<6.8284851459573175E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_79:%.*]] = stablehlo.select [[VAL_5]], [[VAL_77]], [[VAL_78]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_80:%.*]] = stablehlo.constant dense<-9.9298272942317003E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_81:%.*]] = stablehlo.select [[VAL_7]], [[VAL_79]], [[VAL_80]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_82:%.*]] = stablehlo.multiply [[VAL_76]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_83:%.*]] = stablehlo.add [[VAL_81]], [[VAL_82]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_84:%.*]] = stablehlo.constant dense<-1.2975133253453532E-11> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_85:%.*]] = stablehlo.constant dense<2.4031110387097894E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_86:%.*]] = stablehlo.select [[VAL_5]], [[VAL_84]], [[VAL_85]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_87:%.*]] = stablehlo.constant dense<4.5260625972231537E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_88:%.*]] = stablehlo.select [[VAL_7]], [[VAL_86]], [[VAL_87]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_89:%.*]] = stablehlo.multiply [[VAL_83]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_90:%.*]] = stablehlo.add [[VAL_88]], [[VAL_89]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_91:%.*]] = stablehlo.constant dense<-5.4154120542946279E-11> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_92:%.*]] = stablehlo.constant dense<-3.5503752036284748E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_93:%.*]] = stablehlo.select [[VAL_5]], [[VAL_91]], [[VAL_92]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_94:%.*]] = stablehlo.constant dense<-1.9681778105531671E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_95:%.*]] = stablehlo.select [[VAL_7]], [[VAL_93]], [[VAL_94]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_96:%.*]] = stablehlo.multiply [[VAL_90]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_97:%.*]] = stablehlo.add [[VAL_95]], [[VAL_96]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_98:%.*]] = stablehlo.constant dense<1.0512122733215323E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_99:%.*]] = stablehlo.constant dense<9.5328937973738049E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_100:%.*]] = stablehlo.select [[VAL_5]], [[VAL_98]], [[VAL_99]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_101:%.*]] = stablehlo.constant dense<7.5995277030017761E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_102:%.*]] = stablehlo.select [[VAL_7]], [[VAL_100]], [[VAL_101]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_103:%.*]] = stablehlo.multiply [[VAL_97]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_104:%.*]] = stablehlo.add [[VAL_102]], [[VAL_103]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_105:%.*]] = stablehlo.constant dense<-4.1126339803469837E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_106:%.*]] = stablehlo.constant dense<-0.0016882755560235047> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_107:%.*]] = stablehlo.select [[VAL_5]], [[VAL_105]], [[VAL_106]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_108:%.*]] = stablehlo.constant dense<-2.1503011930044477E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_109:%.*]] = stablehlo.select [[VAL_7]], [[VAL_107]], [[VAL_108]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_110:%.*]] = stablehlo.multiply [[VAL_104]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_111:%.*]] = stablehlo.add [[VAL_109]], [[VAL_110]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_112:%.*]] = stablehlo.constant dense<-2.9070369957882005E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_113:%.*]] = stablehlo.constant dense<0.0024914420961078508> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_114:%.*]] = stablehlo.select [[VAL_5]], [[VAL_112]], [[VAL_113]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_115:%.*]] = stablehlo.constant dense<-1.3871931833623122E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_116:%.*]] = stablehlo.select [[VAL_7]], [[VAL_114]], [[VAL_115]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_117:%.*]] = stablehlo.multiply [[VAL_111]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_118:%.*]] = stablehlo.add [[VAL_116]], [[VAL_117]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_119:%.*]] = stablehlo.constant dense<4.2347877827932404E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_120:%.*]] = stablehlo.constant dense<-0.0037512085075692412> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_121:%.*]] = stablehlo.select [[VAL_5]], [[VAL_119]], [[VAL_120]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_122:%.*]] = stablehlo.constant dense<1.0103004648645344> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_123:%.*]] = stablehlo.select [[VAL_7]], [[VAL_121]], [[VAL_122]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_124:%.*]] = stablehlo.multiply [[VAL_118]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_125:%.*]] = stablehlo.add [[VAL_123]], [[VAL_124]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_126:%.*]] = stablehlo.constant dense<-1.3654692000834679E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_127:%.*]] = stablehlo.constant dense<0.0053709145535900636> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_128:%.*]] = stablehlo.select [[VAL_5]], [[VAL_126]], [[VAL_127]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_129:%.*]] = stablehlo.constant dense<4.8499064014085844> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_130:%.*]] = stablehlo.select [[VAL_7]], [[VAL_128]], [[VAL_129]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_131:%.*]] = stablehlo.multiply [[VAL_125]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_132:%.*]] = stablehlo.add [[VAL_130]], [[VAL_131]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_133:%.*]] = stablehlo.constant dense<-1.3882523362786469E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_134:%.*]] = stablehlo.constant dense<1.0052589676941592> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_135:%.*]] = stablehlo.select [[VAL_5]], [[VAL_133]], [[VAL_134]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_136:%.*]] = stablehlo.multiply [[VAL_132]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_137:%.*]] = stablehlo.add [[VAL_135]], [[VAL_136]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_138:%.*]] = stablehlo.select [[VAL_7]], [[VAL_137]], [[VAL_132]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_139:%.*]] = stablehlo.constant dense<1.8673420803405714E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_140:%.*]] = stablehlo.constant dense<3.0838856104922208> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_141:%.*]] = stablehlo.select [[VAL_5]], [[VAL_139]], [[VAL_140]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_142:%.*]] = stablehlo.multiply [[VAL_138]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_143:%.*]] = stablehlo.add [[VAL_141]], [[VAL_142]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_144:%.*]] = stablehlo.select [[VAL_7]], [[VAL_143]], [[VAL_138]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_145:%.*]] = stablehlo.constant dense<-7.4070253416626698E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_146:%.*]] = stablehlo.multiply [[VAL_144]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_147:%.*]] = stablehlo.add [[VAL_145]], [[VAL_146]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_148:%.*]] = stablehlo.select [[VAL_5]], [[VAL_147]], [[VAL_144]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_149:%.*]] = stablehlo.constant dense<-0.0060336708714301491> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_150:%.*]] = stablehlo.multiply [[VAL_148]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_151:%.*]] = stablehlo.add [[VAL_149]], [[VAL_150]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_152:%.*]] = stablehlo.select [[VAL_5]], [[VAL_151]], [[VAL_148]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_153:%.*]] = stablehlo.constant dense<0.24015818242558962> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_154:%.*]] = stablehlo.multiply [[VAL_152]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_155:%.*]] = stablehlo.add [[VAL_153]], [[VAL_154]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_156:%.*]] = stablehlo.select [[VAL_5]], [[VAL_155]], [[VAL_152]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_157:%.*]] = stablehlo.constant dense<1.6536545626831027> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_158:%.*]] = stablehlo.multiply [[VAL_156]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_159:%.*]] = stablehlo.add [[VAL_157]], [[VAL_158]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_160:%.*]] = stablehlo.select [[VAL_5]], [[VAL_159]], [[VAL_156]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_161:%.*]] = stablehlo.multiply [[VAL_160]], [[ARG_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_162:%.*]] = stablehlo.abs [[ARG_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_163:%.*]] = stablehlo.constant dense<1.000000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_164:%.*]] = stablehlo.compare  EQ, [[VAL_162]], [[VAL_163]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_165:%.*]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_166:%.*]] = stablehlo.multiply [[ARG_0]], [[VAL_165]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_167:%.*]] = stablehlo.select [[VAL_164]], [[VAL_166]], [[VAL_161]] : tensor<16x16xi1>, tensor<16x16xf64>
func.func @erf_inv_wide(%arg0 : tensor<16x16xf64>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf64> -> tensor<16x16xf64>
  return
}
