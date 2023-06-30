// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --mlir-print-op-generic %s | FileCheck %s
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)
// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo -emit-bytecode -debug-only=vhlo-bytecode %s 2>&1 | (! grep 'Not Implemented')
// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo -emit-bytecode %s | stablehlo-opt -debug-only=vhlo-bytecode 2>&1 | (! grep 'Not Implemented')

func.func @attr_fft_type_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #vhlo<fft_type_v1 FFT>
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_fft"
