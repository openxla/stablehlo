func.func @ifft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #stablehlo<fft_type IFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}