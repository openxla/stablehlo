// RUN: stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file %s > %t.mlir
// RUN: stablehlo-translate --interpret --split-input-file %t.mlir

// Cumulative sum via chlo.scan, validated through the chlo-to-stablehlo
// decomposition and the reference interpreter. %0#0 is the per-position
// inclusive prefix sum; %0#1 is the final carry.
func.func @scan_cumsum_1d() {
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = chlo.scan(%input) inits(%init) dimension=0 {
  ^bb0(%scan_arg0: tensor<i32>, %scan_arg1: tensor<i32>):
    %1 = stablehlo.add %scan_arg0, %scan_arg1 : tensor<i32>
    stablehlo.return %1, %1 : tensor<i32>, tensor<i32>
  } : (tensor<4xi32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
  check.expect_eq_const %0#0, dense<[1, 3, 6, 10]> : tensor<4xi32>
  check.expect_eq_const %0#1, dense<10> : tensor<i32>
  func.return
}

// -----

// Cumulative sum along a non-zero scan dimension. The carry is rank-reduced
// (the scan dimension is erased), so the init is a tensor<2xi32>.
func.func @scan_cumsum_dim1() {
  %input = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %init = stablehlo.constant dense<0> : tensor<2xi32>
  %0:2 = chlo.scan(%input) inits(%init) dimension=1 {
  ^bb0(%scan_arg0: tensor<2xi32>, %scan_arg1: tensor<2xi32>):
    %1 = stablehlo.add %scan_arg0, %scan_arg1 : tensor<2xi32>
    stablehlo.return %1, %1 : tensor<2xi32>, tensor<2xi32>
  } : (tensor<2x3xi32>, tensor<2xi32>) -> (tensor<2x3xi32>, tensor<2xi32>)
  check.expect_eq_const %0#0, dense<[[1, 3, 6], [4, 9, 15]]> : tensor<2x3xi32>
  check.expect_eq_const %0#1, dense<[6, 15]> : tensor<2xi32>
  func.return
}

// -----

// Cumulative max with argmax indices: the multi-input, multi-carry pattern
// TorchTPU's cummax/cummin lower to. An iota is fed in as a second input to
// supply each element's position; on ties the highest index is kept.
func.func @scan_cummax() {
  %input = stablehlo.constant dense<[3, 1, 4, 1, 5]> : tensor<5xi32>
  %iota = stablehlo.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %val_init = stablehlo.constant dense<-2147483648> : tensor<i32>
  %idx_init = stablehlo.constant dense<0> : tensor<i32>
  %0:4 = chlo.scan(%input, %iota) inits(%val_init, %idx_init) dimension=0 {
  ^bb0(%cur_val: tensor<i32>, %cur_idx: tensor<i32>, %run_val: tensor<i32>, %run_idx: tensor<i32>):
    %new_val = stablehlo.maximum %run_val, %cur_val : tensor<i32>
    %gt = stablehlo.compare GT, %run_val, %cur_val : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %eq = stablehlo.compare EQ, %run_val, %cur_val : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %max_idx = stablehlo.maximum %run_idx, %cur_idx : tensor<i32>
    %tie_or_less = stablehlo.select %eq, %max_idx, %cur_idx : tensor<i1>, tensor<i32>
    %new_idx = stablehlo.select %gt, %run_idx, %tie_or_less : tensor<i1>, tensor<i32>
    stablehlo.return %new_val, %new_idx, %new_val, %new_idx
        : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  } : (tensor<5xi32>, tensor<5xi32>, tensor<i32>, tensor<i32>)
      -> (tensor<5xi32>, tensor<5xi32>, tensor<i32>, tensor<i32>)
  check.expect_eq_const %0#0, dense<[3, 3, 4, 4, 5]> : tensor<5xi32>
  check.expect_eq_const %0#1, dense<[0, 0, 2, 2, 4]> : tensor<5xi32>
  check.expect_eq_const %0#2, dense<5> : tensor<i32>
  check.expect_eq_const %0#3, dense<4> : tensor<i32>
  func.return
}

// -----

// Reverse cumulative sum (suffix sums) via chlo.scan with is_reverse: the scan
// runs from the end, so %0#0 is output[i] = sum(input[i:]) and %0#1 is the
// total carry.
func.func @scan_cumsum_reverse() {
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = chlo.scan(%input) inits(%init) dimension=0 attributes {is_reverse = true} {
  ^bb0(%scan_arg0: tensor<i32>, %scan_arg1: tensor<i32>):
    %1 = stablehlo.add %scan_arg0, %scan_arg1 : tensor<i32>
    stablehlo.return %1, %1 : tensor<i32>, tensor<i32>
  } : (tensor<4xi32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
  check.expect_eq_const %0#0, dense<[10, 9, 7, 4]> : tensor<4xi32>
  check.expect_eq_const %0#1, dense<10> : tensor<i32>
  func.return
}
