# StableHLO

StableHLO is an operation set for high-level operations (HLO) in machine
learning (ML) models. Essentially, it's a portability layer between different
ML frameworks and ML compilers: ML frameworks that produce StableHLO programs
are compatible with ML compilers that consume StableHLO programs.

Our goal is to simplify and accelerate ML development by creating more
interoperability between various ML frameworks (such as TensorFlow, JAX and
PyTorch) and ML compilers (such as XLA and IREE).

StableHLO is based on the MHLO dialect and enhances it with additional
functionality, including serialization and versioning. This provides
[backward and forward
compatibility](https://github.com/openxla/stablehlo/blob/main/rfcs/20220912-compatibility.md)
guarantees for StableHLO programs and ensures compatibility between frameworks
and compilers, even as StableHLO continues to evolve.

This repository includes the [StableHLO
specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
along with an MLIR-based implementation in C++ and Python, which you can use to
define StableHLO programs for consumption by compilers such as XLA and IREE.

## Build steps

Here's how to build the StableHLO repo:

1. Make sure you have the LLVM-based linker `lld` installed:

   ```sh
   sudo apt update && sudo apt install lld
   ```

2. Clone this repo and the LLVM git repository:

   ```sh
   git clone https://github.com/openxla/stablehlo
   ```

   ```sh
   cd stablehlo && git clone https://github.com/llvm/llvm-project.git
   ```

3. Make sure you check out the correct commit in the LLVM repository:

   ```sh
   (cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))
   ```

   You need to do this every time `llvm_version.txt` changes.

4. Configure and build MLIR:

   ```sh
   build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build
   ```

   This will take several minutes.

   Again, you need to do this every time `llvm_version.txt` changes.

5. Build StableHLO as a standalone library:

   ```sh
   mkdir -p build && cd build

   cmake .. -GNinja \
     -DLLVM_ENABLE_LLD=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=On \
     -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
   ```

6. Now you can make sure it works by running some tests:

   ```sh
   ninja check-stablehlo
   ```

   You should see results like this:

   ```txt
   Testing Time: 5.99s
     Passed: 47
   ```

   This runs all the tests in `stablehlo/tests/`.

## Community

Building an amazing portability layer between ML frameworks and ML compilers
requires collaboration across the whole ML industry, so we're happy to have
your help on the StableHLO project.

We're using GitHub issues / pull requests to organize development and
[GitHub discussions](https://github.com/orgs/openxla/discussions/categories/stablehlo)
to have longer discussions. We also have a `#stablehlo`
channel on [the OpenXLA Discord server](https://discord.gg/PeWUTaecrA).

## Roadmap

* Workstream #1: Stable version of HLO/MHLO, including
  [the spec](https://github.com/openxla/stablehlo/labels/Spec),
  the corresponding dialect with high-quality implementations of
  [prettyprinting](https://github.com/openxla/stablehlo/labels/Prettyprinting),
  [verification](https://github.com/openxla/stablehlo/labels/Verification) and
  [type inference](https://github.com/openxla/stablehlo/labels/Type%20inference),
  and [the interpeter](https://github.com/openxla/stablehlo/labels/Interpreter).
  ETA: H2 2022.
* Workstream #2: Evolution beyond what's currently in HLO/MHLO.
  Ongoing work on [dynamism](https://github.com/openxla/stablehlo/labels/Dynamism),
  sparsity, quantization and extensibility. ETA: H2 2022.
* Workstream #3: Support for ML frameworks (TensorFlow, JAX, PyTorch) and
  ML compilers (XLA and IREE). ETA: H2 2022.
