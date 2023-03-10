# StableHLO

StableHLO is an operation set for high-level operations (HLO) in machine
learning (ML) models. Essentially, it's a portability layer between different
ML frameworks and ML compilers: ML frameworks that produce StableHLO programs
are compatible with ML compilers that consume StableHLO programs.

Our goal is to simplify and accelerate ML development by creating more
interoperability between various ML frameworks (such as TensorFlow, JAX and
PyTorch) and ML compilers (such as XLA and IREE).

StableHLO is based on the MHLO dialect and enhances it with additional
functionality, including serialization and versioning. We use MLIR bytecode
as [serialization format](docs/bytecode.md) and provide [backward and forward
compatibility](docs/compatibility.md) guarantees. This ensures compatibility
between frameworks and compilers, even as StableHLO continues to evolve.

This repository includes the [StableHLO specification](docs/spec.md)
along with an MLIR-based implementation in C++ and Python, which you can use to
define StableHLO programs for consumption by compilers such as XLA and IREE.

## Build instructions

Here's how to build the StableHLO repo on Linux or macOS:

1. CMake is our primary build tool, so before you begin make sure that
   you have CMake and Ninja installed.

   If you're using Linux, we recommend installing `lld` as well - we have
   observed it to be noticeably faster than alternatives on our typical software
   and hardware configurations.

   ```sh
   # On Linux
   sudo apt install cmake ninja-build lld

   # On macOS
   brew install cmake ninja
   ```

2. Set the `LLVM_ENABLE_LLD` shell variable depending on your preferences. We
   recommend setting it to `ON` on Linux and to `OFF` on macOS.

   ```sh
   [[ "$(uname)" != "Darwin" ]] && LLVM_ENABLE_LLD="ON" || LLVM_ENABLE_LLD="OFF"
   ```

3. Clone the StableHLO repo and the LLVM repository:

   ```sh
   git clone https://github.com/openxla/stablehlo
   ```

   ```sh
   cd stablehlo && git clone https://github.com/llvm/llvm-project.git
   ```

   Cloning the LLVM repository may take a few minutes.

4. Make sure you check out the correct commit in the LLVM repository:

   ```sh
   (cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))
   ```

   You need to do this every time `llvm_version.txt` changes.

5. Configure and build MLIR:

   ```sh
   build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build
   ```

   This will take a considerable amount of time. For example, on a MacBook Pro
   with an M1 Pro chip, building MLIR took around 10 minutes at the moment
   of writing.

   Again, you need to do this every time `llvm_version.txt` changes.

6. Build StableHLO as a standalone library:

   ```sh
   mkdir -p build && cd build

   cmake .. -GNinja \
     -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=On \
     -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
   ```

7. Now you can make sure it works by running some tests:

   ```sh
   ninja check-stablehlo-tests
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
[openxla-discuss](https://groups.google.com/a/openxla.org/g/openxla-discuss/)
to have longer discussions. We also have a `#stablehlo`
channel on [the OpenXLA Discord server](https://discord.gg/PeWUTaecrA).
