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
   sudo apt install cmake ninja-build lld ccache

   # On macOS
   brew install cmake ninja ccache
   ```

2. Clone the StableHLO repo and the LLVM repository:

   ```sh
   git clone https://github.com/openxla/stablehlo
   ```

   ```sh
   cd stablehlo && git clone https://github.com/llvm/llvm-project.git
   ```

   Cloning the LLVM repository may take a few minutes.

3. Make sure you check out the correct commit in the LLVM repository:

   ```sh
   (cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))
   ```

   You need to do this every time `llvm_version.txt` changes.

4. Build StableHLO as a standalone library and run all the tests:

   ```sh
   # first configure the build system
   cmake --preset debug
   # then build the project
   cmake --build ./build --target check-stablehlo-ci
   ```

   You should see results like this:

   ```txt
   Testing Time: 4.13s

   Total Discovered Tests: 137
   Passed: 137 (100.00%)
   ```

   This runs all the tests in `stablehlo/tests/`. You can change the target
   to build or test specific parts of the project.

## Python

If you'd like to build the Python bindings, you'll need to install a few
additional dependencies.

```sh
pip install  install -r ./llvm-project/mlir/python/requirements.txt
```

If you've built MLIR & StableHLO using the script above, the Python bindings
for MLIR may already built.

After you have built the project you can import the Python bindings to begin
by modifying your Python path variable

```shell
$ PYTHONPATH="./build/python_packages/stablehlo" python3
Python 3.11.6 (main, Oct  8 2023, 05:06:43) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mlir.dialects.stablehlo
>>> from mlir.ir import Context, Location
>>> import mlir.dialects.arith
```

You can also build a wheel yourself using the `setup.py` file.
We also make nightly wheels available on our GitHub Releases page.

```shell
pip install stablehlo -f https://github.com/openxla/stablehlo/releases/expanded_assets/dev-wheels
```

## Community

Building an amazing portability layer between ML frameworks and ML compilers
requires collaboration across the whole ML industry, so we're happy to have
your help on the StableHLO project.

We're using GitHub issues / pull requests to organize development and
[openxla-discuss](https://groups.google.com/a/openxla.org/g/openxla-discuss/)
to have longer discussions. We also have a `#stablehlo`
channel on [the OpenXLA Discord server](https://discord.gg/PeWUTaecrA).
