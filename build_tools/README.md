# Build instructions

To build the code in this repository, you need a clone of the LLVM/MLIR git
repository.

This repository includes a submodule checked out to the proper commit that can
be used. To downloaded at clone time use:

`$ git clone --recurse-submodules https://github.com/openxla/stablehlo.git`

Or after clone using:

`$ git submodule update --init --recursive`

We provide a script to configure and build LLVM/MLIR:

`$ build_tools/build_mlir.sh ${PWD}/external/llvm-project ${PWD}/llvm-build`

This build step must be completed every time you pull from this repository and
the LLVM revision changes.

Finally you can build and test this repository:

```
$ mkdir build && cd build
$ cmake .. -GNinja \
   -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=On \
   -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
$ ninja check-stablehlo
```
