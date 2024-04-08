# IDE setup

You can find on this page some _opinionated_ IDE setup instructions.
Of course the best IDE is the one that _works for you_.

> If you have an improvement or recommendation to any of the setups, we welcome contributions.

## Visual Studio Code (vscode)

### CMake

Visual Studio Code (vscode) can work pretty well with the CMake build system.

The following extensions are recommended:

* [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
* [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)

> Note: Installing the `clangd` extension will inform you that you
will need to disable the default _intellisense_ extension.
This is fine, as `clangd` will provide the same functionality.

We include a [CmakePresets.json](../CMakePresets.json) file in the root of the repository.
This file is used by the `CMake Tools` extension to provide a list of _presets_ that
can be used to configure the build.  The `CMake Tools` extension will automatically
detect this file and provide the presets.

Additionally, all the configured presets generate the `compile_commands.json` file
in the build directory which will then be picked up by `clangd`.

We recommend additionally setting the following in your `.vscode/settings.json` file:

```json
{
    "files.exclude": {
        "**/bazel-*": true
    }
}
```

## Vim

### LLVM/MLIR settings

Check out the official instructions for [LLVM](https://github.com/llvm/llvm-project/blob/main/llvm/utils/vim/README)
and [MLIR](https://github.com/llvm/llvm-project/blob/main/mlir/utils/vim/README)
settings to enable syntax highlighting and other goodies.

### IDE-like features

Check out the official [documentation for
`clangd`](https://releases.llvm.org/9.0.1/tools/clang/tools/extra/docs/clangd/Installation.html)
to enable features like autocompletion, go to definition, etc.
