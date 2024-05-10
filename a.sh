#!/bin/bash

# Update build files
LLVM_REV=$(curl https://raw.githubusercontent.com/tensorflow/mlir-hlo/master/build_tools/llvm_version.txt | tr -d '\n')
echo "$LLVM_REV" > ./build_tools/llvm_version.txt
./build_tools/github_actions/lint_llvm_commit.sh -f .

# Print the commit message
echo "Commit changes with message:"
echo "git add ."
echo "git commit -m \"Integrate LLVM at llvm/llvm-project@${LLVM_REV:0:12}\""
