"""Copyright 2024 The StableHLO Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Construct FileCheck comments from a MLIR source.

Usage:
  build/bin/stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file --verify-diagnostics \
    stablehlo/tests/chlo/chlo_legalize_to_stablehlo.mlir \
  | python build_tools/math/mlir_to_filecheck.py
"""

import re
import sys

state = dict()
tmp_counter = 0
stage = 0

for orig_line in sys.stdin:
  line = orig_line.strip()
  if line.startswith('//') or not line:
    print(orig_line)
  elif line == '}':
    assert stage > 0
    stage -= 1
  elif stage == 0:
    assert line == 'module {', line
    stage = 1
    state.clear()
    tmp_counter = 0
  elif stage == 1:
    assert line.startswith('func.func'), line
    stage = 2
    i = line.index('(')
    j = line.index(')')
    k = line.index('{')
    assert i > 0, line
    assert j > 0, line
    assert k > 0, line
    args_line = line[i + 1:j].strip()
    args = []
    arg_count = 0
    while args_line:
      n = args_line.find("%", 1)
      if n == -1:
        n = len(args_line)
      a, atyp = args_line[:n].split(":", 1)
      arg = a.strip()
      atyp = atyp.strip()
      if atyp.endswith(','):
        atyp = atyp[:-1].strip()
      state[arg] = f'TMP_arg{arg_count}'
      args.append(f'%[[{state[arg]}:.*]]: {atyp}')
      arg_count += 1
      if n == -1:
        break
      args_line = args_line[n:]
    args = ', '.join(args)
    print(f'// CHECK-LABEL:  {line[:i + 1]}')
    print(f'// CHECK-SAME:   {args}{line[j:k].rstrip()}')
  elif stage == 2:
    if '=' in line:
      left, right = line.split('=', 1)
      left, right = left.strip(), right.strip()
      for k, v in state.items():
        right = re.sub(k + r'\b', f'%[[{v}]]', right)
      state[left] = f'TMP_{tmp_counter}'
      tmp_counter += 1
      print(f'// CHECK:   %[[{state[left]}:.*]] = {right}')
    else:
      for k, v in state.items():
        line = re.sub(k + r'\b', f'%[[{v}]]', line)
      print(f'// CHECK:   {line}')
