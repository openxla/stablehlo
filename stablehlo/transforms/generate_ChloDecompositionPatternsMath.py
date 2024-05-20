"""A script to generate ChloDecompositionPatternsMath.td.

Prerequisites:
  python 3.11 or newer
  functional_algorithms 0.1 or newer

Usage:
  Running
    python /path/to/generate_ChloDecompositionPatternsMath.py
  will create
    /path/to/ChloDecompositionPatternsMath.td
"""

import os
import warnings


def main():
    try:
        import functional_algorithms as fa
    except ImportError as msg:
        print(f'Skipping: {msg}')
        return

    output_file = os.path.join(os.path.dirname(__file__),
                               "ChloDecompositionPatternsMath.td")

    sources = []
    target = fa.targets.stablehlo
    for chloname, fname, args in [
        ('CHLO_AsinOp', 'complex_asin', ("z:complex",)),
        ('CHLO_AsinOp', 'real_asin', ("x:float",)),
    ]:
        func = getattr(fa.algorithms, fname, None)
        if func is None:
            warnings.warn(
                '{fa.algorithms.__name__} does not define {fname}. Skipping.')
            continue
        ctx = fa.Context(paths=[fa.algorithms])
        graph = ctx.trace(
            func,
            *args).implement_missing(target).simplify()._props(name=chloname)
        src = graph.tostring(target)
        sources.append(target.make_comment(
            func.__doc__)) if func.__doc__ else None
        sources[-1] += src
    source = "\n\n".join(sources) + "\n"

    if os.path.isfile(output_file):
        f = open(output_file, "r")
        content = f.read()
        f.close()
        if content.endswith(source):
            print(f"{output_file} is up-to-date.")
            return

    f = open(output_file, "w")
    f.write(
        target.make_comment(f"""\
This file is generated using functional_algorithms tool ({fa.__version__}), see
  https://github.com/pearu/functional_algorithms
for more information.""") + "\n")
    f.write(source)
    f.close()
    print(f"Created {output_file}")


if __name__ == '__main__':
    main()
