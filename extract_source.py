#!/usr/bin/env python3

import ast
import inspect
import argparse
import importlib.util
import sys
from types import ModuleType


# ---------- Load module from file ----------
def load_module_from_file(filepath: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("target_module", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["target_module"] = module
    spec.loader.exec_module(module)
    return module


# ---------- AST: find called function names ----------
class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = set()

    def visit_Call(self, node):
        # foo()
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)

        # obj.method()
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)

        self.generic_visit(node)


def get_called_functions(func):
    try:
        source = inspect.getsource(func)
    except OSError:
        return set()

    tree = ast.parse(source)
    visitor = CallVisitor()
    visitor.visit(tree)
    return visitor.calls


# ---------- Recursive collection ----------
def collect_sources(func, module, seen=None):
    if seen is None:
        seen = {}

    if func in seen:
        return seen

    try:
        src = inspect.getsource(func)
    except OSError:
        return seen  # skip builtins / C funcs

    seen[func] = src

    called_names = get_called_functions(func)

    for name in called_names:
        obj = getattr(module, name, None)

        if callable(obj):
            collect_sources(obj, module, seen)

    return seen


# ---------- Main CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Extract source code of a function and all called functions"
    )
    parser.add_argument("--file", required=True, help="Python file")
    parser.add_argument("--function", required=True, help="Function name")

    args = parser.parse_args()

    module = load_module_from_file(args.file)

    if not hasattr(module, args.function):
        print(f"Function '{args.function}' not found in {args.file}")
        sys.exit(1)

    func = getattr(module, args.function)

    if not callable(func):
        print(f"'{args.function}' is not callable")
        sys.exit(1)

    sources = collect_sources(func, module)

    # Print results
    printed = set()
    for f, src in sources.items():
        name = f.__name__
        if name not in printed:
            print(f"# ===== Function: {name} =====")
            print(src)
            print()
            printed.add(name)


if __name__ == "__main__":
    main()
