#!/usr/bin/env python3

import ast
import inspect
import argparse
import importlib.util
import sys
import os
from types import ModuleType

# Keywords that usually signal 3D transformation logic
TRANSFORM_KEYWORDS = {
    "projection", "intrinsics", "extrinsics", "rotation", "translation", 
    "homography", "matmul", "rvec", "tvec", "depth", "camera_matrix",
    "pinhole", "perspective", "euler", "quaternion", "reshape(-1, 3)"
}

class AdvancedVisitor(ast.NodeVisitor):
    """
    Scans code for function calls, method calls, and matrix operators.
    """
    def __init__(self):
        self.calls = set()
        self.found_matrix_math = False

    def visit_Call(self, node):
        # Matches: func()
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        # Matches: obj.method()
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # Matches: A @ B (Matrix Multiplication)
        if isinstance(node.op, ast.MatMult):
            self.found_matrix_math = True
        self.generic_visit(node)

def get_logic_metadata(func):
    """
    Extracts called names and checks for matrix math.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return set(), False

    tree = ast.parse(source)
    visitor = AdvancedVisitor()
    visitor.visit(tree)
    return visitor.calls, visitor.found_matrix_math

def find_in_module(module, name):
    """
    Attempts to find a callable in the module, even if it's inside a class.
    """
    # 1. Check top-level
    obj = getattr(module, name, None)
    if callable(obj) and not inspect.isclass(obj):
        return obj
    
    # 2. Check inside classes defined in this module
    for attr_name in dir(module):
        cls = getattr(module, attr_name)
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            method = getattr(cls, name, None)
            if callable(method):
                return method
    return None

def collect_sources(func, module, seen=None):
    if seen is None:
        seen = {}

    if func in seen:
        return seen

    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return seen  # Skip built-ins

    seen[func] = src
    called_names, has_matmul = get_logic_metadata(func)

    for name in called_names:
        target_obj = find_in_module(module, name)
        if target_obj:
            collect_sources(target_obj, module, seen)

    return seen

def highlight_3d_logic(src):
    """Adds a visual marker if 3D keywords are found in the source block."""
    found = [k for k in TRANSFORM_KEYWORDS if k.lower() in src.lower()]
    if found:
        return f"   # [!] POTENTIAL 3D LOGIC DETECTED: {', '.join(found)}"
    return ""

def load_module_from_file(filepath: str) -> ModuleType:
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(
        description="Deep-dive extraction of 3D transformation logic from Python source."
    )
    parser.add_argument("--file", required=True, help="Path to the .py file")
    parser.add_argument("--function", required=True, help="Starting function (e.g., 'main')")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        sys.exit(1)

    module = load_module_from_file(args.file)
    start_func = find_in_module(module, args.function)

    if not start_func:
        print(f"Error: Could not find function/method '{args.function}' in {args.file}")
        sys.exit(1)

    sources = collect_sources(start_func, module)

    print(f"\n{'='*60}")
    print(f" SOURCE TREE FOR: {args.function}")
    print(f"{'='*60}\n")

    for func_obj, src in sources.items():
        name = getattr(func_obj, '__qualname__', func_obj.__name__)
        alert = highlight_3d_logic(src)
        
        # Check if the function itself uses the @ operator
        _, uses_matmul = get_logic_metadata(func_obj)
        matmul_alert = "   # [!] USES MATRIX MULTIPLICATION (@)" if uses_matmul else ""

        print(f"### {name} ###")
        if alert: print(alert)
        if matmul_alert: print(matmul_alert)
        print("-" * len(name))
        print(src.strip())
        print("\n")

if __name__ == "__main__":
    main()
