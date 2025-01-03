"""
APL-to-Python transpilation.
"""


import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union

import astor


Node = Dict[str, Union[str, List]]


def tree_str_to_tree(tree_str: str) -> Node:
    """
    Constructs a tree given its string representation.

    Args:
        tree_str: String representation of tree.

    Returns:
        Tree constructed from the string representation.
    """
    # Builds the tree in a depth-first fashion.
    # A parser or recursive regex approach could've been used as well.
    def helper(idx: int) -> Dict[str, Union[str, List]]:
        start_idx = idx
        while idx < len(tree_str) and tree_str[idx] not in '(,)':
            idx += 1
        name = tree_str[start_idx:idx].strip()

        children = []
        if idx < len(tree_str) and tree_str[idx] == '(':
            idx += 1
            while idx < len(tree_str) and tree_str[idx] != ')':
                child, idx = helper(idx)
                children.append(child)
                if idx < len(tree_str) and tree_str[idx] == ',':
                    idx += 1
            idx += 1

        return {'name': name, 'children': children}, idx

    # Inlines vector literals.
    # Strand notation only works with literals.
    tree_str = re.sub(r'Vec\[(.*?)\]', r'Inline(\1)', tree_str)

    # Removes valence information, which is later reintegrated into the tree via
    # DyTransformer and OpFusionTransformer below, and AutodiffTransformer
    # in src/autodiff.py.
    tree_str = ' '.join(re.sub(r'\[\d+(,\d+)*\]', '', tree_str).split())
    tree_str = tree_str.replace('[', '').replace(']', '')

    return helper(0)[0]


class Unparse:
    """
    Unparses apl-tree nodes into Python. Each method is self-explanatory
    and receives a node's arguments, returning the unparsed Python code.
    """
    @staticmethod
    def node(node: Node) -> str:
        return (getattr(Unparse, node['name'])(*node['children'])
                if node['children'] else node['name'])

    # This type of node corresponds to Vec from aplparse and
    # inlines its argument as-is when transpiling Python back into APL.
    # It can inline any arbitrary piece of APL code (e.g., dfns) and not merely
    # vectors (see also src/autodiff.py).
    @staticmethod
    def Inline(*vals) -> str:
        return f'Inline("{" ".join([val["name"] for val in vals])}")'

    @staticmethod
    def App1(fun: Node, right: Node) -> str:
        return f'{Unparse.node(fun)}({Unparse.node(right)})'

    @staticmethod
    def App2(fun: Node, left: Node, right: Node) -> str:
        return f'{Unparse.node(fun)}({Unparse.node(left)}, {Unparse.node(right)})'

    @staticmethod
    def AppOpr1(opr: Node, left: Node) -> str:
        return f'{opr["name"]}({Unparse.node(left)})'

    @staticmethod
    def AppOpr2(opr: Node, left: Node, right: Node) -> str:
        return f'{opr["name"]}({Unparse.node(left)}, {Unparse.node(right)})'

    @staticmethod
    def Assign(var: Node, val: Node) -> str:
        prefix = (f'def {var["name"]}(Omega):\n' if val['name'] == 'Lam'
                    else f'{var["name"]} = ' )
        return f'{prefix}{Unparse.node(val)}'

    @staticmethod
    def Lam(*body) -> str:
        unparsed = []
        for i, node in enumerate(body):
            prefix = '    return ' if i == len(body)-1 else '    '
            unparsed.append(prefix + Unparse.node(node))
        return '\n'.join(unparsed)


class DyTransformer(ast.NodeTransformer):
    """
    Appends a 'Dy' suffix to dyadic calls.
    """
    def visit_Call(self, node):
        if len(node.args) == 2 and isinstance(node.func, ast.Name):
            node.func.id += 'Dy'
        self.generic_visit(node)
        return node


class OpFusionTransformer(ast.NodeTransformer):
    """
    Fuses an operator application and the subsequent function call, followed by
    a suffix representing the valences of the operator and the derived function.
    For example, Op(f)(left, right) becomes OpMonDy(f, left, right).
    """
    def visit_Call(self, node):
        # The arguments need to be visited first since they
        # themselves might be operator applications.
        node.func = self.visit(node.func)
        node.args = [self.visit(arg) for arg in node.args]

        if isinstance(node.func, ast.Call):
            # Suffix-free functions are implicitly understood to be monadic.
            # Operators are more ambiguous, so their valence is explicitly stated,
            # even when they're monadic.
            # Dyadic operators are assumed to already have had their valence added
            # by DyTransformer.
            if len(node.func.args) == 1:
                node.func.func.id += 'Mon'

            new_call = ast.Call(func=node.func.func,
                                args=node.func.args + node.args,
                                keywords=[])
            # This is the valence of the derived function, not the operator itself.
            new_call.func.id += 'Mon' if len(node.args) == 1 else 'Dy'

            return ast.copy_location(new_call, node)

        return self.generic_visit(node)


def apl_to_py(apl: str, aplparse: str) -> str:
    """
    Transpiles an APL dfn into a Python function.

    Args:
        apl: APL dfn to transpile into Python.
        aplparse: Path to aplparse executable.

    Returns:
        Source code of transpiled Python function.

    Raises:
        CalledProcessError: The parser failed to execute.
    """
    # aplparse expects an APL source file, so the APL code is
    # temporarily saved to disk and afterwards deleted.
    with open(f'tmp.apl', 'w+') as f:
        f.write(apl)

    try:
        result = subprocess.run([f'./{aplparse}', 'tmp.apl'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f'Parsing failed: {e}')
        sys.exit(1)

    finally:
        Path('tmp.apl').unlink()

    tree_str = result.stdout.strip()
    tree = tree_str_to_tree(tree_str)

    # Though the two transformers can be merged, they're kept separate for simplicity.
    py_ast = ast.parse(Unparse.node(tree))
    py_ast = DyTransformer().visit(py_ast)
    py_ast = OpFusionTransformer().visit(py_ast)

    return astor.to_source(py_ast)
