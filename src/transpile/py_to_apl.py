"""
Python-to-APL transpilation.
"""


import ast
import re
from itertools import product


# Besides Inline, these are standard APL tokens and their aplparse identifiers.
NAME_TO_GLYPH = {
    'Inline': '', # Inlines a piece of APL code as-is.
    'Add': '+',
    'Alpha': '⍺',
    'And': '∧',
    'Cat': ',',
    'Circ': '○',
    'Circstar': '⍟',
    'Disclose': '⊃',
    'Div': '÷',
    'Drop': '↓',
    'Enclose': '⊂',
    'Eq': '=',
    'Gradedown': '⍒',
    'Gradeup': '⍋',
    'Gt': '>',
    'Gteq': '≥',
    'In': '∊',
    'Iota': '⍳',
    'Lt': '<',
    'Lteq': '≤',
    'Match': '≡',
    'Max': '⌈',
    'Min': '⌊',
    'Nand': '⍲',
    'Nmatch': '≢',
    'Nor': '⍱',
    'Omega': '⍵',
    'Or': '∨',
    'Pipe': '|',
    'Pow': '*',
    'Rho': '⍴',
    'Rot': '⌽',
    'Squad': '⌷',
    'Sub': '-',
    'Take': '↑',
    'Tilde': '~',
    'Times': '×',
    'Trans': '⍉',
    'Vcat': '⍪',
    'Vrot': '⊖',
}
OP_TO_GLYPH = {
    'Backslash': '\\', # Scan isn't supported for autodiff, but it's used in the derivative of reduce and reduce first.
    'Dot': '.',
    'JotDia': '⍤',
    'Slash': '/', # Slash as a function, namely, replicate, isn't supported.
    'Slashbar': '⌿', # Slashbar as a function, namely, replicate first, isn't supported.
}

# Appends the various possible valence combinations.
# Many entries will be invalid, but that's not an issue.
for name in list(NAME_TO_GLYPH.keys()):
    NAME_TO_GLYPH[name+'Dy'] = NAME_TO_GLYPH[name]
for op in list(OP_TO_GLYPH.keys()):
    for suff in product(['Mon', 'Dy'], ['Mon', 'Dy']):
        OP_TO_GLYPH[op+''.join(suff)] = OP_TO_GLYPH[op]

NAME_TO_GLYPH = {**NAME_TO_GLYPH, **OP_TO_GLYPH}


def is_op(call: ast.Call) -> bool:
    """
    Checks if a call is an operator or not.
    """
    return hasattr(call.func, 'id') and call.func.id in OP_TO_GLYPH


class Unparse:
    """
    Unparses Python AST into APL. Each method is self-explanatory
    and receives a node, returning the unparsed APL code.
    """
    @staticmethod
    def node(node: ast.AST) -> str:
        return getattr(Unparse, node.__class__.__name__)(node)

    @staticmethod
    def FunctionDef(func: ast.FunctionDef) -> str:
        unparsed = [func.name + '←{']
        for node in func.body:
            unparsed.append('    ' + Unparse.node(node))
        unparsed.append('}')
        return '\n'.join(unparsed)

    @staticmethod
    def Assign(assign: ast.Assign) -> str:
        val = Unparse.node(assign.value)
        if assign.targets[0].id in NAME_TO_GLYPH:
            raise RuntimeError(f'Variable name {assign.targets[0].id} is illegal.')

        # If the unparsed value is a tuple, the assignment is selective,
        # as described in the Call method.
        return (f'{val[0]} ⋄ {assign.targets[0].id}←{val[1]}'
                if isinstance(val, tuple) else
                f'{assign.targets[0].id}←{Unparse.node(assign.value)}')

    @staticmethod
    def Call(call: ast.Call) -> str:
        func = Unparse.node(call.func)
        args = [Unparse.node(arg) for arg in call.args]

        # Selective assignment is a special function used to indicate
        # the second argument is assigned to the selection expression
        # in the first argument, and the third argument is subsequently returned.
        # That is, a = SelectiveAssign(Func(x), y, x) first performs
        # the selective assignment Func(x) = y, and then sets a to x.
        # The return statement below returns Func(x) = y and a = x as a pair (see
        # also src/autodiff.py).
        if hasattr(call.func, 'id') and call.func.id == 'SelectiveAssign':
            return (f'({args[0]})←{args[1]}',
                    args[2])

        if is_op(call):
            if call.func.id.endswith('MonMon'):
                return f'{args[0]}{func}{args[1]}'

            if call.func.id.endswith('MonDy'):
                return f'({args[1]}){args[0]}{func}{args[2]}'

            if call.func.id.endswith('DyMon'):
                return f'({args[0]}{func}{args[1]}){args[2]}'

            if call.func.id.endswith('DyDy'):
                return f'({args[2]})({args[0]}{func}{args[1]}){args[3]}'

        return (f'{func}{args[0]}' if len(call.args) == 1 else
                f'({args[0]}){func}{args[1]}')

    @staticmethod
    def Constant(const: ast.Constant) -> str:
        return const.value

    @staticmethod
    def Name(name: ast.Name) -> str:
        return (NAME_TO_GLYPH[name.id] if name.id in NAME_TO_GLYPH
                else name.id)

    @staticmethod
    def Return(ret: ast.Return) -> str:
        return Unparse.node(ret.value)

    @staticmethod
    def UnaryOp(uop: ast.UnaryOp):
        if not isinstance(uop.op, ast.USub):
            raise RuntimeError(f'Unary operation {uop.op} is invalid.')
        return f'¯{Unparse.node(uop.operand)}'


def py_to_apl(py: str) -> str:
    """
    Transpiles a Python function into an APL dfn.

    Args:
        py: Python function to transpile into APL.

    Returns:
        Source code of the transpiled APL dfn.
    """
    # Due to a bug in Tangent, variables might be assigned the names of operators.
    # This regex appends a suffix to them to avoid conflicts.
    for op in OP_TO_GLYPH:
        py = re.sub(rf'\b{op}\b(?!\()', rf'{op}_var_name', py)

    apl = Unparse.node(ast.parse(py).body[0])
    return re.sub(r'\(([a-zA-Z0-9_⍺⍵¯]+)\)', r'\1', apl) # Strips redundant parentheses away.
