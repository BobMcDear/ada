"""
Source-to-source autodiff for APL.
"""


import ast
import imp
import re
import sys
from inspect import getsource
from pathlib import Path

import astor
import tangent

from .transpile import apl_to_py, py_to_apl


# Refer to the first comment in autodiff regarding why these aren't declared.
# None of the primitives are actually defined, and everything is purely symbolic.
# This is thanks to the adjoints being exclusively composed of APL primitives,
# which will later be transpiled anyhow and don't need to be run in Python.
# For example, nowehere are PowDy, TimesDy, and SubDy mathematically described,
# and the semantics of each is unknown to Tangent and the rest of the codebase.
# However, the derivative of PowDy(x, a) - that is, x^a - with respect to x
# is expressed as TimesDy(a, PowDy(x, SubDy(a, 1))) - that is, a * x^(a-1) -
# which is all that's necessary to conduct autodiff.
# There are three special functions used in addition to APL primitives:
# I) Inline: Inlines an arbitrary expression as-is when transpiling
# Python back into APL (see also src/transpile/apl_to_py.py and AutodiffTransformer below).
# II) SelectiveAssign: Selective assigns a value to a variable given
# a selection expression (see also src/transpile/py_to_apl.py).
# III) Autodiff: Differentiates the primitive supplied by the first argument
# w.r.t. the index supplied by the second argument (see also AutodiffTransformer below).
prims_str = """
from tangent.grads import adjoint


# Functions.
def Inline(_): ...
def AddDy(_, __): ...
def AndDy(_, __): ...
def Cat(_): ...
def CatDy(_, __): ...
def Circ(_): ...
def CircDy(_, __): ...
def Circstar(_): ...
def CircstarDy(_, __): ...
def Disclose(_): ...
def DiscloseDy(_, __): ...
def Div(_): ...
def DivDy(_, __): ...
def DropDy(_, __): ...
def Enclose(_): ...
def EqDy(_, __): ...
def Gradedown(_): ...
def Gradeup(_): ...
def GtDy(_, __): ...
def GteqDy(_, __): ...
def In(_): ...
def Iota(_): ...
def LtDy(_, __): ...
def LteqDy(_, __): ...
def Match(_): ...
def MatchDy(_, __): ...
def Max(_): ...
def MaxDy(_, __): ...
def Min(_): ...
def MinDy(_, __): ...
def NandDy(_, __): ...
def Nmatch(_): ...
def NmatchDy(_, __): ...
def NorDy(_, __): ...
def OrDy(_, __): ...
def Pipe(_): ...
def Pow(_): ...
def PowDy(_, __): ...
def Rho(_): ...
def RhoDy(_, __): ...
def Rot(_): ...
def RotDy(_, __): ...
def SquadDy(_, __): ...
def Sub(_): ...
def SubDy(_, __): ...
def TakeDy(_, __): ...
def Tilde(_): ...
def Times(_): ...
def TimesDy(_, __): ...
def Trans(_): ...
def TransDy(_, __): ...
def Vcat(_): ...
def VcatDy(_, __): ...
def Vrot(_): ...
def VrotDy(_, __): ...

# Operators.
def DotDyDy(_, __, ___, ____): ...
def JotDiaDyMon(_, __, ___): ...
def JotDiaDyDy(_, __, ___, ____): ...
def SlashMonMon(_, __): ...
def SlashbarMonMon(_, __): ...


# Adjoints of differentiable functions.
@adjoint(AddDy)
def dAddDy(y, left, right):
    d[left] = d[y]
    d[right] = d[y]


@adjoint(Cat)
def dCat(y, right):
    d[right] = RhoDy(Rho(right), d[y])


@adjoint(CatDy)
def dCatDy(y, left, right):
    dyT = Trans(d[y])
    n = Nmatch(Trans(left))
    d[left] = Trans(TakeDy(n, dyT))
    d[right] = Trans(DropDy(n, dyT))


@adjoint(Circ)
def dCirc(y, right):
    d[right] = TimesDy(d[y], Circ(1))


@adjoint(CircDy)
def dCircDy(y, left, right):
    d[right] = TimesDy(
                   d[y],
                   TimesDy(
                       Times(SubDy(1.5, left)), # Sin -> 1, cos -> -1
                       CircDy(
                           AddDy(Tilde(SubDy(left, 1)), 1), # Sin -> cos, cos -> sin
                           right
                       ),
                    ),
                )


@adjoint(Circstar)
def dCircstar(y, right):
    d[right] = Div(d[y], right)


@adjoint(CircstarDy)
def dCircstarDy(y, left, right):
    d[left] = TimesDy(
                  d[y],
                  Sub(
                      DivDy(
                          CircstarDy(left, right),
                          TimesDy(left, Circstar(left))
                      )
                  )
              )
    d[right] = DivDy(
                   d[y],
                   TimesDy(Circstar(left), right)
               )


@adjoint(Disclose)
def dDisclose(y, right):
    zeros = TimesDy(0, right)
    d[right] = SelectiveAssign(Disclose(zeros), d[y], zeros)


@adjoint(DiscloseDy)
def dDiscloseDy(y, left, right):
    zeros = TimesDy(0, right)
    d[right] = SelectiveAssign(DiscloseDy(left, zeros), d[y], zeros)


@adjoint(Div)
def dDiv(y, right):
    d[right] = TimesDy(
                   d[y],
                   Sub(
                       Div(TimesDy(right, right))
                   )
               )


@adjoint(DivDy)
def dDivDy(y, left, right):
    d[left] = DivDy(d[y], right)
    d[right] = TimesDy(
                   d[y],
                   Sub(
                       DivDy(left, TimesDy(right, right))
                   )
               )


@adjoint(DropDy)
def dDropDy(y, left, right):
    zeros = TimesDy(0, right)
    d[right] = SelectiveAssign(DropDy(left, zeros), d[y], zeros)


@adjoint(Enclose)
def dEnclose(y, right):
    d[right] = Disclose(d[y])


@adjoint(In)
def dIn(y, right):
    tmp = right
    d[right] = SelectiveAssign(In(tmp), d[y], tmp)


@adjoint(MaxDy)
def dMaxDy(y, left, right):
    d[left] = TimesDy(d[y], GteqDy(left, right))
    d[right] = TimesDy(d[y], GteqDy(right, left))


@adjoint(MinDy)
def dMinDy(y, left, right):
    d[left] = TimesDy(d[y], LteqDy(left, right))
    d[right] = TimesDy(d[y], LteqDy(right, left))


@adjoint(Pipe)
def dPipe(y, right):
    d[right] = TimesDy(d[y], Times(right))


@adjoint(Pow)
def dPow(y, right):
    d[right] = TimesDy(d[y], y)


@adjoint(PowDy)
def dPowDy(y, left, right):
    d[left] = TimesDy(
                  d[y],
                  TimesDy(
                      right,
                      PowDy(left, SubDy(right, 1))
                  )
              )
    d[right] = TimesDy(
                   d[y],
                   TimesDy(
                       Circstar(left),
                       PowDy(left, right)
                   )
               )


@adjoint(RhoDy)
def dRhoDy(y, _, right):
    d[right] = RhoDy(Rho(right), d[y])


@adjoint(Rot)
def dRot(y, right):
    d[right] = Rot(d[y])


@adjoint(RotDy)
def dRotDy(y, left, right):
    d[right] = RotDy(Sub(left), d[y])


@adjoint(SquadDy)
def dSquadDy(y, left, right):
    zeros = TimesDy(0, right)
    d[right] = SelectiveAssign(SquadDy(left, zeros), d[y], zeros)


@adjoint(Sub)
def dSub(y, right):
    d[right] = Sub(d[y])


@adjoint(SubDy)
def dSubDy(y, left, right):
    d[left] = d[y]
    d[right] = Sub(d[y])


@adjoint(TakeDy)
def dTakeDy(y, left, right):
    zeros = TimesDy(0, right)
    d[right] = SelectiveAssign(TakeDy(left, zeros), d[y], zeros)


@adjoint(TimesDy)
def dTimesDy(y, left, right):
    d[left] = TimesDy(d[y], right)
    d[right] = TimesDy(d[y], left)


@adjoint(Trans)
def dTrans(y, right):
    d[right] = Trans(d[y])


@adjoint(TransDy)
def dTransDy(y, left, right):
    zeros = TimesDy(0, right)
    d[right] = SelectiveAssign(TransDy(left, zeros), d[y], zeros)


@adjoint(Vcat)
def dVcat(y, right):
    d[right] = RhoDy(Rho(right), d[y])


@adjoint(VcatDy)
def dVcatDy(y, left, right):
    n = Nmatch(left)
    d[left] = TakeDy(n, d[y])
    d[right] = DropDy(n, d[y])


@adjoint(Vrot)
def dVrot(y, right):
    d[right] = Vrot(d[y])


@adjoint(VrotDy)
def dVrotDy(y, left, right):
    d[right] = VrotDy(Sub(left), d[y])


# Adjoints of operators.
@adjoint(DotDyDy)
def dDotDyDy(y, _, __, left, right):
    # Reshapes vectors or higher-rank arrays into matrices.
    dim_left = SlashMonMon(
                   TimesDy,
                   DropDy(-1, Rho(left))
               )
    dim_right = SlashMonMon(
                    TimesDy,
                    DropDy(1, Rho(right))
                )
    mat_left = RhoDy(
                   CatDy(
                       dim_left,
                       TakeDy(-1, Rho(left))
                   ),
                   left
               )
    mat_right = RhoDy(
                    CatDy(
                        TakeDy(1, Rho(right)),
                        dim_right
                    ),
                    right
                )
    mat_dy = RhoDy(Cat(dim_left, dim_right), d[y])

    # Computes the derivatives of the matrices, then converts them back
    # into their original shapes.
    d[left] = RhoDy(
                  Rho(left),
                  DotDyDy(AddDy, TimesDy, mat_dy, Trans(mat_right))
              )
    d[right] = RhoDy(
                   Rho(right),
                   DotDyDy(AddDy, TimesDy, Trans(mat_left), mat_dy)
               )


@adjoint(JotDiaDyMon)
def dJotDiaDyMon(y, op_left, op_right, right):
    d[right] = TimesDy(
                   d[y],
                   JotDiaDyMon(
                       Autodiff(op_left, 0, False),
                       op_right,
                       right
                   )
               )


@adjoint(JotDiaDyDy)
def dJotDiaDyDy(y, op_left, op_right, left, right):
    # These are the post-broadcasting derivatives.
    full_dleft = JotDiaDyDy(
                     TimesDy,
                     op_right,
                     d[y],
                     JotDiaDyDy(
                         Autodiff(op_left, 0, True),
                         op_right,
                         left,
                         right
                     )
                 )
    full_dright = JotDiaDyDy(
                      TimesDy,
                      op_right,
                      d[y],
                      JotDiaDyDy(
                          Autodiff(op_left, 1, True),
                          op_right,
                          left,
                          right
                      )
                  )

    # Since one argument might be of a smaller rank, the derivatives are
    # summed along the broadcasted axes.
    red_rank_dleft = SubDy(
                         Nmatch(Rho(full_dleft)),
                         Nmatch(Rho(left))
                     )
    red_rank_dright = SubDy(
                          Nmatch(Rho(full_dright)),
                          Nmatch(Rho(right))
                      )
    d[left] = Trans(
                  JotDiaDyMon(
                      Inline('{+/,⍵}'),
                      red_rank_dleft,
                      Trans(full_dleft)
                  )
              )
    d[right] = Trans(
                   JotDiaDyMon(
                       Inline('{+/,⍵}'),
                       red_rank_dright,
                       Trans(full_dright)
                   )
               )


@adjoint(SlashMonMon)
def dSlashMonMon(y, op, right):
    # Evaluates the intermediate results of the reduction.
    scan = BackslashMonMon(op, right)

    # Backpropagates the derivative through the intermediate results.
    chain = CatDy(
                Rot(
                    BackslashMonMon(
                        TimesDy,
                        JotDiaDyDy(
                            DropDy,
                            1,
                            1,
                            Rot(
                                Autodiff(op, 0, True)(scan, RotDy(1, right))
                            )
                        )
                    )
                ),
                1
            )

    # Calculates the derivatives of consecutive elements.
    cons = CatDy(
               1,
               JotDiaDyDy(
                   DropDy,
                   1,
                   1,
                   Autodiff(op, 1, True)(RotDy(-1, scan), right)
               )
           )

    # The derivative of x_i is the derivative of the ith intermediate result
    # times the derivative of that w.r.t. x_i.
    d[right] = JotDiaDyDy(
                   TimesDy,
                   1,
                   RhoDy(
                       CatDy(Rho(d[y]), 1),
                       d[y]
                   ),
                   TimesDy(chain, cons)
               )


@adjoint(SlashbarMonMon)
def dSlashbarMonMon(y, op, right):
    # This adjoint is identical to that of SlashMonMon, except that the data
    # is transposed to account for the change of axis.
    trans_right = Trans(right)
    scan = BackslashMonMon(op, trans_right)
    chain = CatDy(
                Rot(
                    BackslashMonMon(
                        TimesDy,
                        JotDiaDyDy(
                            DropDy,
                            1,
                            1,
                            Rot(
                                Autodiff(op, 0, True)(scan, RotDy(1, trans_right))
                            )
                        )
                    )
                ),
                1
            )
    cons = CatDy(
               1,
               JotDiaDyDy(
                   DropDy,
                   1,
                   1,
                   Autodiff(op, 1, True)(RotDy(-1, scan), trans_right)
               )
           )
    d[right] = Trans(
                   JotDiaDyDy(
                       TimesDy,
                       1,
                       RhoDy(
                           CatDy(Rho(Trans(d[y])), 1),
                           Trans(d[y])
                        ),
                        TimesDy(chain, cons)
                   )
               )


# Adjoins of non-differentiable functions, treated as constants.
def non_diff_mon(_, right):
    d[right] = TimesDy(0, right)


def non_diff_dy(_, left, right):
    d[left] = TimesDy(0, left)
    d[right] = TimesDy(0, right)


for prim in [Inline, Gradedown, Gradeup, Iota, Match, Max, Min, Nmatch, Rho, Tilde, Times]:
    adjoint(prim)(non_diff_mon)

for prim in [AndDy, EqDy, GtDy, GteqDy, LtDy, LteqDy, MatchDy, NandDy, NmatchDy, NorDy, OrDy]:
    adjoint(prim)(non_diff_dy)
"""


def change_dout_name(dpy: str, apl: str, repl: str = '⍺'):
    """
    Changes the name of the variable representing the output's derivative.

    Args:
        dpy: Derivative of the function as Python source code,
            used to extract the current name of the output's derivative.
        apl: APL code for which to change the name of the output's derivatives.
        repl: Replacement for the current name of the output's derivative.

    Returns:
        APL code, with the name of the output's derivative changed.
    """
    dout_name = re.findall(r'def \w+\((?:Alpha, )?Omega, (\w+)', dpy)[0]
    return re.sub(rf'(?<!\w){dout_name}(?!\w)', repl, apl)


class AutodiffTransformer(ast.NodeTransformer):
    """
    Inlines the derivatives of primitives used by Autodiff in adjoints (see
    also prims_str).
    """
    def __init__(self, prims: type) -> None:
        self.prims = prims

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'Autodiff':
            # Autodiff's third argument indicates the primitive's dyadic version
            # is to be used.
            if node.args[2].value:
                node.args[0].id += 'Dy'
            prim = getattr(self.prims, node.args[0].id)

            # Directly calling a primitive isn't supported by Tangent,
            # so an identity wrapper is used.
            def wrapper(Omega):
                return prim(Omega)

            def wrapperDy(Alpha, Omega):
                return prim(Alpha, Omega)

            dprim = tangent.grad(wrapperDy if node.args[0].id.endswith('Dy') else wrapper,
                                 wrt=(node.args[1].value,),
                                 check_dims=False)
            dprim = getsource(dprim).replace('prim', node.args[0].id)
            dfn = change_dout_name(dprim, py_to_apl(dprim), 'out_g')

            body = re.search(r'\{(.*?)\}', dfn, re.DOTALL).group(1)
            anon = ('{out_g←1+0×⍵ ⋄ ' +
                    ' ⋄ '.join([line.strip() for line in body.splitlines()[1:]]) +
                    '}')

            new_call = ast.Call(func=ast.Name(id='Inline', ctx=ast.Load()),
                                args=[ast.Constant(anon)],
                                keywords=[])
            return ast.fix_missing_locations(new_call)

        return self.generic_visit(node)


def autodiff(
    apl: str,
    aplparse: str,
    print_py: bool = False,
    print_dpy: bool = False,
    ) -> str:
    """
    Generates the derivative of an APL dfn.

    Args:
        apl: APL dfn whose derivative is generated.
        aplparse: Path to aplparse executable.
        print_py: Flag to print the transpiled Python code.
        print_dpy: Flag to print the derivative of the transpiled Python code.

    Returns:
        Derivative of the passed dfn as APL source code.

    Raises:
        Exception: The derivative couldn't be generated.
    """
    py = apl_to_py(apl, aplparse)

    if print_py:
        print('Transpiled Python code:\n', py)

    # The following check for known limitations of the adjoints.
    # Traversing the AST to match these cases is more elegant and robust,
    # but a regex strategy works fine, too.
    if len(re.findall(r'CircDy\((?=(1|2))', py)) != py.count('CircDy('):
        raise RuntimeError('Sin and cos are the only supported trig functions.')

    if len(re.findall(r'DotDyDy\((?=Add, Times)', py)) != py.count('DotDyDy('):
        raise RuntimeError('The only supported inner product is matrix multiplication.')

    if re.search(r'JotDia\w+\([a-zA-Z]+, [a-zA-Z]+', py):
        raise RuntimeError('Jot Diaeresis support only extends to rank with scalar right operands.')

    if re.search(r'JotDia\w+\((?=(Cat|Disclose|Drop|Enclose|In|Iota|Row|Squad|Trans|Vcat))', py):
        raise RuntimeError('Rank does not support structural or selection functions.')

    if len(re.findall(r'SlashMonMon\((?=(Add|And|Max|Min|Or|Times))', py)) != py.count('SlashMonMon('):
        raise RuntimeError('Reduce only works with associative functions.')

    if len(re.findall(r'SlashbarMonMon\((?=(Add|And|Max|Min|Or|Times))', py)) != py.count('SlashbarMonMon('):
        raise RuntimeError('Reduce first only works with associative functions.')

    # Tangent expects a function, and since the transpiled Python code
    # is generated at runtime, it's saved to a temporary file and then loaded.
    # Declaring it via exec wouldn't work because the source would be inaccessible.
    with open(f'prims.py', 'w+') as f:
        f.write(prims_str + '\n' + py)

    try:
        import prims
        imp.reload(prims)

        name = re.search(r'def\s+(\w+)\s*\(', py).group(1) # Gets the function's name.
        dpy = getsource(tangent.grad(getattr(prims, name), check_dims=False))
        dpy = astor.to_source(AutodiffTransformer(prims).visit(ast.parse(dpy)))

        # These are Tangent utilities for initializing, summing, and copying derivatives,
        # which are reformulated using their APL equivalents.
        dpy = dpy.replace('tangent.init_grad(', 'TimesDy(0, ')
        dpy = dpy.replace('tangent.add_grad', 'AddDy')
        dpy = dpy.replace('tangent.copy', '') # APL deep-copies on each assignment.

        # A stack is maintained by Tangent to handle calls to other function, which isn't
        # supported by the transpiler. Lines relating to it can be safely erased
        # since the stack's items are primitives only.
        filtered_lines = filter(lambda x: not any(sub in x for sub in ['tangent.Stack',
                                                                       'tangent.push',
                                                                       'tangent.pop',
                                                                       '= None']),
                                dpy.splitlines())
        dpy = '\n'.join(list(filtered_lines))

    except Exception as e:
        print(f'Failed to generate derivative: {e}')
        sys.exit(1)

    finally:
        Path('prims.py').unlink()

    if print_dpy:
        print('Python derivative:\n', dpy)

    apl = py_to_apl(dpy)
    return change_dout_name(dpy, apl)
