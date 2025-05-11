_To avoid conflict with the programming language [Ada](https://en.wikipedia.org/wiki/Ada_(programming_language)), this project has been renamed to APLAD from ada. The command-line tool, Python package, and GitHub repository retain the name ada._

_For kindred APL projects, see_ [trap](https://github.com/BobMcDear/trap) _and_ [APLearn](https://github.com/BobMcDear/aplearn).

# APLAD

• **[Introduction](#introduction)**<br>
• **[How It Works](#how-it-works)**<br>
• **[Usage](#usage)**<br>
• **[Example](#example)**<br>
• **[Supported Primitives](#supported-primitives)**<br>
• **[Source Code Transformation vs. Operator Overloading](#source-code-transformation-vs-operator-overloading)**<br>
• **[Tests](#tests)**<br>

## Introduction

APLAD (formerly called ada) is a reverse-mode autodiff (AD) framework based on source code transformation (SCT) for [Dyalog APL](https://aplwiki.com/wiki/Dyalog_APL). It accepts APL functions and outputs corresponding functions, written in plain APL, that evaluate the originals' derivatives. This extends to inputs of arbitrary dimension, so the partial derivatives of multivariate functions can be computed as easily as the derivatives of scalar ones. Seen through a different lens, APLAD is a source-to-source compiler that produces an APL program's derivative in the same language.

APL, given its array-oriented nature, is particularly suitable for scientific computing and linear algebra. However, AD has become a crucial ingredient of these domains by providing a solution to otherwise intractable problems, and APL, notwithstanding its intimate relationship with mathematics since its inception, substantially lags behind languages like Python, Swift, and Julia in this area. In addition to being error-prone and labour-intensive, implementing derivatives by hand effectively doubles the volume of code, thus defeating one of the main purposes of array programming, namely, brevity. APLAD aims to alleviate this issue by offering a means of automatically generating the derivative of APL code.

## How It Works

APLAD, which is implemented in Python, comprises three stages: First, it leverages an external [Standard ML](https://en.wikipedia.org/wiki/Standard_ML) library, [aplparse](https://github.com/bobmcdear/sml-aplparse) (not affiliated with APLAD), to parse APL code, and then transpiles the syntax tree into a symbolic Python program composed of APL primitives. The core of APLAD lies in the second step, which evaluates the derivative of the transpiled code using [Tangent](https://github.com/google/tangent), a source-to-source AD package for Python. Since the semantics of APL primitives are foreign to Python, the adjoint of each is manually defined, constituting the heart of the codebase. Following this second phase, the third and final part transpiles the derivative produced in the previous step back into APL.

This collage-like design might initially seem a bit odd: An AD tool for APL that's written in Python and utilizes a parser implemented in Standard ML. The reason behind it is to minimize the complexity of APLAD by reusing well-established software instead of reinventing the wheel. Parsing APL, though simpler than parsing, say, C, is still non-trivial and would demand its own bulky module. SCT is even more technically sophisticated given that it's tantamount to writing a compiler for the language. aplparse and Tangent take care of parsing and SCT, respectively, leaving ada with two tasks: I) APL-to-Python & Python-to-APL transpilation and II) Defining derivative rules for APL primitives. This layered approach is somewhat hacky and more convoluted than an hypothetical differential operator built into APL, but it's more practical to develop and maintain as an _initial proof of concept_.

## Usage

[aplparse](https://github.com/bobmcdear/sml-aplparse) isn't shipped with APLAD and must be downloaded separately. Having done so, it needs to be compiled into an executable using [MLton](http://mlton.org/). More information can be found in the aplparse repository.

To install APLAD itself, please run ```pip install git+https://github.com/bobmcdear/ada.git```. APLAD is exposed as a command-line tool, ```ada```, requiring the path to an APL file that'll be differentiated and the parser's executable. The APL file must contain exclusively monadic dfns, and APLAD outputs their derivatives in a new file. Restrictions apply to the types of functions that are consumable by APLAD: They need to be pure, can't call other functions (including anonymous ones), and must only incorporate the primitives listed in [the Supported Primitives section](#supported-primitives). These limitations, besides purity, will be gradually eliminated, but violating them for now will lead to errors or undefined behaviour.

## Example

[trap](https://github.com/BobMcDear/trap), an APL implementation of the transformer architecture, is a case study of array programming's applicability to deep learning, a field currently dominated by Python and its immense ecosystem. Half its code is dedicated to manually handling gradients for backpropagation, and one of APLAD's concrete goals is to facilitate the implementation of neural networks in APL by providing AD capabilities. As a minimal example, below is a regression network with two linear layers and the ReLU activation function sandwiched between them:

```apl
net←{
    x←1⊃⍵ ⋄ y←2⊃⍵ ⋄ w1←3⊃⍵ ⋄ b1←4⊃⍵ ⋄ w2←5⊃⍵ ⋄ b2←6⊃⍵
    z←0⌈b1(+⍤1)x+.×w1
    out←b2+z+.×w2
    (+/(out-y)*2)÷≢y
}
```

Saving this to ```net.aplf``` and running ```ada net.aplf aplparse```, where ```aplparse``` is the parser's executable, will create a file, ```dnet.aplf```, containing the following:

```apl
dnetdOmega←{
    x←1⊃⍵
    y←2⊃⍵
    w1←3⊃⍵
    b1←4⊃⍵
    w2←5⊃⍵
    b2←6⊃⍵
    DotDyDy_var_name←x(+.×)w1
    JotDiaDyDy_var_name←b1(+⍤1)DotDyDy_var_name
    z←0⌈JotDiaDyDy_var_name
    DotDyDy2←z(+.×)w2
    out←b2+DotDyDy2
    Nmatch_y←≢y
    SubDy_out_y←out-y
    _return3←SubDy_out_y*2
    _b_return2←⍺÷Nmatch_y
    b_return2←_b_return2
    scan←+\_return3
    chain←(⌽×\1(↓⍤1)⌽scan{out_g←1+0×⍵ ⋄ bAlpha←out_g ⋄ bAlpha}1⌽_return3),1
    cons←1,1(↓⍤1)(¯1⌽scan){out_g←1+0×⍵ ⋄ bOmega←out_g ⋄ bOmega}_return3
    _b_return3←(((⍴b_return2),1)⍴b_return2)(×⍤1)chain×cons
    b_return3←_b_return3
    _bSubDy_out_y←b_return3×2×SubDy_out_y*2-1
    bSubDy_out_y←_bSubDy_out_y
    _by2←-bSubDy_out_y
    bout←bSubDy_out_y
    by←_by2
    _by←0×y
    by←by+_by
    bb2←bout
    bDotDyDy2←bout
    dim_left←×/¯1↓⍴z
    dim_right←×/1↓⍴w2
    mat_left←(dim_left,¯1↑⍴z)⍴z
    mat_right←((1↑⍴w2),dim_right)⍴w2
    mat_dy←(dim_left,dim_right)⍴bDotDyDy2
    _bz←(⍴z)⍴mat_dy(+.×)⍉mat_right
    _bw2←(⍴w2)⍴(⍉mat_left)(+.×)mat_dy
    bz←_bz
    bw2←_bw2
    _bJotDiaDyDy←bz×JotDiaDyDy_var_name≥0
    bJotDiaDyDy←_bJotDiaDyDy
    full_dleft←bJotDiaDyDy(×⍤1)b1({out_g←1+0×⍵ ⋄ bAlpha←out_g ⋄ bAlpha}⍤1)DotDyDy_var_name
    full_dright←bJotDiaDyDy(×⍤1)b1({out_g←1+0×⍵ ⋄ bOmega←out_g ⋄ bOmega}⍤1)DotDyDy_var_name
    red_rank_dleft←(≢⍴full_dleft)-≢⍴b1
    red_rank_dright←(≢⍴full_dright)-≢⍴DotDyDy_var_name
    _bb1←⍉({+/,⍵}⍤red_rank_dleft)⍉full_dleft
    _bDotDyDy←⍉({+/,⍵}⍤red_rank_dright)⍉full_dright
    bb1←_bb1
    bDotDyDy←_bDotDyDy
    dim_left←×/¯1↓⍴x
    dim_right←×/1↓⍴w1
    mat_left←(dim_left,¯1↑⍴x)⍴x
    mat_right←((1↑⍴w1),dim_right)⍴w1
    mat_dy←(dim_left,dim_right)⍴bDotDyDy
    _bx←(⍴x)⍴mat_dy(+.×)⍉mat_right
    _bw1←(⍴w1)⍴(⍉mat_left)(+.×)mat_dy
    bx←_bx
    bw1←_bw1
    zeros←0×⍵
    (6⊃zeros)←bb2 ⋄ _bOmega6←zeros
    bOmega←_bOmega6
    zeros←0×⍵
    (5⊃zeros)←bw2 ⋄ _bOmega5←zeros
    bOmega←bOmega+_bOmega5
    zeros←0×⍵
    (4⊃zeros)←bb1 ⋄ _bOmega4←zeros
    bOmega←bOmega+_bOmega4
    zeros←0×⍵
    (3⊃zeros)←bw1 ⋄ _bOmega3←zeros
    bOmega←bOmega+_bOmega3
    zeros←0×⍵
    (2⊃zeros)←by ⋄ _bOmega2←zeros
    bOmega←bOmega+_bOmega2
    zeros←0×⍵
    (1⊃zeros)←bx ⋄ _bOmega←zeros
    bOmega←bOmega+_bOmega
    bOmega
}
```

```dnetdOmega``` is a dyadic function whose right and left arguments represent the function's input and the derivative of the output, respectively. It returns the gradients of every input array, but those of the independent & dependent variables should be discarded since the dataset isn't being tuned. The snippet below trains the model on synthetic data for 30000 iterations and prints the final loss, which should converge to <0.001.

```apl
x←?128 8⍴0 ⋄ y←1○+/x
w1←8 8⍴1 ⋄ b1←8⍴0
w2←8⍴1 ⋄ b2←0
lr←0.01

iter←{
    x y w1 b1 w2 b2←⍵
    _ _ dw1 db1 dw2 db2←1 dnetdOmega x y w1 b1 w2 b2
    x y (w1-lr×dw1) (b1-lr×db1) (w2-lr×dw2) (b2-lr×db2)
}

_ _ w1 b1 w2 b2←iter⍣10000⊢x y w1 b1 w2 b2
⎕←net x y w1 b1 w2 b2
```


## Supported Primitives

Below is a table of APL functions that are supported by APLAD. They include most functions that'd normally be used in an AD context, even non-differentiable ones. The latter are effectively treated as constants, e.g., dividing a vector's elements by its length (non-differentiable) is equivalent to dividing it by a constant, that constant being its length and determined at runtime.

| Function | Monadic | Dyadic | Notes |
|-----------|---------|--------|-------|
|     ```+```       |  ✕       |   ✓     |     None.  |
|     ```-```      |   ✓      |      ✓  | None.      |
|    ```×```       |    ✓     |     ✓   |     None.  |
|     ```÷```      |    ✓     |     ✓   |   None.    |
|      ```⌈```     |    ✓     |     ✓   |   None.    |
|     ```⌊```      |    ✓     |     ✓   |  None.     |
|    ```*```       |   ✓      |    ✓    |  None.     |
|      ```\|```     |   ✓      |    ✕    |  None.     |
|      ```⍟```     |   ✓      |    ✓    |   None.    |
|      ```○```     |    ✓     |   ✓     |   Only sin and cos (i.e., left arguments of 1 or 2) are supported.    |
|     ```~```      |    ✓     |    ✕    |    None.   |
|      ```∧```     |   N/A      |     ✓   |  Both arguments must be binary.     |
|     ```∨```      |      N/A   |     ✓   |   Both arguments must be binary.    |
|      ```⍲```     |   N/A      |     ✓   |   None.    |
|      ```⍱```     |     N/A    |     ✓   |   None.    |
|     ```<```      |    N/A     |    ✓    |   None.    |
|     ```>```      |    N/A     |   ✓     |   None.    |
|       ```≤```    |   N/A      |    ✓    |   None.    |
|       ```≥```    |   N/A      |   ✓     |   None.    |
|    ```=```       |    N/A     |      ✓  |   None.    |
|    ```≡```       |     ✓    |     ✓   |     None.  |
|    ```≢```       |     ✓    |    ✓    |   None.    |
|    ```⍴```       |        ✓ |    ✓    |   None.    |
|    ```,```       |      ✓   |   ✓     |   None.    |
|    ```⍪```       |      ✓   |   ✓     |   None.    |
|    ```⌽```       |    ✓     |    ✓    |   None.    |
|    ```⊖```       |      ✓   |    ✓    |    None.   |
|    ```⍉```       |     ✓    |   ✓     |    None.   |
|    ```↑```       |    ✕     |    ✓    |   None.    |
|    ```↓```       |    ✕     |    ✓    |   None.    |
|    ```⊂```       |     ✓    |    ✕    |  None.     |
|    ```∊```       |     ✓    |    ✕    |  None.     |
|    ```⌷```       |   ✕      |   ✓     |   None.    |
|    ```⊃```       |    ✓     |   ✓     |   None.    |
|    ```⍳```       |     ✓    |    ✕    |   None.    |
|    ```⍋```       |  ✓       |   ✕     |  None.     |
|    ```⍒```       |     ✓    |   ✕     |  None.     |

Operators are trickier to differentiate, particularly when using SCT: Not only do the functions passed to them need to be differentiated, but so do the transformations performed on these functions. Because the derivative of the operand function is itself obtained via AD, differentiating operators invokes a two-level nested SCT. In other words, APLAD first performs SCT to generate the derivative of the operator, with the operand function being represented symbolically, and another SCT pass occurs _within_ the first one to differentiate the operand function. That's not to say operators are fundamentally impossible to differentiate using SCT, but it's not an easy job, either. For the time being, APLAD only supports the most essential use cases of a few chief operators, outlined below; more features will be added in the future.

| Operator | Monadic | Dyadic | Notes |
|-----------|---------|--------|-------|
|     ```⍤```       |  N/A       |   ✓     |     Atop or mixed ranks aren't allowed. Moreover, the shape of the derived function's output must match that of at least one of the arguments.   |
|     ```.```      |   N/A      |      ✓  | Matrix multiplication is the only supported application of inner product (i.e., ```+.×```).      |
|    ```/```       |    ✓     |     N/A   |     The combining function must be associative.  |
|     ```⌿```      |    ✓     |     N/A   |   The combining function must be associative.    |

Several more general limitations exist in addition to those enumerated above:

* Complex numbers and strings are invalid.
* Guards, selective and modified assignments, and bracket indexing aren't supported.
* Strand notation only works with literals.

## Source Code Transformation vs. Operator Overloading

AD is commonly implemented via SCT or operator overloading (OO), though it's possible (indeed, beneficial) to employ a blend of both. The former offers several advantages over the latter, a few being:

* __Ease of use__: With SCT, no changes to the function that is to be differentiated are necessary, which translates to greater ease of use. By contrast, OO-powered AD usually entails wrapping values in _tracers_ to track the operations performed on them, and modifications to the code are necessary. Differentiating a cube function, for example, using OO would require replacing the input with a differentiable decimal type, whereas the function can be passed as-is when using SCT.
* __Portability__: SCT yields the derivative as a plain function written in the source language, enabling it to be evaluated without any dependencies in other environments.
* __Efficiency__: OO incurs runtime overhead and is generally not very amenable to optimizations. On the other hand, SCT tends to be faster since it generates the derivative ahead of time, allowing for more extensive optimizations. Efficiency gains become especially pronounced when compiling the code (e.g., [Co-dfns](https://github.com/Co-dfns/Co-dfns)).

The primary downside of SCT is its complexity: Creating a tracer type and extending the definition of a language's operations to render them differentiable is vastly more straightforward than parsing, analyzing, and rewriting source code to generate a function's derivative. Thanks to Tangent, however, APLAD sidesteps this difficulty by taking advantage of a mature SCT-backed AD infrastructure and simply extending its adjoint rules to APL primitives.

## Tests

To ensure the derivatives produced by APLAD are correct, all primitives have associated tests in ```tests/test.dyalog``` that check APLAD's results against the reference derivatives in ```tests/dprims_ref.aplf```. Before running them, ```ada tests/prims.aplf aplparse``` must be executed. Despite these tests, APLAD is most probably affected by unknown bugs, especially when dealing with irregular structures (e.g., nested or ragged), manipulating an array's shape in exotic ways, or extensively using operators; reporting them makes for a great contribution to this project.
