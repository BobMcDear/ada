⍝ Get is meant to be used during development only.
]Get 'file://prims.aplf'
]Get 'file://dprims.aplf'
]Get 'file://dprims_ref.aplf'
⎕RL←1

bin←{0.5<⍵}
zero←{0×⍵}
double_op←{(⍺ ⍺⍺ ⍵) (⍺ ⍵⍵ ⍵)}
test←{
    dprim dprim_ref←(?0×⍺⍺ ⍵) ⍵⍵ ⍵
    dprim≡dprim_ref
}

vec←?8⍴0
mat←?4 8⍴0
ten1←?3 4 8⍴0
ten2←?3 4 8⍴0

⍝ Monadic differentiable functions.
⎕←'cat' ((cat test (dcatdOmega double_op dcatdOmega_ref)) ten1)
⎕←'circ' ((circ test (dcircdOmega double_op dcircdOmega_ref)) ten1)
⎕←'circstar' ((circstar test (dcircstardOmega double_op dcircstardOmega_ref)) ten1)
⎕←'disclose' ((disclose test (ddisclosedOmega double_op ddisclosedOmega_ref)) ten1)
⎕←'div' ((div test (ddivdOmega double_op ddivdOmega_ref)) ten1)
⎕←'enclose' ((enclose test (denclosedOmega double_op denclosedOmega_ref)) ten1)
⎕←'in_' ((in_ test (din_dOmega double_op din_dOmega_ref)) ten1)
⎕←'pipe' ((pipe test (dpipedOmega double_op dpipedOmega_ref)) ten1)
⎕←'pow' ((pow test (dpowdOmega double_op dpowdOmega_ref)) ten1)
⎕←'rot' ((rot test (drotdOmega double_op drotdOmega_ref)) ten1)
⎕←'sub' ((sub test (dsubdOmega double_op dsubdOmega_ref)) ten1)
⎕←'trans' ((trans test (dtransdOmega double_op dtransdOmega_ref)) ten1)
⎕←'vcat' ((vcat test (dvcatdOmega double_op dvcatdOmega_ref)) ten1)
⎕←'vrot' ((vrot test (dvrotdOmega double_op dvrotdOmega_ref)) ten1)

⍝ Dyadic differentiable functions.
⎕←'add_dy' ((add_dy test (dadd_dydOmega double_op dadd_dydOmega_ref)) ten1 ten2)
⎕←'cat_dy' ((cat_dy test (dcat_dydOmega double_op dcat_dydOmega_ref)) ten1 ten2)
⎕←'circ_dy' ((circ_dy test (dcirc_dydOmega double_op dcirc_dydOmega_ref)) ten1)
⎕←'circstar_dy' ((circstar_dy test (dcircstar_dydOmega double_op dcircstar_dydOmega_ref)) ten1 ten2)
⎕←'disclose_dy' ((disclose_dy test (ddisclose_dydOmega double_op ddisclose_dydOmega_ref)) 1 vec)
⎕←'div_dy' ((div_dy test (ddiv_dydOmega double_op ddiv_dydOmega_ref)) ten1 ten2)
⎕←'drop_dy' ((drop_dy test (ddrop_dydOmega double_op ddrop_dydOmega_ref)) 2 ten1)
⎕←'min_dy' ((min_dy test (dmin_dydOmega double_op dmin_dydOmega_ref)) ten1 ten2)
⎕←'pow_dy' ((pow_dy test (dpow_dydOmega double_op dpow_dydOmega_ref)) ten1 ten2)
⎕←'rho_dy' ((rho_dy test (drho_dydOmega double_op drho_dydOmega_ref)) (6 4 4) ten1)
⎕←'rot_dy' ((rot_dy test (drot_dydOmega double_op drot_dydOmega_ref)) 2 ten1)
⎕←'squad_dy' ((squad_dy test (dsquad_dydOmega double_op dsquad_dydOmega_ref)) (⊂1 2) ten1)
⎕←'sub_dy' ((sub_dy test (dsub_dydOmega double_op dsub_dydOmega_ref)) ten1 ten2)
⎕←'take_dy' ((take_dy test (dtake_dydOmega double_op dtake_dydOmega_ref)) 2 ten1)
⎕←'times_dy' ((times_dy test (dtimes_dydOmega double_op dtimes_dydOmega_ref)) ten1 ten2)
⎕←'trans_dy' ((trans_dy test (dtrans_dydOmega double_op dtrans_dydOmega_ref)) (2 3 1) ten1)
⎕←'vcat_dy' ((vcat_dy test (dvcat_dydOmega double_op dvcat_dydOmega_ref)) ten1 ten2)
⎕←'vrot_dy' ((vrot_dy test (dvrot_dydOmega double_op dvrot_dydOmega_ref)) 2 ten1)

⍝ Operators.
⎕←'dot_dy_dy' ((dot_dy_dy test (ddot_dy_dydOmega double_op ddot_dy_dydOmega_ref)) ten1 (⍉ten2))
⎕←'jot_dy_mon1' ((jot_dy_mon1 test (djot_dy_mon1dOmega double_op djot_dy_mon1dOmega_ref)) ten1)
⎕←'jot_dy_mon2' ((jot_dy_mon2 test (djot_dy_mon2dOmega double_op djot_dy_mon2dOmega_ref)) ten1)
⎕←'jot_dy_dy1' ((jot_dy_dy1 test (djot_dy_dy1dOmega double_op djot_dy_dy1dOmega_ref)) ten1 vec)
⎕←'jot_dy_dy2' ((jot_dy_dy2 test (djot_dy_dy2dOmega double_op djot_dy_dy2dOmega_ref)) ten1 mat)
⎕←'slash_mon_mon1' ((slash_mon_mon1 test (dslash_mon_mon1dOmega double_op dslash_mon_mon1dOmega_ref)) ten1)
⎕←'slash_mon_mon2' ((slash_mon_mon2 test (dslash_mon_mon2dOmega double_op dslash_mon_mon2dOmega_ref)) ten1)
⎕←'slashbar_mon_mon1' ((slashbar_mon_mon1 test (dslashbar_mon_mon1dOmega double_op dslashbar_mon_mon1dOmega_ref)) ten1)
⎕←'slashbar_mon_mon2' ((slashbar_mon_mon2 test (dslashbar_mon_mon2dOmega double_op dslashbar_mon_mon2dOmega_ref)) ten1)

⍝ Non-differentiable functions, treated as constants.
⎕←'gradedown' ((gradedown test (dgradedowndOmega double_op zero)) vec)
⎕←'gradeup' ((gradeup test (dgradeupdOmega double_op zero)) vec)
⎕←'iota' ((iota test (diotadOmega double_op zero)) 16)
⎕←'match' ((match test (dmatchdOmega double_op zero)) ten1)
⎕←'max' ((max test (dmaxdOmega double_op zero)) ten1)
⎕←'min' ((min test (dmindOmega double_op zero)) ten1)
⎕←'nmatch' ((nmatch test (dnmatchdOmega double_op zero)) ten1)
⎕←'rho' ((rho test (drhodOmega double_op zero)) ten1)
⎕←'tilde' ((tilde test (dtildedOmega double_op zero)) bin ten1)
⎕←'times' ((times test (dtimesdOmega double_op zero)) ten1)
⎕←'and_dy' ((and_dy test (dand_dydOmega double_op zero)) (bin ten1) (bin ten2))
⎕←'eq_dy' ((eq_dy test (deq_dydOmega double_op zero)) ten1 ten2)
⎕←'gt_dy' ((gt_dy test (dgt_dydOmega double_op zero)) ten1 ten2)
⎕←'gteq_dy' ((gteq_dy test (dgteq_dydOmega double_op zero)) ten1 ten2)
⎕←'lt_dy' ((lt_dy test (dlt_dydOmega double_op zero)) ten1 ten2)
⎕←'lteq_dy' ((lteq_dy test (dlteq_dydOmega double_op zero)) ten1 ten2)
⎕←'match_dy' ((match_dy test (dmatch_dydOmega double_op zero)) ten1 ten2)
⎕←'max_dy' ((max_dy test (dmax_dydOmega double_op dmax_dydOmega_ref)) ten1 ten2)
⎕←'nand_dy' ((nand_dy test (dnand_dydOmega double_op zero)) (bin ten1) (bin ten2))
⎕←'nmatch_dy' ((nmatch_dy test (dnmatch_dydOmega double_op zero)) ten1 ten2)
⎕←'nor_dy' ((nor_dy test (dnor_dydOmega double_op zero)) (bin ten1) (bin ten2))
⎕←'or_dy' ((or_dy test (dor_dydOmega double_op zero)) (bin ten1) (bin ten2))
