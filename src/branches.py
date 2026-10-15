import sympy as sp
from typing import List, Dict, Any

def analyze_branch_cuts(expr: sp.Expr) -> List[Dict[str, Any]]:
    """
    Scans a SymPy expression for branch-cut inducing functions (log, fractional powers).
    Returns a list of dictionaries detailing the branch points and cut descriptions.
    """
    z = sp.Symbol('z')
    cuts = []

    def scan(e):
        if isinstance(e, sp.log):
            arg = e.args[0]
            try:
                roots = sp.solve(arg, z)
                for r in roots:
                    cuts.append({
                        'type': 'logarithmic',
                        'branch_point': r,
                        'expression': e,
                        'direction': 'Negative real axis from branch point (Standard/Principal cut)'
                    })
            except Exception:
                pass
        elif isinstance(e, sp.Pow):
            base, exp_val = e.args
            if not exp_val.is_integer:
                try:
                    roots = sp.solve(base, z)
                    for r in roots:
                        cuts.append({
                            'type': 'algebraic',
                            'exponent': exp_val,
                            'branch_point': r,
                            'expression': e,
                            'direction': 'Negative real axis from branch point (Standard/Principal cut)'
                        })
                except Exception:
                    pass
        for arg in e.args:
            scan(arg)

    scan(expr)
    return cuts
