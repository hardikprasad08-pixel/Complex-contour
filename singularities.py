import sympy as sp
from typing import List, Dict, Any

def find_singularities(expr: sp.Expr) -> List[Dict[str, Any]]:
    """
    Analyzes a SymPy expression to locate and classify its singularities.
    Identifies poles, removable, essential, and branch points.
    Returns a list of dictionaries with keys: 'point', 'type', 'order'.
    """
    z = sp.Symbol('z')
    singularities = []

    # 1. Denominator zeros (candidate poles / removable singularities)
    num, den = sp.fraction(expr)
    candidates = []
    if den != 1:
        try:
            roots = sp.solve(den, z)
            for r in roots:
                candidates.append(r)
        except Exception:
            try:
                from sympy.calculus.util import singularities as sp_sing
                for r in sp_sing(expr, z):
                    candidates.append(r)
            except Exception:
                pass

    # Explicit scan for functions like exp(1/z) or sin(1/z) which produce essential singularities
    def scan_essential(e):
        pts = set()
        if isinstance(e, sp.exp) or isinstance(e, sp.sin) or isinstance(e, sp.cos):
            arg = e.args[0]
            num_arg, den_arg = sp.fraction(arg)
            if den_arg != 1:
                try:
                    roots = sp.solve(den_arg, z)
                    for r in roots:
                        pts.add(r)
                except Exception:
                    pass
        for arg in e.args:
            pts.update(scan_essential(arg))
        return pts

    for ep in scan_essential(expr):
        candidates.append(ep)

    # Deduplicate candidates
    candidates = list(set(candidates))

    for c in candidates:
        # Check if it's essential first if it comes from an exponential/trig non-polynomial arg
        # Let's verify via limits
        is_essential = False
        try:
            # If limits from different directions mismatch or are undefined/infinite, check limit of abs
            lim_val = sp.limit(expr, z, c)
            if not lim_val.is_finite and lim_val != sp.oo and lim_val != -sp.oo:
                is_essential = True
        except Exception:
            # Limit calculation failure often indicates highly oscillatory / essential singularity
            is_essential = True

        # Double check if limit of (z-c)^n * expr is never finite and non-zero for any n
        if is_essential:
            singularities.append({
                'point': c,
                'type': 'essential',
                'order': float('inf')
            })
            continue

        # Removable singularity check
        try:
            lim_val = sp.limit(expr, z, c)
            if lim_val.is_finite:
                singularities.append({
                    'point': c,
                    'type': 'removable',
                    'order': 0
                })
                continue
        except Exception:
            pass

        # Pole check
        is_pole = False
        for n in range(1, 10):
            try:
                test_expr = (z - c)**n * expr
                lim_test = sp.limit(test_expr, z, c)
                if lim_test.is_finite and lim_test != 0:
                    singularities.append({
                        'point': c,
                        'type': 'pole',
                        'order': n
                    })
                    is_pole = True
                    break
            except Exception:
                pass

        if not is_pole and not is_essential:
            singularities.append({
                'point': c,
                'type': 'essential',
                'order': float('inf')
            })

    # 2. Find branch points (log(z), z**a where a is fractional)
    def scan_branch_points(e):
        pts = set()
        if isinstance(e, sp.log):
            arg = e.args[0]
            try:
                roots = sp.solve(arg, z)
                for r in roots:
                    pts.add(r)
            except Exception:
                pass
        elif isinstance(e, sp.Pow):
            base, exp_val = e.args
            if not exp_val.is_integer:
                try:
                    roots = sp.solve(base, z)
                    for r in roots:
                        pts.add(r)
                except Exception:
                    pass
        for arg in e.args:
            pts.update(scan_branch_points(arg))
        return pts

    for bp in scan_branch_points(expr):
        if not any(sp.simplify(s['point'] - bp) == 0 for s in singularities):
            singularities.append({
                'point': bp,
                'type': 'branch_point',
                'order': None
            })

    return singularities
