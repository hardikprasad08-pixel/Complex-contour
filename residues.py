import sympy as sp

def calculate_residue(expr: sp.Expr, point: sp.Expr) -> sp.Expr:
    """
    Computes the symbolic residue of `expr` at a given `point`.
    Uses SymPy's built-in residue function, with a fallback to limit-based calculation.
    """
    z = sp.Symbol('z')
    try:
        res = sp.residue(expr, z, point)
        return sp.simplify(res)
    except Exception:
        # Fallback limit formula for a pole of order n (checking up to order 5)
        for n in range(1, 6):
            try:
                test_expr = (z - point)**n * expr
                diff_expr = sp.diff(test_expr, z, n - 1)
                fact = sp.factorial(n - 1)
                res_val = sp.limit(diff_expr / fact, z, point)
                if res_val.is_finite:
                    return sp.simplify(res_val)
            except Exception:
                pass
        raise ValueError(f"Could not symbolically calculate residue of {expr} at {point}")
