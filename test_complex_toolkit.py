import sympy as sp
import numpy as np
import pytest

from src.parser import safe_parse_expression
from src.contours import Circle, Line, Polygonal
from src.singularities import find_singularities
from src.residues import calculate_residue
from src.branches import analyze_branch_cuts
from src.numerical import integrate_trapezoidal, integrate_adaptive

# 1. Safe parsing of rational functions
def test_parser_rational():
    expr = safe_parse_expression("1/(z^2 + 1)")
    assert isinstance(expr, sp.Expr)

# 2. Safe parsing security rejection
def test_parser_security_reject():
    with pytest.raises(ValueError):
        safe_parse_expression("import os; os.system('echo')")

# 3. Simple pole identification
def test_simple_pole():
    z = sp.Symbol('z')
    expr = 1 / (z - 2j)
    sings = find_singularities(expr)
    assert any(s['type'] == 'pole' and s['order'] == 1 and sp.simplify(s['point'] - 2*sp.I) == 0 for s in sings)

# 4. Higher order pole identification
def test_higher_order_pole():
    z = sp.Symbol('z')
    expr = 1 / (z + 1)**3
    sings = find_singularities(expr)
    assert any(s['type'] == 'pole' and s['order'] == 3 and sp.simplify(s['point'] + 1) == 0 for s in sings)

# 5. Removable singularity detection
def test_removable_singularity():
    z = sp.Symbol('z')
    expr = sp.sin(z) / z
    sings = find_singularities(expr)
    assert any(s['type'] == 'removable' for s in sings)

# 6. Essential singularity detection
def test_essential_singularity():
    z = sp.Symbol('z')
    expr = sp.exp(1/z)
    sings = find_singularities(expr)
    assert any(s['type'] == 'essential' for s in sings)

# 7. Logarithmic branch cut identification
def test_log_branch_cuts():
    z = sp.Symbol('z')
    expr = sp.log(z)
    cuts = analyze_branch_cuts(expr)
    assert len(cuts) > 0
    assert cuts[0]['type'] == 'logarithmic'

# 8. Fractional power algebraic branch cut identification
def test_algebraic_branch_cuts():
    z = sp.Symbol('z')
    expr = sp.sqrt(z - 1)
    cuts = analyze_branch_cuts(expr)
    assert len(cuts) > 0
    assert cuts[0]['type'] == 'algebraic'

# 9. Circle contour sampling and derivatives
def test_circle_contour():
    circle = Circle(center=0, radius=2.5)
    t, z = circle.sample(100)
    assert len(z) == 100
    assert np.allclose(np.abs(z), 2.5)

# 10. Polygonal contour closed-loop property
def test_polygonal_closed():
    poly = Polygonal([0, 1, 1+1j, 1j, 0])
    assert np.allclose(poly.evaluate(0.0), 0)
    assert np.allclose(poly.evaluate(1.0), 0)

# 11. Trapezoidal integration accuracy
def test_numerical_trapezoidal():
    contour = Circle(center=0, radius=1.0)
    f = lambda z: 1.0 / z
    res = integrate_trapezoidal(f, contour, num_points=1000)
    assert np.isclose(res, 2j * np.pi, rtol=1e-4)

# 12. Adaptive integration accuracy with high-precision tolerance
def test_numerical_adaptive():
    contour = Circle(center=0, radius=1.0)
    f = lambda z: 1.0 / z
    res = integrate_adaptive(f, contour, tolerance=1e-8)
    assert np.isclose(res, 2j * np.pi, rtol=1e-6)
