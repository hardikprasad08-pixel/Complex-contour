import sympy as sp
from typing import Dict, Any, List
from src.contours import ComplexContour, Circle, Line, Polygonal
from src.singularities import find_singularities

def analyze_integration_strategy(expr: sp.Expr, contour: ComplexContour) -> Dict[str, Any]:
    """
    Rule-based recommender engine that analyzes a SymPy expression and a complex contour
    to suggest the optimal integration method (Residue Theorem vs Numerical Quadrature vs Special Methods).
    Outputs audit verification grades (A, B, C, F).
    """
    singularities = find_singularities(expr)
    
    # Check if contour is closed (for simplified recommendation rules)
    is_closed = False
    if isinstance(contour, Circle):
        is_closed = True
    elif isinstance(contour, Polygonal):
        # If start and end vertices are very close, consider it closed
        if abs(contour.vertices[0] - contour.vertices[-1]) < 1e-9:
            is_closed = True

    # Check if there are branch points
    has_branch_points = any(s['type'] == 'branch_point' for s in singularities)
    has_essential_singularities = any(s['type'] == 'essential' for s in singularities)
    poles = [s for s in singularities if s['type'] == 'pole']

    # Determine if any singularity lies directly on the contour path
    # We sample the contour and check proximity to singularities
    t_vals, z_vals = contour.sample(200)
    on_contour_threshold = 1e-2
    poles_on_contour = []
    for pole in poles:
        p_val = complex(pole['point'].evalf())
        min_dist = min(abs(z - p_val) for z in z_vals)
        if min_dist < on_contour_threshold:
            poles_on_contour.append(pole)

    # Recommendations and Grading Logic
    if poles_on_contour:
        recommendation = "Numerical Integration (Adaptive with Principal Value handling) or Deformation of path."
        explanation = "Poles detected extremely close or directly on the contour path. Standard Residue Theorem cannot be applied without principal value deformation."
        audit_grade = "C" 
    elif has_branch_points:
        recommendation = "Specialized Branch-Cut Numerical Quadrature or Keyhole Contour Deformation."
        explanation = "Expression contains multi-valued branch points requiring careful branch-cut handling; standard Residue Theorem requires branch-cut avoidance path integration."
        audit_grade = "B"
    elif has_essential_singularities:
        recommendation = "Numerical Quadrature (Adaptive)."
        explanation = "Essential singularities present. Computing residues symbolically is highly unstable or non-algebraic."
        audit_grade = "B"
    elif is_closed and len(poles) > 0:
        recommendation = "Cauchy's Residue Theorem (Symbolic)."
        explanation = "Contour is closed and contains poles inside with no branch cuts. This is the most computationally exact approach."
        audit_grade = "A"
    elif not is_closed:
        recommendation = "Numerical Quadrature (Adaptive)."
        explanation = "Contour is open. Residue Theorem is not applicable; direct numerical parameterization is highly recommended."
        audit_grade = "A"
    else:
        recommendation = "Cauchy-Goursat Theorem (Symbolic is Zero)."
        explanation = "Contour is closed, and there are no singularities inside or along the path. Integral is analytically zero."
        audit_grade = "A"

    return {
        'recommended_method': recommendation,
        'explanation': explanation,
        'audit_grade': audit_grade,
        'singularities_found': len(singularities),
        'poles_on_contour_detected': len(poles_on_contour) > 0
    }
