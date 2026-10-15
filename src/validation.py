import numpy as np
from typing import Callable, Dict, Any, List
from src.contours import ComplexContour
from src.numerical import integrate_adaptive, integrate_trapezoidal

def validate_numerical_vs_analytical(f: Callable[[complex], complex], 
                                     contour: ComplexContour, 
                                     analytical_val: complex) -> Dict[str, Any]:
    """
    Validates numerical integration results against analytical solutions,
    calculating errors and checking convergence behaviors.
    """
    # Adaptive integration
    try:
        num_val_adaptive = integrate_adaptive(f, contour, tolerance=1e-9)
    except Exception as e:
        num_val_adaptive = float('nan')
        
    # Multigrid / Multi-sampling Trapezoidal test to analyze convergence
    sampling_sizes = [100, 500, 2000, 5000]
    trapezoidal_results = {}
    
    for size in sampling_sizes:
        try:
            val = integrate_trapezoidal(f, contour, num_points=size)
            abs_err = abs(val - analytical_val)
            rel_err = abs_err / abs(analytical_val) if abs(analytical_val) != 0 else abs_err
            trapezoidal_results[size] = {
                'value': val,
                'abs_error': abs_err,
                'rel_error': rel_err
            }
        except Exception as e:
            trapezoidal_results[size] = {'error': str(e)}

    # Calculate errors for adaptive
    if not np.isnan(num_val_adaptive):
        abs_err_adapt = abs(num_val_adaptive - analytical_val)
        rel_err_adapt = abs_err_adapt / abs(analytical_val) if abs(analytical_val) != 0 else abs_err_adapt
    else:
        abs_err_adapt = float('nan')
        rel_err_adapt = float('nan')

    return {
        'analytical_value': analytical_val,
        'adaptive_value': num_val_adaptive,
        'adaptive_abs_error': abs_err_adapt,
        'adaptive_rel_error': rel_err_adapt,
        'convergence_history': trapezoidal_results
    }
