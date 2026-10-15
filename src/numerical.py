import numpy as np
from typing import Callable
from src.contours import ComplexContour

def integrate_trapezoidal(f: Callable[[complex], complex], contour: ComplexContour, num_points: int = 1000) -> complex:
    """
    Integrates f(z) along the given contour using the composite trapezoidal rule.
    """
    t_vals = np.linspace(0.0, 1.0, num_points)
    dt = 1.0 / (num_points - 1)
    
    integrand_vals = []
    for t in t_vals:
        z = contour.evaluate(t)
        dz = contour.derivative(t)
        integrand_vals.append(f(z) * dz)
        
    integrand_vals = np.array(integrand_vals, dtype=complex)
    
    # Composite trapezoidal rule
    integral = (np.sum(integrand_vals) - 0.5 * (integrand_vals[0] + integrand_vals[-1])) * dt
    return integral

def integrate_adaptive(f: Callable[[complex], complex], contour: ComplexContour, tolerance: float = 1e-8) -> complex:
    """
    Adaptive numerical complex integration using an adaptive interval subdivision scheme.
    """
    def adaptive_step(t_start: float, t_end: float, fa: complex, fb: complex, fmid: complex) -> complex:
        t_mid = 0.5 * (t_start + t_end)
        
        # Quarter points
        t_q1 = 0.5 * (t_start + t_mid)
        t_q2 = 0.5 * (t_mid + t_end)
        
        # Evaluate z, dz and f at all points
        def eval_integrand(t):
            return f(contour.evaluate(t)) * contour.derivative(t)
            
        fq1 = eval_integrand(t_q1)
        fq2 = eval_integrand(t_q2)
        
        h = t_end - t_start
        # Simpson's rule approximations
        S1 = (h / 6.0) * (fa + 4.0 * fmid + fb)
        S2 = (h / 12.0) * (fa + 4.0 * fq1 + 2.0 * fmid + 4.0 * fq2 + fb)
        
        if abs(S2 - S1) < tolerance:
            return S2 + (S2 - S1) / 15.0  # Richardson extrapolation
            
        return (adaptive_step(t_start, t_mid, fa, fmid, fq1) +
                adaptive_step(t_mid, t_end, fmid, fb, fq2))

    # Initial evaluations
    def eval_integrand(t):
        return f(contour.evaluate(t)) * contour.derivative(t)
        
    fa = eval_integrand(0.0)
    fb = eval_integrand(1.0)
    fmid = eval_integrand(0.5)
    
    return adaptive_step(0.0, 1.0, fa, fb, fmid)
