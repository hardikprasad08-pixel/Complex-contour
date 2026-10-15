import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable
from src.contours import ComplexContour

def plot_complex_domain(contour: ComplexContour,
                        singularities: List[Dict[str, Any]] = None, 
                        branch_cuts: List[Dict[str, Any]] = None,
                        f: Callable[[complex], complex] = None,
                        grid_limit: float = 3.0) -> plt.Figure:
    """
    Generates a high-quality visualization of the complex plane, displaying
    the contour path, classified singularities, branch cuts, and optional vector field flow of f(z).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Plot the vector field flow if f is provided
    if f is not None:
        x = np.linspace(-grid_limit, grid_limit, 25)
        y = np.linspace(-grid_limit, grid_limit, 25)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # Vector field mapping: represent complex value w = f(z) as (u, v)
        # We use a try-except block to handle singularities during grid evaluation
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    # To avoid exact division by zero at poles, add a tiny perturbation
                    z_val = Z[i, j]
                    if abs(z_val) < 1e-9:
                        z_val += 1e-9 * (1 + 1j)
                    val = f(z_val)
                    if np.isnan(val) or np.isinf(val):
                        U[i, j], V[i, j] = 0, 0
                    else:
                        # Normalize to show directional flow clearly without extreme magnitudes
                        mag = np.hypot(val.real, val.imag)
                        if mag > 0:
                            U[i, j] = val.real / mag
                            V[i, j] = val.imag / mag
                except Exception:
                    U[i, j], V[i, j] = 0, 0
                    
        ax.quiver(X, Y, U, V, color='lightgray', alpha=0.5, angles='xy', scale_units='xy', scale=3, label='f(z) Flow Direction')

    # 2. Plot grid axes
    ax.axhline(0, color='black', linewidth=1.2, alpha=0.7)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Plot contour
    _, z_vals = contour.sample_adaptive(initial_points=100)
    ax.plot(z_vals.real, z_vals.imag, 'b-', linewidth=2.5, label='Contour Path (Directional)')
    
    # Draw directional arrows along the contour
    arrow_indices = [len(z_vals) // 4, len(z_vals) // 2, (3 * len(z_vals)) // 4]
    for idx in arrow_indices:
        if idx < len(z_vals) - 1:
            p1 = z_vals[idx]
            p2 = z_vals[idx+1]
            dp = p2 - p1
            if abs(dp) > 0:
                ax.annotate('', xy=(p2.real, p2.imag), xytext=(p1.real, p1.imag),
                            arrowprops=dict(arrowstyle="->", color='blue', lw=2.5, mutation_scale=15))

    # 4. Plot singularities
    if singularities:
        for sing in singularities:
            pt = complex(sing['point'].evalf())
            stype = sing['type']
            if stype == 'pole':
                ax.plot(pt.real, pt.imag, 'rx', markersize=10, markeredgewidth=2.5, label=f"Pole (Order {sing.get('order', 1)})")
            elif stype == 'essential':
                ax.plot(pt.real, pt.imag, 'ro', markersize=10, label='Essential Singularity')
            elif stype == 'branch_point':
                ax.plot(pt.real, pt.imag, 'g*', markersize=12, label='Branch Point')
            elif stype == 'removable':
                ax.plot(pt.real, pt.imag, 'ko', fillstyle='none', markersize=8, label='Removable Singularity')

    # 5. Plot branch cuts as stylized lines
    if branch_cuts:
        for cut in branch_cuts:
            bp = complex(cut['branch_point'].evalf())
            # Draw principal branch cut along negative real axis from branch point
            cut_x = np.linspace(bp.real - grid_limit * 2, bp.real, 100)
            cut_y = np.full_like(cut_x, bp.imag)
            ax.plot(cut_x, cut_y, 'g--', linewidth=2.0, alpha=0.8, label='Branch Cut (Principal)')

    ax.set_xlim(-grid_limit, grid_limit)
    ax.set_ylim(-grid_limit, grid_limit)
    ax.set_xlabel('Real Axis (Re)')
    ax.set_ylabel('Imaginary Axis (Im)')
    ax.set_title('Complex Plane Domain & Contour Visualisation')
    
    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    return fig
