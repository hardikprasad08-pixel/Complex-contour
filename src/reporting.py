from typing import Dict, Any, List

def get_physics_case_studies() -> List[Dict[str, Any]]:
    """
    Returns predefined physics and engineering case studies with rich narrative details,
    mathematical context, formulations, and default analytic structures.
    """
    return [
        {
            'title': "Quantum Mechanics: Free-Particle Green's Function",
            'formula_latex': r"G(x, x'; E) = \\frac{1}{2\\pi} \\int_{-\\infty}^{\\infty} \\frac{e^{ik(x - x')}}{E - \\frac{\\hbar^2 k^2}{2m} + i\\epsilon} dk",
            'description': """
            In quantum scattering theory, the propagator or Green's function describes how a particle propagates
            from x' to x at energy E. The mathematical singularity appears as a pole on or near the real axis. 
            By shifting the pole into the complex plane via the i*epsilon prescription, contour integration 
            using the Residue Theorem produces the outgoing or incoming spherical waves.
            """,
            'expression': "exp(I*z) / (z^2 + 1)",
            'default_contour': {
                'type': 'Circle',
                'center': 0,
                'radius': 2.0,
                'ccw': True
            },
            'analytical_solution': "pi * E / I"
        },
        {
            'title': "Classical Electrodynamics: Dispersive Wave Packets",
            'formula_latex': r"D(\\omega) = \\int_{-\\infty}^{\\infty} \\frac{\\epsilon(\\omega') E(\\omega') e^{-i\\omega' t}}{\\omega' - \\omega} d\\omega'",
            'description': """
            When modeling light dispersion in materials, Kramers-Kronig relation and Cauchy principal values
            are used to prove causality (the response cannot precede the stimulus). The complex integration is 
            performed on a contour bounding the upper half complex frequency plane, where the material response 
            is holomorphic.
            """,
            'expression': "1 / (z - 1)",
            'default_contour': {
                'type': 'Circle',
                'center': 0,
                'radius': 2.0,
                'ccw': True
            },
            'analytical_solution': "2 * pi * I"
        }
    ]
