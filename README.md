# Complex Contour Integration and Residue Theory Toolkit

An elite, modular mathematical computing system and visual proof laboratory written in Python for analyzing complex mathematical functions, locating and classifying singularities, and validating contour integrals symbolically and numerically.

## Repository Architecture
```
- app.py                      # Multi-tab Streamlit dashboard interface
- requirements.txt            # System dependencies
- src/
  - parser.py                 # Safe SymPy parsing engine
  - contours.py               # Parameterized circular, linear, polygonal contours
  - singularities.py          # Singularity identification & classification
  - residues.py               # Symbolic residue computation
  - branches.py               # Multivalued function branch-cut mapping
  - numerical.py              # Trapezoidal and Adaptive numerical integration
  - decision_engine.py        # Rule-based recommender and quality audit
  - visualisation.py          # Vector fields, contours, and complex planes plotting
  - validation.py             # Convergence analysis and statistical verification
  - ai_audit.py               # Audit trails tracking session state
  - reporting.py              # Engineering / physics-based case study templates
- tests/
  - test_complex_toolkit.py  # Pytest suite with 12 mathematical edge cases
```

## Mathematical Formulations
### Residue Theorem
$$\oint_C f(z) dz = 2\pi i \sum_{k} \text{Res}(f, z_k)$$

### Composite Trapezoidal Rule
$$\int_C f(z) dz = \int_0^1 f(z(t)) z'(t) dt \approx \left( \frac{f(z(t_0))z'(t_0) + f(z(t_n))z'(t_n)}{2} + \sum_{j=1}^{n-1} f(z(t_j))z'(t_j) \right) \Delta t$$

## Installation & Run Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run unit tests: `pytest`
3. Start the dashboard application: `streamlit run app.py`
