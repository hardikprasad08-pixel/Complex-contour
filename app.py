import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Import toolkit modules
from src.parser import safe_parse_expression
from src.contours import Circle, Line, Polygonal
from src.singularities import find_singularities
from src.residues import calculate_residue
from src.branches import analyze_branch_cuts
from src.numerical import integrate_trapezoidal, integrate_adaptive
from src.decision_engine import analyze_integration_strategy
from src.visualisation import plot_complex_domain
from src.validation import validate_numerical_vs_analytical
from src.ai_audit import AIAuditTrail
from src.reporting import get_physics_case_studies

# Configure Streamlit page
st.set_page_config(page_title="Complex Analysis Integration Toolkit", layout="wide")

# Initialize audit trail and session state
auditor = AIAuditTrail()
if 'expr_str' not in st.session_state:
    st.session_state['expr_str'] = "1/(z^2 + 1)"
if 'contour_type' not in st.session_state:
    st.session_state['contour_type'] = "Circle"
if 'circle_center' not in st.session_state:
    st.session_state['circle_center'] = 0.0 + 0j
if 'circle_radius' not in st.session_state:
    st.session_state['circle_radius'] = 2.0

st.title("∠ Complex Contour Integration & Residue Theory Toolkit")
st.write("An elite, modular mathematical computing system & visual proof laboratory.")

tabs = st.tabs([
    'Overview',
    'Workbench',
    'Singularity Explorer',
    'Branch-Cut Analyser',
    'Numerical Integration',
    'Benchmark Lab',
    'AI Audit',
    'Physics Applications',
    'Research Report'
])

def try_parse_current():
    try:
        return safe_parse_expression(st.session_state['expr_str'])
    except Exception as e:
        return None

def build_contour_from_state():
    ctype = st.session_state['contour_type']
    if ctype == "Circle":
        return Circle(center=st.session_state['circle_center'], radius=st.session_state['circle_radius'], ccw=True)
    elif ctype == "Line":
        return Line(start=0.0, end=2.0 + 2.0j)
    else:
        return Polygonal(vertices=[0, 1+1j, 2j, 0])

# -----------------------------------------------------
# 1. Overview Tab
# -----------------------------------------------------
with tabs[0]:
    st.header("Project Overview & Methodology")
    st.markdown("""
    ### Welcome to the Complex Contour Integration Laboratory
    This application provides an interactive workspace for studying and calculating complex integrals
    using symbolic and numerical techniques.

    #### Key Architecture & Modules:
    1. **Mathematical Parser (`src/parser.py`)**: Safe parsing of mathematical text.
    2. **Parameterized Contours (`src/contours.py`)**: Support for Circle, Line, and Polygonal contours.
    3. **Singularity Classifier (`src/singularities.py`)**: Identifies poles, removable, essential, and branch points.
    4. **Residue Calculator (`src/residues.py`)**: Symbolic residue calculations.
    5. **Branch-Cut Analyser (`src/branches.py`)**: Finds and visualizes cuts for multi-valued functions.
    6. **Numerical Integrators (`src/numerical.py`)**: Composite Trapezoidal and Adaptive Quadrature rules.
    7. **Decision Engine (`src/decision_engine.py`)**: Automatic recommendation systems and audit grades.
    8. **Visualizer (`src/visualisation.py`)**: Renders complex mappings, contours, singularities, and flow lines.
    """)

# -----------------------------------------------------
# 2. Workbench Tab
# -----------------------------------------------------
with tabs[1]:
    st.header("Mathematical Workbench")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Configure Expression and Contour")
        expr_input = st.text_input("Expression f(z):", value=st.session_state['expr_str'])
        st.session_state['expr_str'] = expr_input

        contour_sel = st.selectbox("Contour Type:", ["Circle", "Line", "Polygonal"], index=0)
        st.session_state['contour_type'] = contour_sel

        if contour_sel == "Circle":
            r_val = st.number_input("Radius:", value=2.0, min_value=0.1, step=0.1)
            st.session_state['circle_radius'] = r_val

        parsed_expr = try_parse_current()
        if parsed_expr is not None:
            st.success("✓ Expression parsed safely with SymPy!")
            st.latex(sp.latex(parsed_expr))
        else:
            st.error("✗ Invalid expression syntax or unsafe character detected.")

    with col2:
        st.subheader("Complex Plane Visualisation")
        if parsed_expr is not None:
            contour_obj = build_contour_from_state()
            sings = find_singularities(parsed_expr)
            cuts = analyze_branch_cuts(parsed_expr)

            f_lambda = sp.lambdify(sp.Symbol('z'), parsed_expr, 'numpy')
            f_safe = lambda z: complex(f_lambda(z))

            fig = plot_complex_domain(contour_obj, singularities=sings, branch_cuts=cuts, f=f_safe)
            st.pyplot(fig)
            plt.close(fig)

# -----------------------------------------------------
# 3. Singularity Explorer Tab
# -----------------------------------------------------
with tabs[2]:
    st.header("Singularity Explorer & Symbolic Residues")
    parsed_expr = try_parse_current()
    if parsed_expr is not None:
        sings = find_singularities(parsed_expr)
        if sings:
            for idx, sing in enumerate(sings):
                st.write(f"**Singularity #{idx+1}:** {sing['point']}")
                st.write(f"- Type: `{sing['type']}`")
                st.write(f"- Order: `{sing['order']}`")
                try:
                    res_val = calculate_residue(parsed_expr, sing['point'])
                    st.latex(rf"\text{{Residue at }} z = {sp.latex(sing['point'])} \quad \Rightarrow \quad {sp.latex(res_val)}")
                except Exception as e:
                    st.write(f"- Residue calculation failed: {str(e)}")
        else:
            st.info("No singularities detected for this expression.")
    else:
        st.error("Please configure a valid expression on the Workbench.")

# -----------------------------------------------------
# 4. Branch-Cut Analyser Tab
# -----------------------------------------------------
with tabs[3]:
    st.header("Branch-Cut Analyser")
    parsed_expr = try_parse_current()
    if parsed_expr is not None:
        cuts = analyze_branch_cuts(parsed_expr)
        if cuts:
            for idx, cut in enumerate(cuts):
                st.markdown(f"""
                **Branch Cut #{idx+1} ({cut['type']})**
                - **Branch Point**: {cut['branch_point']}
                - **Expression**: `{cut['expression']}`
                - **Principal Direction**: {cut['direction']}
                """)
        else:
            st.info("No branch cuts detected (holomorphic/meromorphic on C).")
    else:
        st.error("Configure a valid expression on the Workbench.")

# -----------------------------------------------------
# 5. Numerical Integration Tab
# -----------------------------------------------------
with tabs[4]:
    st.header("Numerical Contour Integration")
    parsed_expr = try_parse_current()
    if parsed_expr is not None:
        contour_obj = build_contour_from_state()
        f_lambda = sp.lambdify(sp.Symbol('z'), parsed_expr, 'numpy')
        f_safe = lambda z: complex(f_lambda(z))

        method = st.selectbox("Select Integration Algorithm:", ["Adaptive Quadrature", "Trapezoidal Rule"])
        if method == "Adaptive Quadrature":
            tol = st.number_input("Tolerance:", value=1e-8, format="%.1e")
            if st.button("Integrate"):
                ans = integrate_adaptive(f_safe, contour_obj, tolerance=tol)
                st.metric("Calculated Value:", f"{ans:.8f}")
        else:
            pts = st.slider("Sampling Points:", min_value=100, max_value=10000, value=1000, step=100)
            if st.button("Integrate"):
                ans = integrate_trapezoidal(f_safe, contour_obj, num_points=pts)
                st.metric("Calculated Value:", f"{ans:.8f}")
    else:
        st.error("Configure a valid expression on the Workbench.")

# -----------------------------------------------------
# 6. Benchmark Lab Tab
# -----------------------------------------------------
with tabs[5]:
    st.header("Benchmark Verification Lab")
    if os.path.exists('data/benchmarks.json'):
        with open('data/benchmarks.json', 'r') as f:
            benchmarks = json.load(f)

        selected_b_name = st.selectbox("Select Preloaded Benchmark Integral:", [b['name'] for b in benchmarks])
        b_data = next(b for b in benchmarks if b['name'] == selected_b_name)

        st.markdown(f"**Description:** {b_data['description']}")
        st.markdown(f"- **Expression:** `{b_data['expression']}`")
        st.markdown(f"- **Contour:** {b_data['contour']}")
        st.markdown(f"- **Analytical Solution:** `{b_data['analytical_solution']}`")

        if st.button("Run Benchmark Audit Validation"):
            b_expr = safe_parse_expression(b_data['expression'])
            b_f_lambda = sp.lambdify(sp.Symbol('z'), b_expr, 'numpy')
            b_f_safe = lambda z: complex(b_f_lambda(z))

            b_contour = Circle(center=0, radius=float(b_data['contour'].get('radius', 1.0)))
            analytical_parsed = complex(sp.parse_expr(b_data['analytical_solution']).evalf())

            report = validate_numerical_vs_analytical(b_f_safe, b_contour, analytical_parsed)
            st.write("**Adaptive Integration Result:**", report['adaptive_value'])
            st.write("**Adaptive Absolute Error:**", report['adaptive_abs_error'])

            st.subheader("Multi-sampling Trapezoidal Convergence")
            st.write(report['convergence_history'])

            auditor.log_event(
                expression=b_data['expression'],
                contour_info=b_data['contour'],
                decision_info={"recommended_method": "Benchmark Verification Lab Run"},
                results_info={"analytical": str(analytical_parsed), "numerical": str(report['adaptive_value'])}
            )
    else:
        st.error("Benchmark data file not found!")

# -----------------------------------------------------
# 7. AI Audit Tab
# -----------------------------------------------------
with tabs[6]:
    st.header("AI Audit Trail Log")
    logs = auditor.get_logs()
    if logs:
        st.write(f"Tracked {len(logs)} calculation events in the active session:")
        st.json(logs)
    else:
        st.info("No audit events captured yet. Run Workbench / Benchmarks to populate details.")

# -----------------------------------------------------
# 8. Physics Applications Tab
# -----------------------------------------------------
with tabs[7]:
    st.header("Predefined Physics Case Studies")
    studies = get_physics_case_studies()
    for study in studies:
        with st.expander(study['title']):
            st.markdown(study['description'])
            st.latex(study['formula_latex'])
            st.markdown(f"**Target expression:** `{study['expression']}`")
            if st.button(f"Evaluate {study['title']}"):
                p_expr = safe_parse_expression(study['expression'])
                p_f_lambda = sp.lambdify(sp.Symbol('z'), p_expr, 'numpy')
                p_f_safe = lambda z: complex(p_f_lambda(z))
                p_contour = Circle(center=complex(study['default_contour']['center']), radius=float(study['default_contour']['radius']))

                sol = integrate_adaptive(p_f_safe, p_contour)
                st.success(f"Adaptive Integrator Output: {sol}")

# -----------------------------------------------------
# 9. Research Report Tab
# -----------------------------------------------------
with tabs[8]:
    st.header("Research & Session Report")
    st.markdown("""
    Generate a formatted comprehensive LaTeX / HTML summary report detailing all active operations
    completed during this computational analytical session.
    """)
    logs = auditor.get_logs()
    if logs:
        report_md = "# Complex Analysis Session Report\n\n"
        for idx, item in enumerate(logs):
            report_md += f"### Session Run #{idx+1}\n"
            report_md += f"- **Expression:** `{item['expression_str']}`\n"
            report_md += f"- **Contour:** `{item['contour']}`\n"
            report_md += f"- **Numerical result:** `{item['results'].get('numerical')}`\n\n"

        st.download_button("Download Report as Markdown", data=report_md, file_name="Session_Report.md")
        st.markdown(report_md)
    else:
        st.info("No active data points or logs collected yet to construct the Research Report.")
