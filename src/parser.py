import sympy as sp
import re

def safe_parse_expression(expr_str: str) -> sp.Expr:
    """
    Safely parses a mathematical string expression containing 'z' into a SymPy expression.
    Ensures the string only contains allowed mathematical functions, operators, and characters
    to prevent arbitrary code execution.
    """
    if not isinstance(expr_str, str):
        raise TypeError("Expression must be a string.")

    # Normalize whitespaces
    expr_str = expr_str.strip()

    # Security Check: Reject suspicious patterns or symbols
    # Allowed: z, pi, E, I, exp, log, sin, cos, tan, sinh, cosh, tanh, sqrt, asin, acos, atan
    if re.search(r'[^a-zA-Z0-9\s\+\-\*\/\^\(\)\.,]', expr_str):
        raise ValueError("Expression contains invalid characters or symbols.")

    # Check for forbidden Python keywords and builtins
    forbidden_keywords = ['import', 'exec', 'eval', 'os', 'sys', 'lambda', 'def', 'class', '_', 'self']
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', expr_str):
            raise ValueError(f"Expression contains forbidden keyword: {kw}")

    # Define local SymPy variables and functions
    z = sp.Symbol('z')
    local_dict = {
        'z': z,
        'pi': sp.pi,
        'E': sp.E,
        'I': sp.I,
        'exp': sp.exp,
        'log': sp.log,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        'sqrt': sp.sqrt,
        'asin': sp.asin,
        'acos': sp.acos,
        'atan': sp.atan
    }

    try:
        # Convert standard caret symbol to power operator
        expr_str_cleaned = expr_str.replace('^', '**')
        parsed_expr = sp.parse_expr(expr_str_cleaned, local_dict=local_dict, evaluate=False)

        # Double check that the parsed expression contains no unexpected free symbols
        for symbol in parsed_expr.free_symbols:
            if symbol != z:
                raise ValueError(f"Unexpected free symbol in expression: {symbol}")

        return parsed_expr
    except Exception as e:
        raise ValueError(f"Failed to safely parse mathematical expression: {str(e)}")
