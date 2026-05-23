"""MATH answer extraction and comparison utilities.

Handles diverse formats: fractions, radicals, polynomials, etc.
Uses sympy for symbolic equivalence checking.
"""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer, handling nested braces."""
    if text is None:
        return None
    idx = text.rfind('\\boxed{')
    if idx == -1:
        idx = text.rfind('boxed{')
        if idx == -1:
            return None
        idx += 6
    else:
        idx += 7

    depth = 1
    end = idx
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1

    if depth != 0:
        return None
    return text[idx:end-1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize a MATH answer string for comparison."""
    if answer is None:
        return ""
    s = answer.strip()
    s = s.replace('$', '')
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = s.replace(' ', '')
    s = s.replace('\\left(', '(').replace('\\right)', ')')
    s = s.replace('\\left[', '[').replace('\\right]', ']')
    s = s.replace('\\left\\{', '{').replace('\\right\\}', '}')
    s = s.replace('\\{', '{').replace('\\}', '}')
    s = s.replace('\\,', '').replace('\\!', '').replace('\\;', '')
    s = s.replace('\\quad', '').replace('\\qquad', '')
    s = s.replace('\\cdot', '*').replace('\\times', '*').replace('\\div', '/')
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    return s


def _try_numeric_equal(a: str, b: str) -> Optional[bool]:
    """Try numeric comparison after basic transformations."""
    def to_float(s):
        s = s.strip()
        if '/' in s and '\\' not in s:
            parts = s.split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        m = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', s)
        if m:
            try:
                return float(m.group(1)) / float(m.group(2))
            except (ValueError, ZeroDivisionError):
                pass
        try:
            return float(s)
        except ValueError:
            return None

    va = to_float(a)
    vb = to_float(b)
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-6
    return None


def _try_sympy_equal(a: str, b: str) -> Optional[bool]:
    """Try symbolic comparison via sympy. Returns None on parse failure."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify

        expr_a = parse_latex(a)
        expr_b = parse_latex(b)

        if expr_a == expr_b:
            return True

        diff = simplify(expr_a - expr_b)
        if diff == 0:
            return True

        try:
            val_a = float(expr_a.evalf())
            val_b = float(expr_b.evalf())
            if abs(val_a - val_b) < 1e-6:
                return True
        except (TypeError, ValueError, AttributeError):
            pass

        return False
    except Exception:
        return None


def answers_equal(pred: str, gold: str) -> bool:
    """Compare predicted answer against ground truth.

    Strategy: normalize -> string match -> numeric match -> sympy match.
    """
    if pred is None or gold is None:
        return False

    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)

    if norm_pred == norm_gold:
        return True

    num_result = _try_numeric_equal(norm_pred, norm_gold)
    if num_result is not None:
        return num_result

    sym_result = _try_sympy_equal(norm_pred, norm_gold)
    if sym_result is not None:
        return sym_result

    return False


if __name__ == "__main__":
    tests = [
        ("\\frac{1}{2}", "0.5", True),
        ("\\frac{3}{4}", "\\dfrac{3}{4}", True),
        ("x^2 + 1", "1 + x^2", True),
        ("42", "42", True),
        ("42", "43", False),
    ]
    for a, b, expected in tests:
        result = answers_equal(a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: answers_equal({a!r}, {b!r}) = {result} (expected {expected})")
