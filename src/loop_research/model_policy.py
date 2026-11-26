"""Submission limits for this reversible local sklearn benchmark, not an OS sandbox."""
from __future__ import annotations

import ast
import math

from .training import validate_code

LIMITS = {'pipeline_disk_cache': False, 'n_jobs': 1, 'max_polynomial_degree': 3,
          'max_transformed_features': 8192, 'max_estimators': 512}


def validate_proposal(code: str, seed: int):
    tree = validate_code(code)

    def literal(node):
        if isinstance(node, ast.Name) and node.id == 'seed':
            return seed
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError) as exc:
            raise ValueError('Compute-limited parameters must be literal values') from exc

    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword):
            continue
        if node.arg == 'memory' and literal(node.value) is not None:
            raise ValueError('Pipeline disk caching is not permitted by the local benchmark protocol')
        if node.arg == 'n_jobs' and literal(node.value) not in (None, 1):
            raise ValueError('Local training requires n_jobs=1 or None')
        if node.arg == 'n_estimators':
            value = literal(node.value)
            if type(value) is not int or not 1 <= value <= LIMITS['max_estimators']:
                raise ValueError('Estimator count exceeds local compute bounds')
    def parameters(node):
        return {kw.arg: kw.value for kw in node.keywords}

    def stages(node):
        if not isinstance(node, ast.Call):
            # Invalid factory syntax is a protocol error handled by the controller.
            raise ValueError('Every model stage must be a constructed estimator')  # noqa: TRY004
        if node.func.id == 'make_pipeline':
            return [stage for arg in node.args for stage in stages(arg)]
        if node.func.id == 'Pipeline':
            steps = node.args[0] if node.args else parameters(node).get('steps')
            if not isinstance(steps, (ast.List, ast.Tuple)):
                raise ValueError('Pipeline steps must be an explicit sequence')
            result = []
            for item in steps.elts:
                if not isinstance(item, (ast.List, ast.Tuple)) or len(item.elts) != 2:
                    raise ValueError('Pipeline steps must be named estimator pairs')
                result.extend(stages(item.elts[1]))
            return result
        return [node]

    expression = next(n for n in tree.body if isinstance(n, ast.FunctionDef)).body[0].value
    features = 30  # Frozen bundled benchmark has 30 input features.
    for node in stages(expression):
        params = parameters(node)
        if node.func.id == 'PolynomialFeatures':
            value = literal(params['degree']) if 'degree' in params else literal(node.args[0]) if node.args else 2
            degrees = list(value) if isinstance(value, (list, tuple)) else [value]
            if not degrees or any(type(d) is not int or not 0 <= d <= LIMITS['max_polynomial_degree'] for d in degrees):
                raise ValueError('Polynomial degree exceeds local compute bounds')
            # Conservative dimension ceiling; no data allocation or model fitting occurs here.
            features = math.comb(features + max(degrees), max(degrees))
        elif node.func.id == 'PCA':
            value = literal(params['n_components']) if 'n_components' in params else literal(node.args[0]) if node.args else None
            features = min(features, value) if type(value) is int and value > 0 else min(features, 341)
        elif node.func.id == 'SelectKBest':
            value = literal(params['k']) if 'k' in params else 10
            if type(value) is int and value >= 0:
                features = min(features, value)
        if features > LIMITS['max_transformed_features']:
            raise ValueError('Feature expansion exceeds local compute bounds')
    return tree
