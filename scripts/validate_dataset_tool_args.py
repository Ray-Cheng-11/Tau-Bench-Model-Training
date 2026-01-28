"""Validate dataset gold_actions arguments against tool invoke signatures.
Outputs a JSON report to results/dataset_tool_arg_mismatch_report.json
"""
import os
import re
import json
from pathlib import Path

TOOLS_DIR = Path("envs/retail/tools")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset.jsonl', help='Path to dataset jsonl')
args = parser.parse_args()

DATASET_PATH = Path(args.dataset)
REPORT_PATH = Path("results/dataset_tool_arg_mismatch_report.json")

# Gather tool invoke parameter names using AST parsing for accuracy
import ast

tool_sigs = {}
for py in TOOLS_DIR.glob("*.py"):
    name = py.stem
    src = py.read_text(encoding='utf-8')
    try:
        tree = ast.parse(src)
    except Exception:
        continue
    param_names = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'invoke':
            # get arg names excluding 'self' and 'data'
            param_names = [a.arg for a in node.args.args if a.arg not in ('self', 'data')]
            break
    # get canonical function name from get_info JSON in source as a fallback
    fn_m = re.search(r'"name"\s*:\s*"([^\"]+)"', src)
    fn_name = fn_m.group(1) if fn_m else name
    tool_sigs[fn_name] = {
        'file': str(py),
        'parameters': param_names
    }

# Read dataset and compare
report = {
    'summary': {
        'total_actions': 0,
        'tools_seen': {},
        'mismatches': 0
    },
    'tools': {}
}

# Initialize per-tool
for tool in tool_sigs:
    report['tools'][tool] = {
        'expected_parameters': tool_sigs[tool]['parameters'],
        'occurrences': 0,
        'examples': [],
        'mismatches': []
    }

# scan dataset
with DATASET_PATH.open('r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        task = obj.get('task') or obj
        actions = task.get('gold_actions') or task.get('expected_actions') or []
        for a in actions:
            report['summary']['total_actions'] += 1
            name = a.get('name')
            args = a.get('arguments') or {}
            arg_keys = set(args.keys())
            if name not in report['tools']:
                # record unknown tools too
                report['tools'].setdefault(name, {
                    'expected_parameters': None,
                    'occurrences': 0,
                    'examples': [],
                    'mismatches': [{'line': i, 'provided': list(arg_keys), 'expected': None}]
                })
                report['tools'][name]['occurrences'] += 1
                report['tools'][name]['examples'].append({'line': i, 'provided': list(arg_keys)})
                report['summary']['mismatches'] += 1
                continue
            # known tool
            exp_params = report['tools'][name].get('expected_parameters')
            report['tools'][name]['occurrences'] += 1
            if len(report['tools'][name]['examples']) < 5:
                report['tools'][name]['examples'].append({'line': i, 'provided': list(arg_keys)})

            # If we don't have expected params (unknown tool), record mismatch and continue
            if exp_params is None:
                report['tools'][name]['mismatches'].append({
                    'line': i,
                    'missing': None,
                    'extra': None,
                    'provided': list(arg_keys),
                    'expected': None
                })
                report['summary']['mismatches'] += 1
                continue

            exp = set(exp_params)
            missing = sorted(list(exp - arg_keys))
            extra = sorted(list(arg_keys - exp))
            if missing or extra:
                report['tools'][name]['mismatches'].append({
                    'line': i,
                    'missing': missing,
                    'extra': extra,
                    'provided': list(arg_keys),
                    'expected': list(exp)
                })
                report['summary']['mismatches'] += 1

# Add per-tool summary
for t, info in report['tools'].items():
    info['num_mismatches'] = len(info.get('mismatches', []))

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(f"Wrote report to {REPORT_PATH}")
print(json.dumps(report['summary'], indent=2))
