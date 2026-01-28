"""Normalize arguments in dataset.jsonl using utils.arg_normalizer.normalize_action_arguments
Writes normalized dataset to dataset_normalized.jsonl (does not overwrite original).
"""
import json
from pathlib import Path
import sys
from pathlib import Path
# Ensure repo root is on sys.path so 'utils' package imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.arg_normalizer import normalize_action_arguments

DATASET = Path('dataset.jsonl')
OUT = Path('dataset_normalized.jsonl')

with DATASET.open('r', encoding='utf-8') as fr, OUT.open('w', encoding='utf-8') as fw:
    for i, line in enumerate(fr, 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        task = obj.get('task') or obj
        actions = task.get('gold_actions') or task.get('expected_actions') or []
        new_actions = []
        for a in actions:
            name = a.get('name')
            args = a.get('arguments', {}) or {}
            new_args = normalize_action_arguments(name, args)
            new_actions.append({'name': name, 'arguments': new_args})
        # Write out normalized record
        out_obj = dict(obj)
        if 'gold_actions' in task:
            out_obj['gold_actions'] = new_actions
        elif 'expected_actions' in task:
            out_obj['expected_actions'] = new_actions
        else:
            # Keep shape
            out_obj['gold_actions'] = new_actions
        fw.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

print(f"Wrote normalized dataset to {OUT}")