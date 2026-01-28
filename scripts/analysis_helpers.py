from pathlib import Path
import json
from typing import List, Optional, Any
from configs import TauBenchConfig

# Use TauBenchConfig.write_keywords instead of a separate constants module
WRITE_KEYWORDS = TauBenchConfig().write_keywords


def classify_action(action_name: str) -> str:
    """Classify an action name as 'write' or 'read' based on keywords"""
    name = action_name.lower() if action_name else ''
    for w in WRITE_KEYWORDS:
        if w in name:
            return 'write'
    return 'read'


def load_test_results(path: Path):
    """Read JSON test results from path and return parsed data"""
    return json.loads(path.read_text(encoding='utf-8'))


def extract_expected_actions(entry: dict) -> List[str]:
    """Extract expected action names from an analyzer entry.

    Strategy:
    - Prefer 'details.expected_actions' when present and is a list of strings
    - Fallback to 'task.expected_actions' where entries may be dicts with 'name'
    """
    det = entry.get('details') or {}
    expected = det.get('expected_actions')
    if expected and isinstance(expected, list) and expected and isinstance(expected[0], str):
        return expected

    task = entry.get('task') or {}
    exp2 = task.get('expected_actions') or []
    names = []
    for a in exp2:
        if isinstance(a, dict):
            n = a.get('name')
            if n:
                names.append(n)
        elif isinstance(a, str):
            names.append(a)
    return names
