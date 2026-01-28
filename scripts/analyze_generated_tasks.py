import json
from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
from analysis_helpers import classify_action



# Reuse common classify_action from analysis_helpers


def analyze(tasks):
    per_task = []
    agg = {
        'total_tasks': 0,
        'total_actions': 0,
        'actions_by_name': Counter(),
        'tasks_by_action_count': Counter(),
        'tasks_by_read_write_mix': Counter(),
    }

    for task in tasks:
        agg['total_tasks'] += 1
        actions = task.get('agt', []) or []
        action_count = len(actions)
        agg['total_actions'] += action_count
        agg['tasks_by_action_count'][action_count] += 1

        read_count = 0
        write_count = 0
        names = []
        for a in actions:
            name = a.get('name') if isinstance(a, dict) else str(a)
            names.append(name)
            agg['actions_by_name'][name] += 1
            kind = classify_action(name)
            if kind == 'write':
                write_count += 1
            else:
                read_count += 1

        per_task.append({
            'q': task.get('q', '')[:200],
            'action_count': action_count,
            'read_actions': read_count,
            'write_actions': write_count,
            'action_names': names,
        })

        mix_key = f"r{read_count}_w{write_count}"
        agg['tasks_by_read_write_mix'][mix_key] += 1

    # Convert counters to normal dicts
    agg['actions_by_name'] = dict(agg['actions_by_name'])
    agg['tasks_by_action_count'] = dict(agg['tasks_by_action_count'])
    agg['tasks_by_read_write_mix'] = dict(agg['tasks_by_read_write_mix'])

    # extra metrics
    distinct_actions = len([k for k in agg['actions_by_name'].keys() if k])
    agg['distinct_actions'] = distinct_actions
    # coverage (fraction of available tools) - compute later when tools available

    # entropy (Shannon) of action frequency distribution
    freqs = list(agg['actions_by_name'].values())
    total = sum(freqs) or 1
    entropy = -sum((f/total) * math.log2(f/total) for f in freqs if f > 0)
    agg['action_entropy'] = entropy

    return {'per_task': per_task, 'aggregate': agg}


if __name__ == '__main__':
    path = Path('generated_tasks/Sampled_Tasks.json')
    if not path.exists():
        print('Sampled_Tasks.json not found in repo root')
        raise SystemExit(1)

    tasks = json.loads(path.read_text(encoding='utf-8'))
    report = analyze(tasks)

    out_dir = Path('results/analysis')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'analysis_generated_tasks.json'
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Print a concise summary
    agg = report['aggregate']
    print('Analysis summary:')
    print(f"  Total tasks: {agg['total_tasks']}")
    print(f"  Total actions: {agg['total_actions']}")
    print(f"  Average actions per task: {agg['total_actions'] / agg['total_tasks'] if agg['total_tasks'] else 0:.2f}")
    print('  Distribution of tasks by action count:')
    for k, v in sorted(agg['tasks_by_action_count'].items()):
        print(f"    {k} actions: {v} tasks")

    print('  Read/Write mix distribution:')
    for k, v in sorted(agg['tasks_by_read_write_mix'].items()):
        print(f"    {k}: {v} tasks")

    print('  Top actions by frequency:')
    top_actions = sorted(agg['actions_by_name'].items(), key=lambda x: x[1], reverse=True)[:10]
    for name, cnt in top_actions:
        print(f"    {name}: {cnt}")

    print(f"  Distinct actions used: {agg.get('distinct_actions',0)}")
    print(f"  Action frequency entropy (bits): {agg.get('action_entropy',0):.3f}")

    # Generate plots if matplotlib available
    if plt is not None:
        # Top actions bar
        names = [n for n,c in top_actions]
        counts = [c for n,c in top_actions]
        plt.figure(figsize=(8,4))
        plt.bar(names, counts, color='C0')
        plt.title('Top actions by frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / 'top_actions.png')
        plt.close()

        # Distribution of tasks by action count
        x = sorted(agg['tasks_by_action_count'].items())
        xs = [k for k,v in x]
        ys = [v for k,v in x]
        plt.figure(figsize=(6,3))
        plt.bar([str(k) for k in xs], ys, color='C1')
        plt.title('Tasks by action count')
        plt.xlabel('Number of actions')
        plt.ylabel('Number of tasks')
        plt.tight_layout()
        plt.savefig(out_dir / 'tasks_by_action_count.png')
        plt.close()

        # Read/write mix
        mix_items = sorted(agg['tasks_by_read_write_mix'].items(), key=lambda x: x[1], reverse=True)
        mix_labels = [k for k,v in mix_items]
        mix_vals = [v for k,v in mix_items]
        plt.figure(figsize=(8,3))
        plt.bar(mix_labels, mix_vals, color='C2')
        plt.title('Read/Write mix distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / 'read_write_mix.png')
        plt.close()

        print(f"Saved plots to {out_dir} (top_actions.png, tasks_by_action_count.png, read_write_mix.png)")
    else:
        print('matplotlib not available; skipping plots')

    print(f"Saved full analysis to {out_path}")
