import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from analysis_helpers import classify_action, load_test_results, extract_expected_actions


# Reuse shared helpers from analysis_helpers for classification and extraction


def analyze(results):
    by_action_count = defaultdict(lambda: {'total': 0, 'success': 0, 'exact_match': 0})
    by_read_write = defaultdict(lambda: {'total': 0, 'success': 0, 'exact_match': 0})
    # groups by expected-action-sequence
    expected_groups = {}

    total = 0
    for entry in results:
        total += 1
        expected_actions = extract_expected_actions(entry)
        action_count = len(expected_actions)

        # classify read/write counts
        read = 0
        write = 0
        for a in expected_actions:
            if not a:
                continue
            kind = classify_action(a)
            if kind == 'write':
                write += 1
            else:
                read += 1

        mix_key = f"r{read}_w{write}"

        # update buckets
        ac_bucket = by_action_count[str(action_count)]
        ac_bucket['total'] += 1
        if entry.get('success'):
            ac_bucket['success'] += 1
        if entry.get('exact_action_match'):
            ac_bucket['exact_match'] += 1

        rw_bucket = by_read_write[mix_key]
        rw_bucket['total'] += 1
        if entry.get('success'):
            rw_bucket['success'] += 1
        if entry.get('exact_action_match'):
            rw_bucket['exact_match'] += 1

        # accumulate expected-action-sequence groups (preserve order)
        # build canonical key
        if expected_actions:
            key = ' -> '.join([str(a) for a in expected_actions])
        else:
            key = '<none>'

        g = expected_groups.get(key)
        if g is None:
            g = {'total': 0, 'successful': 0, 'exact_action_match': 0, 'sample_task_ids': []}
            expected_groups[key] = g

        g['total'] += 1
        if entry.get('success'):
            g['successful'] += 1
        if entry.get('exact_action_match'):
            g['exact_action_match'] += 1
        if len(g['sample_task_ids']) < 5:
            g['sample_task_ids'].append(entry.get('task_id'))

    # Convert to rates
    def bucket_summary(d):
        out = {}
        for k, v in sorted(d.items(), key=lambda x: (int(x[0]) if x[0].isdigit() else x[0])):
            t = v['total']
            out[k] = {
                'total': t,
                'success_count': v['success'],
                'success_rate': (v['success']/t if t else 0.0),
                'exact_match_count': v['exact_match'],
                'exact_match_rate': (v['exact_match']/t if t else 0.0),
            }
        return out

    return {
        'summary': {'total_tasks_processed': total},
        'by_action_count': bucket_summary(by_action_count),
        'by_read_write_mix': bucket_summary(by_read_write),
        'expected_action_distribution': {
            'summary_count': len(expected_groups),
            'groups': sorted(
                [
                    {
                        'expected_action_sequence': k,
                        'total': v['total'],
                        'successful': v['successful'],
                        'exact_action_match': v['exact_action_match'],
                        'success_rate': (v['successful']/v['total'] if v['total'] else 0.0),
                        'exact_match_rate': (v['exact_action_match']/v['total'] if v['total'] else 0.0),
                        'sample_task_ids': v['sample_task_ids']
                    }
                    for k, v in expected_groups.items()
                ],
                key=lambda x: x['total'],
                reverse=True
            )
        }
    }


if __name__ == '__main__':
    root = Path('.')
    test_results = sorted(root.glob('results/test_results_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not test_results:
        print("No test results found in 'results' directory")
        raise SystemExit(1)
    test_path = test_results[0]
    print(f"Using latest test results: {test_path}")

    results = load_test_results(test_path)
    report = analyze(results)

    out_dir = root / 'results/analysis'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'analysis_successful_tasks.json'
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Print concise summary
    print(f"Processed {report['summary']['total_tasks_processed']} tasks")
    print('\nBy action count:')
    for k, v in report['by_action_count'].items():
        print(f"  {k} actions: {v['total']} tasks — success {v['success_count']}/{v['total']} ({v['success_rate']:.2%}), exact_match {v['exact_match_count']}/{v['total']} ({v['exact_match_rate']:.2%})")

    print('\nBy read/write mix:')
    for k, v in report['by_read_write_mix'].items():
        print(f"  {k}: {v['total']} tasks — success {v['success_count']}/{v['total']} ({v['success_rate']:.2%}), exact_match {v['exact_match_count']}/{v['total']} ({v['exact_match_rate']:.2%})")

    # Print top expected-action groups
    print('\nTop 10 expected-action groups:')
    groups = report.get('expected_action_distribution', {}).get('groups', [])
    for i, g in enumerate(groups[:10], start=1):
        seq = g.get('expected_action_sequence')
        total = g.get('total')
        succ = g.get('successful')
        exact = g.get('exact_action_match')
        srate = g.get('success_rate', 0.0)
        erate = g.get('exact_match_rate', 0.0)
        samples = g.get('sample_task_ids', [])
        print(f"{i:2}. {total:3} tasks | success={succ}/{total} ({srate:.2%}) | exact={exact}/{total} ({erate:.2%})")
        print(f"     sequence: {seq}")
        if samples:
            print(f"     samples: {', '.join(samples)}")

    print(f"Saved analysis to {out_path}")

    # Generate plots if matplotlib available
    if plt is not None:
        out_dir = out_dir
        # top expected-action groups
        groups = report.get('expected_action_distribution', {}).get('groups', [])
        top = groups[:10]
        if top:
            labels = [g['expected_action_sequence'][:60] + ("..." if len(g['expected_action_sequence'])>60 else "") for g in top]
            vals = [g['total'] for g in top]
            plt.figure(figsize=(10,4))
            plt.barh(range(len(vals))[::-1], vals, color='C0')
            plt.yticks(range(len(vals))[::-1], labels)
            plt.title('Top expected-action groups (by task count)')
            plt.tight_layout()
            p = out_dir / 'top_expected_action_groups.png'
            plt.savefig(p)
            plt.close()

        # tasks by action count
        bac = report.get('by_action_count', {})
        if bac:
            xs = sorted(bac.items(), key=lambda x: int(x[0]))
            labels = [k for k,v in xs]
            totals = [v['total'] for k,v in xs]
            successes = [v['success_count'] for k,v in xs]

            # grouped bar: total vs success
            x_pos = range(len(labels))
            plt.figure(figsize=(6,3))
            plt.bar(x_pos, totals, color='C1', label='total')
            plt.bar(x_pos, successes, color='C0', label='success')
            plt.xticks(x_pos, labels)
            plt.title('Tasks by action count — total vs success')
            plt.xlabel('Number of actions')
            plt.ylabel('Number of tasks')
            plt.legend()
            plt.tight_layout()
            p2 = out_dir / 'tasks_by_action_count_total_vs_success.png'
            plt.savefig(p2)
            plt.close()

            # success rate bar
            rates = [(s / t if t else 0.0) for s, t in zip(successes, totals)]
            plt.figure(figsize=(6,3))
            plt.bar(labels, rates, color='C2')
            plt.title('Tasks by action count — success rate')
            plt.xlabel('Number of actions')
            plt.ylabel('Success rate')
            plt.ylim(0,1)
            plt.tight_layout()
            p2b = out_dir / 'tasks_by_action_count_success_rate.png'
            plt.savefig(p2b)
            plt.close()

        # read/write mix
        mix = report.get('by_read_write_mix', {})
        if mix:
            items = sorted(mix.items(), key=lambda x: x[1]['total'], reverse=True)
            labels = [k for k,v in items]
            totals = [v['total'] for k,v in items]
            successes = [v['success_count'] for k,v in items]

            # grouped bar: total vs success
            x_pos = range(len(labels))
            plt.figure(figsize=(8,3))
            plt.bar(x_pos, totals, color='C1', label='total')
            plt.bar(x_pos, successes, color='C0', label='success')
            plt.xticks(x_pos, labels, rotation=45, ha='right')
            plt.title('Read/Write mix — total vs success')
            plt.ylabel('Number of tasks')
            plt.legend()
            plt.tight_layout()
            p3 = out_dir / 'read_write_mix_total_vs_success.png'
            plt.savefig(p3)
            plt.close()

            # success rate bar
            rates = [(s / t if t else 0.0) for s, t in zip(successes, totals)]
            plt.figure(figsize=(8,3))
            plt.bar(labels, rates, color='C2')
            plt.title('Read/Write mix — success rate')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0,1)
            plt.tight_layout()
            p3b = out_dir / 'read_write_mix_success_rate.png'
            plt.savefig(p3b)
            plt.close()

        print(f"Saved plots to {out_dir} (top_expected_action_groups.png, tasks_by_action_count.png, read_write_mix.png)")
    else:
        print('matplotlib not available; skipping plots')
