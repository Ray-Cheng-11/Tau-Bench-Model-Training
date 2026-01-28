"""
Analyze Real Data References in Generated Tasks

This script analyzes how many actions in generated tasks reference real data
from the environment (user_id, order_id, product_id in envs/retail/data).

Usage:
    python scripts/analyze_real_data_references.py <task_file.json>
    python scripts/analyze_real_data_references.py generated_tasks/Completed_Tasks.json

The script will:
- Load the task file and environment data
- Check each action's user_id/order_id/product_id against real data
- Generate detailed statistics and reports
- Save results to a JSON file
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_reader import TauBenchDataReader


def normalize_order_id(order_id: str) -> str:
    """Normalize order_id to match envs data format (#W prefix)."""
    if not order_id:
        return ''
    
    normalized = order_id.strip().strip('"\'')
    # Remove existing # or #W prefix if present
    if normalized.startswith('#W'):
        normalized = normalized[2:]
    elif normalized.startswith('#'):
        normalized = normalized[1:]
    # Remove W prefix if present
    if normalized.startswith('W'):
        normalized = normalized[1:]
    # Add standard #W prefix
    return f"#W{normalized}"


def analyze_real_data_references(tasks, data_reader):
    """
    Analyze real data references in tasks.
    
    Returns:
        Dict with detailed statistics about real data references
    """
    if not tasks:
        return {
            'error': 'No tasks provided',
            'total_tasks': 0
        }
    
    # Load real data
    try:
        data = data_reader.read_data_files()
        users = data.get('users', {})
        orders = data.get('orders', {})
        products = data.get('products', {})
    except Exception as e:
        return {
            'error': f'Failed to load data: {str(e)}',
            'total_tasks': len(tasks)
        }
    
    # Statistics
    total_tasks = len(tasks)
    total_actions = 0
    
    # ID reference tracking
    total_user_ids = 0
    valid_user_ids = 0
    invalid_user_ids = []
    
    total_order_ids = 0
    valid_order_ids = 0
    invalid_order_ids = []
    
    total_product_ids = 0
    valid_product_ids = 0
    invalid_product_ids = []
    
    # Order-User association validation
    total_order_user_checks = 0
    valid_order_user_associations = 0
    invalid_order_user_associations = []
    
    # Per-task tracking
    tasks_with_valid_references = 0
    tasks_with_any_references = 0
    
    # Analyze each task
    for task_idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        
        task_has_references = False
        task_all_valid = True
        task_reference_count = 0
        task_valid_count = 0
        
        # Track the canonical user_id for this task (most common one)
        task_user_ids = []
        
        actions = task.get('agt', [])
        if not isinstance(actions, list):
            continue
        
        # First pass: collect all user_ids in this task
        for action in actions:
            if not isinstance(action, dict):
                continue
            args = action.get('arguments', {})
            if isinstance(args, dict) and 'user_id' in args:
                task_user_ids.append(args['user_id'])
        
        # Determine canonical user_id (most common)
        canonical_user_id = None
        if task_user_ids:
            from collections import Counter
            canonical_user_id = Counter(task_user_ids).most_common(1)[0][0]
        
        for action in actions:
            if not isinstance(action, dict):
                continue
            
            total_actions += 1
            args = action.get('arguments', {})
            if not isinstance(args, dict):
                continue
            
            action_name = action.get('name', 'unknown')
            action_user_id = args.get('user_id')
            
            # Check user_id
            if 'user_id' in args:
                user_id = args['user_id']
                total_user_ids += 1
                task_has_references = True
                task_reference_count += 1
                
                if user_id in users:
                    valid_user_ids += 1
                    task_valid_count += 1
                else:
                    task_all_valid = False
                    invalid_user_ids.append({
                        'task_index': task_idx,
                        'user_id': user_id,
                        'action': action_name,
                        'query_preview': task.get('q', '')[:100]
                    })
            
            # Check order_id
            if 'order_id' in args:
                order_id = str(args['order_id'])
                total_order_ids += 1
                task_has_references = True
                task_reference_count += 1
                
                normalized_oid = normalize_order_id(order_id)
                
                if normalized_oid in orders:
                    valid_order_ids += 1
                    task_valid_count += 1
                    
                    # Additional check: does this order belong to the user?
                    if action_user_id and canonical_user_id:
                        total_order_user_checks += 1
                        order_data = orders[normalized_oid]
                        order_owner = order_data.get('user_id')
                        
                        if order_owner != canonical_user_id:
                            task_all_valid = False
                            invalid_order_user_associations.append({
                                'task_index': task_idx,
                                'order_id': order_id,
                                'normalized': normalized_oid,
                                'order_owner': order_owner,
                                'action_user_id': action_user_id,
                                'canonical_user_id': canonical_user_id,
                                'action': action_name,
                                'query_preview': task.get('q', '')[:100]
                            })
                        else:
                            valid_order_user_associations += 1
                else:
                    task_all_valid = False
                    invalid_order_ids.append({
                        'task_index': task_idx,
                        'order_id': order_id,
                        'normalized': normalized_oid,
                        'action': action_name,
                        'query_preview': task.get('q', '')[:100]
                    })
            
            # Check product_id
            if 'product_id' in args:
                product_id = args['product_id']
                total_product_ids += 1
                task_has_references = True
                task_reference_count += 1
                
                if product_id in products:
                    valid_product_ids += 1
                    task_valid_count += 1
                else:
                    task_all_valid = False
                    invalid_product_ids.append({
                        'task_index': task_idx,
                        'product_id': product_id,
                        'action': action_name,
                        'query_preview': task.get('q', '')[:100]
                    })
        
        if task_has_references:
            tasks_with_any_references += 1
            if task_all_valid and task_reference_count > 0:
                tasks_with_valid_references += 1
    
    # Calculate totals
    total_id_references = total_user_ids + total_order_ids + total_product_ids
    valid_references = valid_user_ids + valid_order_ids + valid_product_ids
    invalid_references = len(invalid_user_ids) + len(invalid_order_ids) + len(invalid_product_ids)
    
    reference_ratio = (valid_references / total_id_references * 100) if total_id_references > 0 else 0.0
    task_validity_ratio = (tasks_with_valid_references / tasks_with_any_references * 100) if tasks_with_any_references > 0 else 0.0
    order_user_match_ratio = (valid_order_user_associations / total_order_user_checks * 100) if total_order_user_checks > 0 else 0.0
    
    return {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tasks': total_tasks,
            'tasks_with_references': tasks_with_any_references,
            'tasks_100_percent_valid': tasks_with_valid_references,
            'task_validity_ratio': task_validity_ratio,
            'total_actions': total_actions,
            'total_id_references': total_id_references,
            'valid_references': valid_references,
            'invalid_references': invalid_references,
            'reference_ratio': reference_ratio,
            'order_user_association_checks': total_order_user_checks,
            'valid_order_user_associations': valid_order_user_associations,
            'invalid_order_user_associations': len(invalid_order_user_associations),
            'order_user_match_ratio': order_user_match_ratio
        },
        'details': {
            'user_id': {
                'total': total_user_ids,
                'valid': valid_user_ids,
                'invalid': len(invalid_user_ids),
                'valid_ratio': (valid_user_ids / total_user_ids * 100) if total_user_ids > 0 else 0.0,
                'invalid_samples': invalid_user_ids[:10]
            },
            'order_id': {
                'total': total_order_ids,
                'valid': valid_order_ids,
                'invalid': len(invalid_order_ids),
                'valid_ratio': (valid_order_ids / total_order_ids * 100) if total_order_ids > 0 else 0.0,
                'invalid_samples': invalid_order_ids[:10]
            },
            'product_id': {
                'total': total_product_ids,
                'valid': valid_product_ids,
                'invalid': len(invalid_product_ids),
                'valid_ratio': (valid_product_ids / total_product_ids * 100) if total_product_ids > 0 else 0.0,
                'invalid_samples': invalid_product_ids[:10]
            },
            'order_user_association': {
                'total_checks': total_order_user_checks,
                'valid': valid_order_user_associations,
                'invalid': len(invalid_order_user_associations),
                'valid_ratio': order_user_match_ratio,
                'invalid_samples': invalid_order_user_associations[:10]
            }
        },
        'environment_data': {
            'total_users': len(users),
            'total_orders': len(orders),
            'total_products': len(products)
        }
    }


def print_report(results):
    """Print a formatted report of the analysis results."""
    print('\n' + '='*80)
    print('REAL DATA REFERENCE ANALYSIS REPORT')
    print('='*80)
    
    summary = results.get('summary', {})
    details = results.get('details', {})
    env_data = results.get('environment_data', {})
    
    print(f'\n📊 Overall Statistics:')
    print(f'   Total Tasks: {summary.get("total_tasks", 0)}')
    print(f'   Tasks with ID References: {summary.get("tasks_with_references", 0)}')
    print(f'   Tasks with 100% Valid References: {summary.get("tasks_100_percent_valid", 0)}')
    print(f'   Task Validity Ratio: {summary.get("task_validity_ratio", 0):.2f}%')
    print(f'   Total Actions: {summary.get("total_actions", 0)}')
    
    print(f'\n🎯 ID Reference Summary:')
    print(f'   Total ID References: {summary.get("total_id_references", 0)}')
    print(f'   ✅ Valid References: {summary.get("valid_references", 0)}')
    print(f'   ❌ Invalid References: {summary.get("invalid_references", 0)}')
    print(f'   📈 Reference Ratio: {summary.get("reference_ratio", 0):.2f}%')
    
    print(f'\n🔗 Order-User Association Validation:')
    print(f'   Total Order-User Checks: {summary.get("order_user_association_checks", 0)}')
    print(f'   ✅ Valid Associations: {summary.get("valid_order_user_associations", 0)}')
    print(f'   ❌ Invalid Associations: {summary.get("invalid_order_user_associations", 0)}')
    print(f'   📈 Match Ratio: {summary.get("order_user_match_ratio", 0):.2f}%')
    
    print(f'\n🔍 ID Type Breakdown:')
    for id_type in ['user_id', 'order_id', 'product_id']:
        id_stats = details.get(id_type, {})
        if id_stats.get('total', 0) > 0:
            print(f'\n   {id_type.upper()}:')
            print(f'      Total: {id_stats.get("total", 0)}')
            print(f'      Valid: {id_stats.get("valid", 0)} ({id_stats.get("valid_ratio", 0):.1f}%)')
            print(f'      Invalid: {id_stats.get("invalid", 0)}')
            
            invalid_samples = id_stats.get('invalid_samples', [])
            if invalid_samples:
                print(f'      Invalid Samples (showing first {min(3, len(invalid_samples))}):')
                for sample in invalid_samples[:3]:
                    task_idx = sample.get('task_index', '?')
                    id_value = sample.get(id_type, sample.get('order_id', sample.get('product_id', '?')))
                    action = sample.get('action', '?')
                    preview = sample.get('query_preview', '')[:60]
                    print(f'        • Task {task_idx}: {id_type}={id_value} in action={action}')
                    if preview:
                        print(f'          Query: "{preview}..."')
    
    # Show order-user association details
    assoc_stats = details.get('order_user_association', {})
    if assoc_stats.get('invalid', 0) > 0:
        print(f'\n⚠️  ORDER-USER ASSOCIATION MISMATCHES:')
        invalid_assoc = assoc_stats.get('invalid_samples', [])
        for sample in invalid_assoc[:5]:
            task_idx = sample.get('task_index', '?')
            order_id = sample.get('order_id', '?')
            order_owner = sample.get('order_owner', '?')
            canonical_user = sample.get('canonical_user_id', '?')
            action = sample.get('action', '?')
            preview = sample.get('query_preview', '')[:60]
            print(f'      • Task {task_idx}: order {order_id} belongs to {order_owner}')
            print(f'        but task uses user_id={canonical_user} in action={action}')
            if preview:
                print(f'        Query: "{preview}..."')
    
    print(f'\n📚 Environment Data:')
    print(f'   Total Users: {env_data.get("total_users", 0)}')
    print(f'   Total Orders: {env_data.get("total_orders", 0)}')
    print(f'   Total Products: {env_data.get("total_products", 0)}')
    
    print('\n' + '='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze real data references in generated tasks'
    )
    parser.add_argument(
        'task_file',
        type=str,
        help='Path to the task JSON file to analyze'
    )
    parser.add_argument(
        '--envs-path',
        type=str,
        default='envs/retail',
        help='Path to the retail environment directory (default: envs/retail)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Check if task file exists
    task_path = Path(args.task_file)
    if not task_path.exists():
        print(f'Error: Task file not found: {args.task_file}')
        return 1
    
    # Load tasks
    print(f'Loading tasks from: {args.task_file}')
    try:
        with open(task_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
            
        # Handle different formats
        if isinstance(tasks, dict) and 'tasks' in tasks:
            tasks = tasks['tasks']
        elif not isinstance(tasks, list):
            tasks = [tasks]
        
        print(f'Loaded {len(tasks)} tasks')
    except Exception as e:
        print(f'Error loading tasks: {e}')
        return 1
    
    # Initialize data reader
    print(f'Loading environment data from: {args.envs_path}')
    try:
        data_reader = TauBenchDataReader(args.envs_path)
    except Exception as e:
        print(f'Error initializing data reader: {e}')
        return 1
    
    # Analyze
    print('Analyzing real data references...')
    results = analyze_real_data_references(tasks, data_reader)
    
    # Print report
    print_report(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_name = task_path.stem
        output_path = Path('generated_tasks') / f'real_data_analysis_{task_name}_{ts}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\n💾 Results saved to: {output_path}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
