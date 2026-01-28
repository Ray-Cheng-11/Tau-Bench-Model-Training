import json
from pathlib import Path
from typing import Dict, List, Set
from analysis_helpers import load_test_results


# Use centralized load_test_results from analysis_helpers to avoid duplication


def get_failed_task_ids(results: List[Dict]) -> Set[str]:
    """Extract task IDs that failed from test results."""
    failed_task_ids = set()
    for entry in results:
        if not entry.get('success', False):
            task_id = entry.get('task_id')
            if task_id:
                failed_task_ids.add(task_id)
    return failed_task_ids


def filter_sample_tasks(sample_tasks: List[Dict], failed_task_ids: Set[str]) -> tuple[List[Dict], List[Dict]]:
    """Filter sample tasks into successful and unsuccessful lists."""
    successful_tasks = []
    unsuccessful_tasks = []
    
    for i, task in enumerate(sample_tasks):
        # Generate task_id in the same format as used in test results
        task_id = f"task_{i+1:03d}"  # task_001, task_002, etc.
        
        if task_id not in failed_task_ids:
            successful_tasks.append(task)
        else:
            unsuccessful_tasks.append(task)
            print(f"Found unsuccessful task {task_id}")
    
    print(f"Successful tasks: {len(successful_tasks)}")
    print(f"Unsuccessful tasks: {len(unsuccessful_tasks)}")
    print(f"Total tasks: {len(sample_tasks)}")
    
    return successful_tasks, unsuccessful_tasks


def filter_sample_file(input_path: Path, output_dir: Path, failed_task_ids: Set[str]) -> None:
    """Filter a sample file and save both successful and unsuccessful tasks."""
    print(f"\nProcessing {input_path.name}...")
    
    # Load the sample file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check the structure and filter accordingly
    if isinstance(data, list):
        # If it's directly a list of tasks
        successful_tasks, unsuccessful_tasks = filter_sample_tasks(data, failed_task_ids)
        successful_data = successful_tasks
        unsuccessful_data = unsuccessful_tasks
    elif isinstance(data, dict):
        # If it's a dictionary containing tasks
        successful_data = data.copy()
        unsuccessful_data = data.copy()
        
        # Look for various possible keys that might contain tasks
        possible_task_keys = ['tasks', 'samples', 'data', 'entries']
        task_key_found = None
        
        for key in possible_task_keys:
            if key in data and isinstance(data[key], list):
                task_key_found = key
                break
        
        if task_key_found:
            print(f"Found tasks under key: '{task_key_found}'")
            successful_tasks, unsuccessful_tasks = filter_sample_tasks(data[task_key_found], failed_task_ids)
            successful_data[task_key_found] = successful_tasks
            unsuccessful_data[task_key_found] = unsuccessful_tasks
        else:
            # Check if the root level contains task-like objects
            # This handles the case where the JSON has a structure like the retail_unified_sample.json
            # which contains domain info and other metadata, but no direct task list
            print("No direct task list found in this file structure.")
            print("This might be a configuration file rather than a sample task file.")
            return
    else:
        print(f"Unexpected data structure in {input_path.name}")
        return
    
    # Save both successful and unsuccessful data
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save successful tasks
    successful_output_path = output_dir / f"successful_{input_path.name}"
    with open(successful_output_path, 'w', encoding='utf-8') as f:
        json.dump(successful_data, f, indent=2, ensure_ascii=False)
    print(f"Saved successful tasks to: {successful_output_path}")
    
    # Save unsuccessful tasks
    unsuccessful_output_path = output_dir / f"unsuccessful_{input_path.name}"
    with open(unsuccessful_output_path, 'w', encoding='utf-8') as f:
        json.dump(unsuccessful_data, f, indent=2, ensure_ascii=False)
    print(f"Saved unsuccessful tasks to: {unsuccessful_output_path}")


def main():
    root = Path('.')
    
    # Find the latest test results
    test_results = sorted(root.glob('results/test_results_*.json'), 
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not test_results:
        print("No test results found in 'results' directory")
        return
    
    test_path = test_results[0]
    print(f"Using latest test results: {test_path}")
    
    # Load test results and extract failed task IDs
    results = load_test_results(test_path)
    failed_task_ids = get_failed_task_ids(results)
    
    print(f"Found {len(failed_task_ids)} failed tasks:")
    for task_id in sorted(failed_task_ids):
        print(f"  - {task_id}")
    
    # Find sample task files to filter
    sample_files = [
        root / 'generated_tasks' / 'Sampled_Tasks.json',
    ]
    
    # Create output directory
    output_dir = root / 'filtered_samples'
    output_dir.mkdir(exist_ok=True)
    
    # Filter each sample file
    for sample_file in sample_files:
        if sample_file.exists():
            filter_sample_file(sample_file, output_dir, failed_task_ids)
        else:
            print(f"Sample file not found: {sample_file}")
    
    # Create a summary report
    summary = {
        'test_results_file': str(test_path),
        'total_tasks_tested': len(results),
        'failed_tasks_count': len(failed_task_ids),
        'failed_task_ids': sorted(failed_task_ids),
        'success_rate': (len(results) - len(failed_task_ids)) / len(results) if results else 0,
        'successful_tasks_count': len(results) - len(failed_task_ids),
        'filtered_files': {
            'successful': [f"successful_{f.name}" for f in sample_files if f.exists()],
            'unsuccessful': [f"unsuccessful_{f.name}" for f in sample_files if f.exists()]
        }
    }
    
    summary_path = output_dir / 'filtering_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nFiltering Summary:")
    print(f"  Total tasks tested: {summary['total_tasks_tested']}")
    print(f"  Successful tasks: {summary['successful_tasks_count']}")
    print(f"  Failed tasks: {summary['failed_tasks_count']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()