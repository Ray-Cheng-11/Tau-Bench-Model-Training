import json
from pathlib import Path


def find_specific_tasks():
    """Find and display specific failed tasks from the sample tasks."""
    
    # The task IDs we're looking for
    target_task_ids = [
        "task_005", "task_006", "task_016", "task_027", "task_030", 
        "task_033", "task_051", "task_055", "task_060", "task_064", 
        "task_065", "task_072", "task_077", "task_082", "task_090"
    ]
    
    # Load the sample tasks
    sample_file = Path('generated_tasks/Sampled_Tasks.json')
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        return
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_tasks = json.load(f)
    
    print(f"Looking for {len(target_task_ids)} specific failed tasks in {len(sample_tasks)} total tasks:\n")
    
    found_tasks = {}
    
    # Find each target task
    for target_id in target_task_ids:
        # Extract task number (e.g., "task_005" -> 5)
        task_num = int(target_id.split('_')[1])
        task_index = task_num - 1  # Convert to 0-based index
        
        if 0 <= task_index < len(sample_tasks):
            task = sample_tasks[task_index]
            found_tasks[target_id] = task
            
            print(f"=== {target_id.upper()} ===")
            print(f"Question: {task['q'][:100]}{'...' if len(task['q']) > 100 else ''}")
            print(f"Expected Actions ({len(task['agt'])}):")
            for i, action in enumerate(task['agt'], 1):
                print(f"  {i}. {action['name']}")
            print(f"Expected Output: {task['ogt'][0][:80] if task['ogt'] else 'No output specified'}{'...' if task['ogt'] and len(task['ogt'][0]) > 80 else ''}")
            print()
        else:
            print(f"ERROR: {target_id} - Index {task_index} out of range (total tasks: {len(sample_tasks)})")
    
    # Save the found tasks to a separate file
    output_file = Path('filtered_samples/specific_failed_tasks.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(found_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"Found and saved {len(found_tasks)} specific failed tasks to: {output_file}")
    
    # Create a summary analysis
    action_patterns = {}
    question_lengths = []
    
    for task_id, task in found_tasks.items():
        # Analyze action patterns
        actions = [action['name'] for action in task['agt']]
        action_sequence = ' -> '.join(actions)
        action_patterns[action_sequence] = action_patterns.get(action_sequence, 0) + 1
        
        # Analyze question length
        question_lengths.append(len(task['q']))
    
    print("\n=== ANALYSIS OF FAILED TASKS ===")
    print(f"Average question length: {sum(question_lengths) / len(question_lengths):.1f} characters")
    print(f"Question length range: {min(question_lengths)} - {max(question_lengths)} characters")
    
    print("\nMost common action patterns in failed tasks:")
    for pattern, count in sorted(action_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count}x: {pattern}")


if __name__ == '__main__':
    find_specific_tasks()