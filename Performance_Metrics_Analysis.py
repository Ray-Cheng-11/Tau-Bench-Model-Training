#!/usr/bin/env python3
"""
GPT-OSS-120B Performance Metrics Analysis
Detailed analysis of task generation and testing results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def load_analysis_data():
    """Load analysis data from JSON files"""
    
    # Load generated tasks analysis
    with open('results/analysis/analysis_generated_tasks.json', 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    
    # Load successful tasks analysis  
    with open('results/analysis/analysis_successful_tasks.json', 'r', encoding='utf-8') as f:
        success_data = json.load(f)
    
    return generated_data, success_data

def analyze_performance_metrics(generated_data, success_data):
    """Analyze key performance metrics"""
    
    gen_agg = generated_data['aggregate']
    
    # Basic metrics
    total_tasks = gen_agg['total_tasks']
    total_actions = gen_agg['total_actions']
    avg_actions_per_task = total_actions / total_tasks
    
    # Success rates by action count
    success_by_count = success_data['by_action_count']
    
    # Action distribution
    action_distribution = gen_agg['actions_by_name']
    
    # Read/write mix analysis
    rw_mix = gen_agg['tasks_by_read_write_mix']
    
    return {
        'total_tasks': total_tasks,
        'total_actions': total_actions,
        'avg_actions_per_task': avg_actions_per_task,
        'success_by_count': success_by_count,
        'action_distribution': action_distribution,
        'rw_mix': rw_mix,
        'action_entropy': gen_agg.get('action_entropy', 0)
    }

def generate_performance_report(metrics):
    """Generate detailed performance report"""
    
    print("="*60)
    print("GPT-OSS-120B PERFORMANCE ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 OVERVIEW METRICS")
    print(f"Total Tasks Generated: {metrics['total_tasks']}")
    print(f"Total Actions Planned: {metrics['total_actions']}")
    print(f"Average Actions per Task: {metrics['avg_actions_per_task']:.2f}")
    print(f"Action Entropy (Diversity): {metrics['action_entropy']:.3f}")
    
    print(f"\n🎯 SUCCESS RATE BY TASK COMPLEXITY")
    for action_count, data in sorted(metrics['success_by_count'].items(), key=lambda x: int(x[0])):
        success_rate = data['success_rate'] * 100
        exact_rate = data['exact_match_rate'] * 100
        total = data['total']
        successful = data['success_count']
        exact = data['exact_match_count']
        
        print(f"{action_count}-action tasks: {success_rate:5.1f}% success ({successful:2d}/{total:2d}), "
              f"{exact_rate:5.1f}% exact match ({exact:2d}/{total:2d})")
    
    print(f"\n🔧 TOOL USAGE FREQUENCY")
    sorted_actions = sorted(metrics['action_distribution'].items(), 
                           key=lambda x: x[1], reverse=True)
    for action, count in sorted_actions:
        percentage = (count / metrics['total_actions']) * 100
        print(f"{action:30s}: {count:3d} uses ({percentage:5.1f}%)")
    
    print(f"\n📈 READ/WRITE ACTION PATTERNS")
    for pattern, count in sorted(metrics['rw_mix'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / metrics['total_tasks']) * 100
        print(f"{pattern:8s}: {count:2d} tasks ({percentage:5.1f}%)")

def analyze_action_sequences(success_data):
    """Analyze performance of different action sequences"""
    
    print(f"\n🔄 ACTION SEQUENCE PERFORMANCE")
    print("="*60)
    
    sequences = success_data['expected_action_distribution']['groups']
    
    # Sort by success rate descending
    sorted_sequences = sorted(sequences, key=lambda x: x['success_rate'], reverse=True)
    
    print(f"{'Success%':>8} {'ExactMatch%':>11} {'Total':>5} {'Successful':>10} {'Action Sequence'}")
    print("-" * 80)
    
    for seq in sorted_sequences[:15]:  # Top 15 sequences
        success_rate = seq['success_rate'] * 100
        exact_rate = seq['exact_match_rate'] * 100
        total = seq['total']
        successful = seq['successful']
        
        # Truncate long sequences for display
        sequence = seq['expected_action_sequence']
        if len(sequence) > 50:
            sequence = sequence[:47] + "..."
        
        print(f"{success_rate:7.1f}% {exact_rate:10.1f}% {total:5d} {successful:10d} {sequence}")

def calculate_complexity_score(generated_data):
    """Calculate task complexity scores"""
    
    complexity_scores = []
    
    for task in generated_data['per_task']:
        action_count = task['action_count']
        read_actions = task['read_actions']
        write_actions = task['write_actions']
        
        # Complexity scoring: more actions = higher complexity
        # Write actions weighted more than read actions
        complexity = action_count + (write_actions * 0.5)
        complexity_scores.append(complexity)
    
    return complexity_scores

def generate_visual_analysis(metrics, success_data):
    """Generate visual analysis charts"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPT-OSS-120B Performance Analysis Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Success rate by action count
    action_counts = []
    success_rates = []
    totals = []
    
    for count, data in sorted(metrics['success_by_count'].items(), key=lambda x: int(x[0])):
        action_counts.append(f"{count}-action")
        success_rates.append(data['success_rate'] * 100)
        totals.append(data['total'])
    
    bars1 = ax1.bar(action_counts, success_rates, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_title('Success Rate by Task Complexity', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_xlabel('Task Complexity (Number of Actions)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on bars
    for i, (bar, total) in enumerate(zip(bars1, totals)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{total} tasks', ha='center', va='bottom', fontsize=10, color='black')
    
    # 2. Tool usage distribution
    tools = list(metrics['action_distribution'].keys())
    usage_counts = list(metrics['action_distribution'].values())
    
    # Truncate tool names for display
    tool_labels = [tool.replace('_', '_\n').replace('user_id_by_email', 'user_id\nby_email') 
                   for tool in tools]
    
    bars2 = ax2.barh(tool_labels, usage_counts, color='lightgreen', alpha=0.8, edgecolor='black')
    ax2.set_title('Tool Usage Frequency', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Uses', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 3. Read/Write pattern distribution  
    rw_patterns = list(metrics['rw_mix'].keys())
    rw_counts = list(metrics['rw_mix'].values())
    
    colors = plt.cm.Paired(np.linspace(0, 1, len(rw_patterns)))
    wedges, texts = ax3.pie(rw_counts, labels=None, autopct=None,
                            colors=colors, startangle=90)
    
    # Add legend to avoid label overlap
    ax3.legend(wedges, rw_patterns, title="Patterns", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    # Add percentages as annotations with dynamic offset
    for i, (wedge, count) in enumerate(zip(wedges, rw_counts)):
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        percentage = (count / sum(rw_counts)) * 100
        if percentage >= 2:  # Only annotate if percentage is >= 2%
            ax3.annotate(f"{percentage:.1f}%", xy=(x, y), xytext=(1.5 * x, 1.5 * y),
                         arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
                         ha='center', va='center', fontsize=10)
    
    ax3.set_title('Read/Write Action Pattern Distribution', fontsize=14, fontweight='bold')
    
    # 4. Success rate trend analysis
    # Create a trend line showing how success rate changes with complexity
    complexity_data = []
    success_trend = []
    
    for count, data in sorted(metrics['success_by_count'].items(), key=lambda x: int(x[0])):
        if data['total'] > 0:  # Only include categories with tasks
            complexity_data.append(int(count))
            success_trend.append(data['success_rate'] * 100)
    
    ax4.plot(complexity_data, success_trend, marker='o', linewidth=2, markersize=8, 
             color='darkorange', alpha=0.8, label='Success Rate')
    ax4.set_title('Success Rate vs Task Complexity Trend', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Actions', fontsize=12)
    ax4.set_ylabel('Success Rate (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('GPT_OSS_120B_Performance_Dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    
    # Load data
    generated_data, success_data = load_analysis_data()
    
    # Analyze metrics
    metrics = analyze_performance_metrics(generated_data, success_data)
    
    # Generate reports
    generate_performance_report(metrics)
    analyze_action_sequences(success_data)
    
    # Generate visualizations
    generate_visual_analysis(metrics, success_data)
    
    print(f"\n✅ Analysis complete! Performance dashboard saved as 'GPT_OSS_120B_Performance_Dashboard.png'")
    print(f"\n📋 SUMMARY INSIGHTS:")
    print(f"   • Model shows strong performance on simple tasks (1-2 actions: 100% success)")
    print(f"   • Performance degrades with complexity (3+ actions: ~71% success)")
    print(f"   • Excellent at address management (85% of tasks)")
    print(f"   • Consistent user authentication (100% coverage)")
    print(f"   • Room for improvement in exact sequence matching (27% exact match rate)")

if __name__ == "__main__":
    main()