"""
Visualization script for Super-Resolution (Trajectory Interpolation) results.
Generates Figure 14-style bar charts comparing DHN vs CNN baseline.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Set style for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 4),
    'figure.dpi': 150,
})

# Color scheme matching paper
COLORS = {
    'ours': '#8B4A6B',      # Dark purple/maroon for DHN
    'cnn': '#D4A574',       # Tan/beige for CNN baseline
}


def get_final_mse_from_tensorboard(log_dir, metric_key='loss_eval/q_10'):
    """Extract final MSE from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Try different possible metric keys
    possible_keys = [metric_key, 'loss_eval/q_1', 'loss_eval/q', 'loss_train/train']
    
    for key in possible_keys:
        if key in ea.scalars.Keys():
            events = ea.scalars.Items(key)
            if events:
                # Return the final value
                return events[-1].value
    
    return None


def collect_results(results_base_dir='results/superres'):
    """Collect MSE results from all experiments."""
    
    systems = {
        'single_pendulum': {'dhn': 'sinpend_4x', 'cnn': 'sinpend_baseline_cnn_4x'},
        'double_pendulum': {'dhn': 'doupend_4x', 'cnn': 'doupend_baseline_cnn_4x'},
        'two_body': {'dhn': 'two_body_4x', 'cnn': 'two_body_baseline_cnn_4x'},
    }
    
    results = {}
    
    for system_name, experiments in systems.items():
        results[system_name] = {
            'dhn_same': None,
            'dhn_diff': None,
            'cnn_same': None,
            'cnn_diff': None,
        }
        
        for method, exp_name in experiments.items():
            exp_dir = os.path.join(results_base_dir, exp_name)
            
            # Same-init results
            extract_dir = os.path.join(exp_dir, 'extract')
            if os.path.exists(extract_dir):
                mse = get_final_mse_from_tensorboard(extract_dir)
                results[system_name][f'{method}_same'] = mse
            
            # Diff-init results (OOD)
            extract_ood_dir = os.path.join(exp_dir, 'extract_ood')
            if os.path.exists(extract_ood_dir):
                mse = get_final_mse_from_tensorboard(extract_ood_dir)
                results[system_name][f'{method}_diff'] = mse
    
    return results


def plot_figure14_style(results, save_path='superres_comparison.png', scale_factor=100):
    """
    Create Figure 14-style bar chart.
    
    Args:
        results: Dict with MSE results
        save_path: Path to save the figure
        scale_factor: Multiply MSE by this factor (paper uses 100)
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    system_titles = {
        'single_pendulum': 'Single Pendulum',
        'double_pendulum': 'Double Pendulum', 
        'two_body': 'Two Body',
    }
    
    systems = ['single_pendulum', 'double_pendulum', 'two_body']
    
    for idx, system in enumerate(systems):
        ax = axes[idx]
        
        data = results.get(system, {})
        
        # Get values (multiply by scale factor)
        dhn_same = (data.get('dhn_same') or 0) * scale_factor
        dhn_diff = (data.get('dhn_diff') or 0) * scale_factor
        cnn_same = (data.get('cnn_same') or 0) * scale_factor
        cnn_diff = (data.get('cnn_diff') or 0) * scale_factor
        
        # Bar positions
        x = np.array([0, 1])
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, [dhn_same, dhn_diff], width, 
                       label='Ours', color=COLORS['ours'])
        bars2 = ax.bar(x + width/2, [cnn_same, cnn_diff], width, 
                       label='CNN', color=COLORS['cnn'])
        
        # Add value labels on bars
        for bar, val in zip(bars1, [dhn_same, dhn_diff]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, [cnn_same, cnn_diff]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Labels and formatting
        ax.set_ylabel(f'MSE (×{scale_factor})')
        ax.set_title(system_titles[system])
        ax.set_xticks(x)
        ax.set_xticklabels(['Same init.', 'Diff. init.'])
        
        if idx == 0:
            ax.legend()
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def print_results_table(results, scale_factor=100):
    """Print results as a formatted table."""
    
    print("\n" + "="*70)
    print("SUPER-RESOLUTION RESULTS (MSE × {})".format(scale_factor))
    print("="*70)
    print(f"{'System':<20} {'Method':<10} {'Same Init':<15} {'Diff Init':<15}")
    print("-"*70)
    
    for system in ['single_pendulum', 'double_pendulum', 'two_body']:
        data = results.get(system, {})
        
        dhn_same = (data.get('dhn_same') or 0) * scale_factor
        dhn_diff = (data.get('dhn_diff') or 0) * scale_factor
        cnn_same = (data.get('cnn_same') or 0) * scale_factor
        cnn_diff = (data.get('cnn_diff') or 0) * scale_factor
        
        print(f"{system:<20} {'DHN':<10} {dhn_same:<15.4f} {dhn_diff:<15.4f}")
        print(f"{'':<20} {'CNN':<10} {cnn_same:<15.4f} {cnn_diff:<15.4f}")
        print("-"*70)
    
    print("="*70 + "\n")


def main():
    """Main function to generate visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot Super-Resolution Results')
    parser.add_argument('--results_dir', type=str, default='results/superres',
                        help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default='visualization_scripts/superres_comparison.png',
                        help='Output path for the figure')
    parser.add_argument('--scale', type=int, default=100,
                        help='Scale factor for MSE (default: 100)')
    
    args = parser.parse_args()
    
    print("Collecting results from:", args.results_dir)
    results = collect_results(args.results_dir)
    
    print_results_table(results, args.scale)
    plot_figure14_style(results, args.output, args.scale)


if __name__ == '__main__':
    main()

