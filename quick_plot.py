"""Quick plotting script that evaluates models directly instead of parsing TensorBoard."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input_pipeline import create_dataloader
from models_hamiltonian import get_model_hamiltoinian


def evaluate_model(config_path, workdir, num_batches=10, use_extract_dir=None, is_ood=False):
    """Load a trained model and evaluate MSE on test data.
    
    Args:
        use_extract_dir: None = use root checkpoint, 'extract' or 'extract_ood' = use that subdir
        is_ood: If True, use OOD time range (512-1025) instead of training range (0-513)
    """
    # Import config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    
    config.workdir = workdir
    
    # For OOD evaluation, use the second half of trajectories (different time range)
    if is_ood:
        config.model.train_step_span = (512, 1025)
        print(f"    Using OOD time range: {config.model.train_step_span}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Create model
    model = get_model_hamiltoinian(config.model, dtype=dtype)
    model = model.to(dtype).to(device)
    
    # Determine checkpoint path
    if use_extract_dir:
        extract_dir = os.path.join(workdir, use_extract_dir)
        if os.path.exists(extract_dir):
            ckpt_path = os.path.join(extract_dir, 'checkpoint.pth')
        else:
            print(f"    WARNING: {use_extract_dir}/ not found, using root checkpoint")
            ckpt_path = os.path.join(workdir, 'checkpoint.pth')
    else:
        ckpt_path = os.path.join(workdir, 'checkpoint.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found at {ckpt_path}")
        return None
    
    print(f"    Loading: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataloader
    test_loader = create_dataloader(config.data, split='test')
    
    # Evaluate
    mse_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= num_batches:
                break
            
            # Move data to device
            for k in data:
                if isinstance(data[k], torch.Tensor) and k != 'idx':
                    data[k] = data[k].to(dtype).to(device)
            data['idx'] = data['idx'].to(torch.long).to(device)
            
            # Get predictions
            try:
                dict_losses, dict_vals = model.inference(data)
                # DHN returns: loss_eval/q_1, loss_eval/q_10 (use q_10 for full denoising)
                # CNN returns: loss_eval/q
                if 'loss_eval/q_10' in dict_losses:
                    mse_list.append(dict_losses['loss_eval/q_10'])
                elif 'loss_eval/q' in dict_losses:
                    mse_list.append(dict_losses['loss_eval/q'])
                elif 'loss_eval/q_1' in dict_losses:
                    mse_list.append(dict_losses['loss_eval/q_1'])
                else:
                    # Fallback to first loss value
                    print(f"    Available keys: {list(dict_losses.keys())}")
                    mse_list.append(list(dict_losses.values())[0])
            except Exception as e:
                print(f"  Error during inference: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    if mse_list:
        return np.mean(mse_list)
    return None


def main():
    results_dir = 'results/superres'
    
    # Define experiments
    experiments = {
        'Single Pendulum': {
            'dhn': ('configs/superres/sinpend_4x.py', 'sinpend_4x'),
            'cnn': ('configs/superres/sinpend_baseline_cnn_4x.py', 'sinpend_baseline_cnn_4x'),
        },
        'Double Pendulum': {
            'dhn': ('configs/superres/doupend_4x.py', 'doupend_4x'),
            'cnn': ('configs/superres/doupend_baseline_cnn_4x.py', 'doupend_baseline_cnn_4x'),
        },
        'Two-Body': {
            'dhn': ('configs/superres/two_body_4x.py', 'two_body_4x'),
            'cnn': ('configs/superres/two_body_baseline_cnn_4x.py', 'two_body_baseline_cnn_4x'),
        },
    }
    
    # Collect results for both scenarios
    results_same_init = defaultdict(dict)
    results_ood = defaultdict(dict)
    
    print("="*70)
    print("EVALUATING MODELS")
    print("="*70)
    print("\nScenario 1: Same-Init (DHN uses extract/, CNN uses root)")
    print("-"*70)
    
    for system_name, models in experiments.items():
        print(f"\n{system_name}:")
        for model_type, (config_path, exp_name) in models.items():
            workdir = os.path.join(results_dir, exp_name)
            if os.path.exists(workdir):
                # DHN: use extract/ (optimized latent codes)
                # CNN: use root checkpoint
                use_extract_dir = 'extract' if model_type == 'dhn' else None
                print(f"  {model_type.upper()}:")
                mse = evaluate_model(config_path, workdir, num_batches=20, use_extract_dir=use_extract_dir)
                if mse is not None:
                    results_same_init[system_name][model_type] = mse
                    print(f"    MSE: {mse:.6f}")
                else:
                    print(f"    Failed to evaluate")
            else:
                print(f"  {workdir} not found")
    
    print("\n" + "="*70)
    print("Scenario 2: OOD / Diff-Init (DHN uses extract_ood/, CNN uses extract_ood/)")
    print("-"*70)
    
    for system_name, models in experiments.items():
        print(f"\n{system_name}:")
        for model_type, (config_path, exp_name) in models.items():
            workdir = os.path.join(results_dir, exp_name)
            extract_ood_dir = os.path.join(workdir, 'extract_ood')
            if os.path.exists(extract_ood_dir):
                print(f"  {model_type.upper()}:")
                mse = evaluate_model(config_path, workdir, num_batches=20, use_extract_dir='extract_ood', is_ood=True)
                if mse is not None:
                    results_ood[system_name][model_type] = mse
                    print(f"    MSE: {mse:.6f}")
                else:
                    print(f"    Failed to evaluate")
            else:
                print(f"  {model_type.upper()}: extract_ood/ not found")
    
    # Create comparison plot
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Same-Init
    ax = axes[0]
    systems = list(results_same_init.keys())
    if systems:
        x = np.arange(len(systems))
        width = 0.35
        
        dhn_values = [results_same_init[s].get('dhn', 0) for s in systems]
        cnn_values = [results_same_init[s].get('cnn', 0) for s in systems]
        
        bars1 = ax.bar(x - width/2, dhn_values, width, label='DHN', color='#2ecc71')
        bars2 = ax.bar(x + width/2, cnn_values, width, label='CNN', color='#e74c3c')
        
        ax.set_xlabel('Physical System', fontsize=12)
        ax.set_ylabel('MSE (Test)', fontsize=12)
        ax.set_title('Same-Init (In-Distribution)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=15)
        ax.legend()
        ax.set_yscale('log')
        
        for bar, val in zip(bars1, dhn_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', 
                        ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, cnn_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', 
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 2: OOD
    ax = axes[1]
    systems_ood = list(results_ood.keys())
    if systems_ood:
        x = np.arange(len(systems_ood))
        width = 0.35
        
        dhn_values = [results_ood[s].get('dhn', 0) for s in systems_ood]
        cnn_values = [results_ood[s].get('cnn', 0) for s in systems_ood]
        
        bars1 = ax.bar(x - width/2, dhn_values, width, label='DHN', color='#2ecc71')
        bars2 = ax.bar(x + width/2, cnn_values, width, label='CNN', color='#e74c3c')
        
        ax.set_xlabel('Physical System', fontsize=12)
        ax.set_ylabel('MSE (Test)', fontsize=12)
        ax.set_title('Diff-Init (Out-of-Distribution)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(systems_ood, rotation=15)
        ax.legend()
        ax.set_yscale('log')
        
        for bar, val in zip(bars1, dhn_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', 
                        ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, cnn_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', 
                        ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No OOD data available\n(extract_ood/ not found)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Diff-Init (Out-of-Distribution)', fontsize=14)
    
    plt.suptitle('4× Trajectory Super-Resolution: DHN vs CNN Baseline', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = 'figure14_reproduction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {os.path.abspath(output_path)}")
    
    pdf_path = 'figure14_reproduction.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {os.path.abspath(pdf_path)}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\n--- Same-Init (In-Distribution) ---")
    for system in results_same_init.keys():
        dhn_mse = results_same_init[system].get('dhn', None)
        cnn_mse = results_same_init[system].get('cnn', None)
        print(f"\n{system}:")
        print(f"  DHN: {dhn_mse:.6f}" if dhn_mse else "  DHN: N/A")
        print(f"  CNN: {cnn_mse:.6f}" if cnn_mse else "  CNN: N/A")
        if dhn_mse and cnn_mse:
            if cnn_mse > dhn_mse:
                improvement = (cnn_mse - dhn_mse) / cnn_mse * 100
                print(f"  → DHN better by {improvement:.1f}%")
            else:
                degradation = (dhn_mse - cnn_mse) / dhn_mse * 100
                print(f"  → CNN better by {degradation:.1f}%")
    
    if results_ood:
        print("\n--- Diff-Init (Out-of-Distribution) ---")
        for system in results_ood.keys():
            dhn_mse = results_ood[system].get('dhn', None)
            cnn_mse = results_ood[system].get('cnn', None)
            print(f"\n{system}:")
            print(f"  DHN: {dhn_mse:.6f}" if dhn_mse else "  DHN: N/A")
            print(f"  CNN: {cnn_mse:.6f}" if cnn_mse else "  CNN: N/A")
            if dhn_mse and cnn_mse:
                if cnn_mse > dhn_mse:
                    improvement = (cnn_mse - dhn_mse) / cnn_mse * 100
                    print(f"  → DHN better by {improvement:.1f}%")
                else:
                    degradation = (dhn_mse - cnn_mse) / dhn_mse * 100
                    print(f"  → CNN better by {degradation:.1f}%")


if __name__ == '__main__':
    main()


