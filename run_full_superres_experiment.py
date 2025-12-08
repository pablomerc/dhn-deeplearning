#!/usr/bin/env python
"""
Master script to reproduce Figure 14 and extend to two-body problem.

This script runs the complete super-resolution experiment pipeline:
1. Generate data for all systems (single pendulum, double pendulum, two-body)
2. Train DHN and CNN baseline models
3. Extract (autodecode) for same-init and diff-init segments
4. Generate Figure 14-style plots

Usage:
    python run_full_superres_experiment.py --phase all
    python run_full_superres_experiment.py --phase data
    python run_full_superres_experiment.py --phase train
    python run_full_superres_experiment.py --phase extract
    python run_full_superres_experiment.py --phase plot
"""

import os
import sys
import argparse
import subprocess
import time
import pickle
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_command(cmd, description=""):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
    return result.returncode


def phase_data():
    """Phase 1: Generate data for all systems."""
    print("\n" + "="*80)
    print("PHASE 1: DATA GENERATION")
    print("="*80)
    
    from data_gen.simulators import get_simulator
    import ml_collections
    
    systems = {
        'single_pendulum': {
            'constants': {
                'g': 9.81,
                'm': (0.5, 1.5),
                'l': (0.5, 1.5),
            }
        },
        'double_pendulum': {
            'constants': {
                'g': 9.81,
                'm1': 1.0,
                'm2': 1.0,
                'l1': 1.0,
                'l2': (0.5, 1.5),
            }
        },
        'two_body': {
            'constants': {
                'G': 1.0,
                'm1': (0.5, 2.0),
                'm2': (0.5, 2.0),
                'r1': (0.5, 1.0),
                'r2': (1.1, 1.5),
            }
        },
    }
    
    for system_name, system_config in systems.items():
        print(f"\n--- Generating data for {system_name} ---")
        
        for split, num_data, seed in [('train', 1000, 0), ('test', 200, 1)]:
            save_dir = os.path.join(PROJECT_ROOT, 'data', system_name, split)
            os.makedirs(save_dir, exist_ok=True)
            
            # Check if data already exists
            data_path = os.path.join(save_dir, 'data.pkl')
            if os.path.exists(data_path):
                print(f"  {split} data already exists at {data_path}, skipping...")
                continue
            
            print(f"  Generating {split} data ({num_data} trajectories)...")
            
            np.random.seed(seed)
            
            # Build config
            config = ml_collections.ConfigDict()
            config.simulator = system_name
            config.save_dir = save_dir
            config.num_data = num_data
            config.num_vis = 0
            config.t_span = (0, 10)
            config.num_timesteps = 1025
            config.constants = ml_collections.ConfigDict(system_config['constants'])
            
            trajectories = []
            for i_data in range(num_data):
                simulator = get_simulator(config)
                traj = simulator.sample_trajectory()
                traj['idx'] = i_data
                trajectories.append(traj)
                
                if i_data % 100 == 0:
                    print(f"    Generated {i_data}/{num_data} trajectories...")
            
            with open(data_path, 'wb') as fp:
                pickle.dump(trajectories, fp)
            
            print(f"  Saved to {data_path}")
    
    print("\nData generation complete!")


def phase_train(num_epochs=200):
    """Phase 2: Train all models."""
    print("\n" + "="*80)
    print("PHASE 2: MODEL TRAINING")
    print("="*80)
    
    experiments = [
        # Single pendulum
        ('superres', 'sinpend_4x', 'DHN Single Pendulum'),
        ('superres', 'sinpend_baseline_cnn_4x', 'CNN Single Pendulum'),
        # Double pendulum
        ('superres', 'doupend_4x', 'DHN Double Pendulum'),
        ('superres', 'doupend_baseline_cnn_4x', 'CNN Double Pendulum'),
        # Two-body
        ('superres', 'two_body_4x', 'DHN Two-Body'),
        ('superres', 'two_body_baseline_cnn_4x', 'CNN Two-Body'),
    ]
    
    for exp_class, exp_name, description in experiments:
        result_dir = os.path.join(PROJECT_ROOT, 'results', exp_class, exp_name)
        
        # Check if already trained
        ckpt_path = os.path.join(result_dir, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            print(f"\n{description}: Checkpoint exists, skipping training...")
            continue
        
        cmd = (
            f'python main.py '
            f'--config=configs/{exp_class}/{exp_name}.py '
            f'--mode=train '
            f'--config.workdir={result_dir} '
            f'--config.optim.num_epochs={num_epochs}'
        )
        
        run_command(cmd, f"Training {description}")


def phase_extract(num_epochs=1000, lr=1e-2):
    """Phase 3: Run extraction (autodecoding) for all models."""
    print("\n" + "="*80)
    print("PHASE 3: EXTRACTION (AUTODECODING)")
    print("="*80)
    
    experiments = [
        ('superres', 'sinpend_4x'),
        ('superres', 'sinpend_baseline_cnn_4x'),
        ('superres', 'doupend_4x'),
        ('superres', 'doupend_baseline_cnn_4x'),
        ('superres', 'two_body_4x'),
        ('superres', 'two_body_baseline_cnn_4x'),
    ]
    
    for exp_class, exp_name in experiments:
        result_dir = os.path.join(PROJECT_ROOT, 'results', exp_class, exp_name)
        
        # Check if model checkpoint exists
        ckpt_files = [f for f in os.listdir(result_dir) if f.startswith('checkpoint_ep')]
        if not ckpt_files:
            print(f"\nNo checkpoint for {exp_name}, skipping extraction...")
            continue
        
        # Same-init extraction (in-distribution)
        extract_dir = os.path.join(result_dir, 'extract')
        if not os.path.exists(os.path.join(extract_dir, 'checkpoint.pth')):
            cmd = (
                f'python main.py '
                f'--config=configs/{exp_class}/{exp_name}.py '
                f'--mode=extract '
                f'--config.workdir={result_dir} '
                f'--config.model.num_embeddings=200 '
                f'--config.logging.num_eval_batches=1000000 '
                f'--config.data.batch_size=100 '
                f'--config.optim.num_epochs={num_epochs} '
                f'--config.optim.lr={lr}'
            )
            run_command(cmd, f"Extracting {exp_name} (same-init)")
        else:
            print(f"\n{exp_name} same-init extraction exists, skipping...")
        
        # Diff-init extraction (out-of-distribution)
        extract_ood_dir = os.path.join(result_dir, 'extract_ood')
        if not os.path.exists(os.path.join(extract_ood_dir, 'checkpoint.pth')):
            cmd = (
                f'python main.py '
                f'--config=configs/{exp_class}/{exp_name}.py '
                f'--mode=extract '
                f'--config.workdir={result_dir} '
                f'--config.model.num_embeddings=200 '
                f'--config.logging.num_eval_batches=1000000 '
                f'--config.data.batch_size=100 '
                f'--config.optim.num_epochs={num_epochs} '
                f'--config.optim.lr={lr} '
                f'--config.model.train_step_span="(512,1025)" '
                f'--work_subdir=extract_ood'
            )
            run_command(cmd, f"Extracting {exp_name} (diff-init / OOD)")
        else:
            print(f"\n{exp_name} diff-init extraction exists, skipping...")


def phase_plot():
    """Phase 4: Generate Figure 14-style plots."""
    print("\n" + "="*80)
    print("PHASE 4: GENERATING PLOTS")
    print("="*80)
    
    from visualization_scripts.plot_superres_results import (
        collect_results, plot_figure14_style, print_results_table
    )
    
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'superres')
    output_path = os.path.join(PROJECT_ROOT, 'visualization_scripts', 'figure14_reproduction.png')
    
    print(f"Collecting results from: {results_dir}")
    results = collect_results(results_dir)
    
    print_results_table(results, scale_factor=100)
    plot_figure14_style(results, output_path, scale_factor=100)
    
    print(f"\nFigure saved to: {output_path}")
    print(f"PDF also saved to: {output_path.replace('.png', '.pdf')}")


def main():
    parser = argparse.ArgumentParser(description='Run Figure 14 reproduction experiment')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'data', 'train', 'extract', 'plot'],
                        help='Which phase to run')
    parser.add_argument('--train_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--extract_epochs', type=int, default=1000,
                        help='Number of extraction (autodecoding) epochs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FIGURE 14 REPRODUCTION + TWO-BODY EXTENSION")
    print("="*80)
    print(f"Phase: {args.phase}")
    print(f"Training epochs: {args.train_epochs}")
    print(f"Extraction epochs: {args.extract_epochs}")
    
    start_time = time.time()
    
    if args.phase in ['all', 'data']:
        phase_data()
    
    if args.phase in ['all', 'train']:
        phase_train(num_epochs=args.train_epochs)
    
    if args.phase in ['all', 'extract']:
        phase_extract(num_epochs=args.extract_epochs)
    
    if args.phase in ['all', 'plot']:
        phase_plot()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print("="*80)


if __name__ == '__main__':
    main()


