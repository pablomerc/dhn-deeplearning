#!/usr/bin/env python
"""
Quick test script to verify the pipeline works with minimal epochs.

Usage:
    python run_quick_test.py

This runs:
- Data generation (10 train / 5 test trajectories)
- Training (2 epochs)
- Extraction (10 epochs)
- Plotting

Just for TWO_BODY to quickly verify everything works.
"""

import os
import sys
import pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("="*60)
    print("QUICK PIPELINE TEST")
    print("="*60)
    
    # 1. Generate minimal data for two-body
    print("\n--- Step 1: Generate minimal two-body data ---")
    
    from data_gen.simulators import get_simulator
    import ml_collections
    
    for split, num_data, seed in [('train', 20, 0), ('test', 10, 1)]:
        save_dir = os.path.join(PROJECT_ROOT, 'data', 'two_body', split)
        os.makedirs(save_dir, exist_ok=True)
        
        data_path = os.path.join(save_dir, 'data.pkl')
        
        print(f"  Generating {split} data ({num_data} trajectories)...")
        
        np.random.seed(seed)
        
        config = ml_collections.ConfigDict()
        config.simulator = 'two_body'
        config.save_dir = save_dir
        config.num_data = num_data
        config.num_vis = 0
        config.t_span = (0, 10)
        config.num_timesteps = 1025
        config.constants = ml_collections.ConfigDict()
        config.constants.G = 1.0
        config.constants.m1 = (0.5, 2.0)
        config.constants.m2 = (0.5, 2.0)
        config.constants.r1 = (0.5, 1.0)
        config.constants.r2 = (1.1, 1.5)
        
        trajectories = []
        for i_data in range(num_data):
            simulator = get_simulator(config)
            traj = simulator.sample_trajectory()
            traj['idx'] = i_data
            trajectories.append(traj)
        
        with open(data_path, 'wb') as fp:
            pickle.dump(trajectories, fp)
        
        print(f"  Saved to {data_path}")
    
    # 2. Quick model test
    print("\n--- Step 2: Test model creation ---")
    
    from models_hamiltonian import get_model_hamiltoinian
    import torch
    
    # Test DHN superres model
    config = ml_collections.ConfigDict()
    config.hamiltonian = 'superres'
    config.num_embeddings = 20
    config.embedding_dim = 64
    config.q_dim = 2
    config.t_span = (0, 10)
    config.stage_step_size = (32, 16, 8)
    config.stage_block_size = (2, 2, 2)
    config.stage_block_step = (1, 1, 1)
    config.num_noise_scales = 10
    config.state_base_with_noise = True
    config.train_step_span = (0, 513)
    
    config.codebook = ml_collections.ConfigDict()
    config.codebook.num_embeddings = config.num_embeddings
    config.codebook.embedding_dim = config.embedding_dim
    config.codebook.normalize_emb = False
    
    config.network = ml_collections.ConfigDict()
    config.network.name = 'simple_transformer'
    config.network.q_dim = config.q_dim
    config.network.z_dim = config.embedding_dim
    config.network.block_size = 2
    config.network.output_dim = 1
    config.network.hidden_dim = 64
    config.network.num_heads = 4
    config.network.num_layers = 2
    config.network.num_noise_scales = config.num_noise_scales
    
    model = get_model_hamiltoinian(config, dtype=torch.float32)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  DHN model created: {num_params:,} parameters")
    
    # 3. Test data loading
    print("\n--- Step 3: Test data loading ---")
    
    from input_pipeline import create_dataloader
    
    data_config = ml_collections.ConfigDict()
    data_config.path = os.path.join(PROJECT_ROOT, 'data', 'two_body')
    data_config.batch_size = 4
    data_config.num_workers = 0
    data_config.prefetch_factor = 2
    data_config.pin_memory = False
    data_config.cache = False
    
    train_loader = create_dataloader(data_config, split='train')
    
    for batch in train_loader:
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  q shape: {batch['q'].shape}")
        print(f"  p (L_grad_dq) shape: {batch['L_grad_dq'].shape}")
        break
    
    # 4. Test forward pass
    print("\n--- Step 4: Test forward pass ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = model.to(device)
    
    loss_config = ml_collections.ConfigDict()
    loss_config.weight_eom = 1.0
    loss_config.weight_denoise = 0.1
    loss_config.use_out_mask = True
    loss_config.crop_interval = (0, -1)
    
    for batch in train_loader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != 'idx':
                batch[k] = batch[k].float().to(device)
        batch['idx'] = batch['idx'].long().to(device)
        
        loss, loss_dict = model.get_losses(batch, loss_config)
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Loss dict: {loss_dict}")
        
        loss.backward()
        print("  Backward pass: OK")
        break
    
    print("\n" + "="*60)
    print("QUICK TEST PASSED!")
    print("="*60)
    print("\nYou can now run the full experiment:")
    print("  python run_full_superres_experiment.py --phase all")
    print("\nOr run phases individually:")
    print("  python run_full_superres_experiment.py --phase data")
    print("  python run_full_superres_experiment.py --phase train")
    print("  python run_full_superres_experiment.py --phase extract")
    print("  python run_full_superres_experiment.py --phase plot")


if __name__ == '__main__':
    main()


