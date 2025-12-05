#!/usr/bin/env python
"""
Quick sanity test for Super-Resolution (Trajectory Interpolation) training.
Verifies that all components work before committing to a full training run.
"""

import os
import sys
import tempfile
import shutil
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Test 1: Testing imports...")
    print("=" * 60)
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import ml_collections
        print("  ✓ ml_collections")
    except ImportError as e:
        print(f"  ✗ ml_collections import failed: {e}")
        return False
    
    try:
        from scipy.integrate import solve_ivp
        print("  ✓ scipy")
    except ImportError as e:
        print(f"  ✗ scipy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError as e:
        print(f"  ✗ matplotlib import failed: {e}")
        return False
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        print("  ✓ tensorboard")
    except ImportError as e:
        print(f"  ✗ tensorboard import failed: {e}")
        return False
    
    print("  All imports successful!\n")
    return True


def test_data_generation():
    """Test that data generation works for all systems."""
    print("=" * 60)
    print("Test 2: Testing data generation...")
    print("=" * 60)
    
    from data_gen.simulators import get_simulator
    
    systems = ['single_pendulum', 'double_pendulum', 'two_body']
    
    for system in systems:
        try:
            # Create minimal config
            import ml_collections
            config = ml_collections.ConfigDict()
            config.simulator = system
            config.save_dir = 'tmp'
            config.num_data = 2
            config.num_vis = 0
            config.t_span = (0, 10)
            config.num_timesteps = 1025
            
            # System-specific constants
            config.constants = ml_collections.ConfigDict()
            if system == 'single_pendulum':
                config.constants.l = (0.5, 1.5)
                config.constants.m = (0.5, 1.5)
                config.constants.g = 9.81
            elif system == 'double_pendulum':
                config.constants.l1 = (0.5, 1.5)
                config.constants.l2 = (0.5, 1.5)
                config.constants.m1 = (0.5, 1.5)
                config.constants.m2 = (0.5, 1.5)
                config.constants.g = 9.81
            elif system == 'two_body':
                config.constants.G = 1.0
                config.constants.m1 = (0.5, 2.0)
                config.constants.m2 = (0.5, 2.0)
                config.constants.r1 = (0.5, 1.0)
                config.constants.r2 = (1.1, 1.5)
            
            simulator = get_simulator(config)
            traj = simulator.sample_trajectory()
            
            # Verify trajectory has required fields
            required_fields = ['time', 'q', 'dq', 'L', 'L_grad_q', 'L_grad_dq']
            for field in required_fields:
                assert field in traj, f"Missing field: {field}"
            
            # Verify shapes
            assert traj['time'].shape[0] == 1025, f"Wrong time shape: {traj['time'].shape}"
            assert traj['q'].shape[0] == 1025, f"Wrong q shape: {traj['q'].shape}"
            
            print(f"  ✓ {system}: q_dim={traj['q'].shape[1]}, timesteps={traj['q'].shape[0]}")
            
        except Exception as e:
            print(f"  ✗ {system} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("  All data generation tests passed!\n")
    return True


def test_model_initialization():
    """Test that models can be initialized."""
    print("=" * 60)
    print("Test 3: Testing model initialization...")
    print("=" * 60)
    
    import torch
    from models_hamiltonian import get_model_hamiltoinian
    
    # Test DHN superres model
    try:
        import ml_collections
        config = ml_collections.ConfigDict()
        config.hamiltonian = 'superres'
        config.num_embeddings = 100
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
        print(f"  ✓ DHN Superres model: {num_params:,} parameters")
        
    except Exception as e:
        print(f"  ✗ DHN Superres model failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test CNN baseline model
    try:
        config = ml_collections.ConfigDict()
        config.hamiltonian = 'baseline_cnn_superres'
        config.num_embeddings = 100
        config.q_dim = 2
        config.t_span = (0, 10)
        config.stage_step_size = (32, 8)
        config.step_size = 8
        config.train_step_span = (0, 513)
        
        config.network = ml_collections.ConfigDict()
        config.network.name = 'baseline_cnn'
        config.network.q_dim = config.q_dim
        config.network.hidden_dim = 64
        config.network.num_stages = 2
        
        model = get_model_hamiltoinian(config, dtype=torch.float32)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ CNN Baseline model: {num_params:,} parameters")
        
    except Exception as e:
        print(f"  ✗ CNN Baseline model failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  All model initialization tests passed!\n")
    return True


def test_forward_pass():
    """Test that forward passes work correctly."""
    print("=" * 60)
    print("Test 4: Testing forward passes...")
    print("=" * 60)
    
    import torch
    from models_hamiltonian import get_model_hamiltoinian
    import ml_collections
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_timesteps = 1025
    q_dim = 2
    
    # Create dummy data
    dummy_data = {
        'q': torch.randn(batch_size, num_timesteps, q_dim).to(device),
        'dq': torch.randn(batch_size, num_timesteps, q_dim).to(device),
        'L': torch.randn(batch_size, num_timesteps).to(device),
        'L_grad_q': torch.randn(batch_size, num_timesteps, q_dim).to(device),
        'L_grad_dq': torch.randn(batch_size, num_timesteps, q_dim).to(device),
        'time': torch.linspace(0, 10, num_timesteps).unsqueeze(0).repeat(batch_size, 1).to(device),
        'idx': torch.arange(batch_size).to(device),
        'cond_dict': {},
    }
    
    # Test DHN model forward
    try:
        config = ml_collections.ConfigDict()
        config.hamiltonian = 'superres'
        config.num_embeddings = 100
        config.embedding_dim = 64
        config.q_dim = q_dim
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
        
        loss_config = ml_collections.ConfigDict()
        loss_config.weight_eom = 1.0
        loss_config.weight_denoise = 0.1
        loss_config.use_out_mask = True
        loss_config.crop_interval = (0, -1)
        
        model = get_model_hamiltoinian(config, dtype=torch.float32).to(device)
        loss, loss_dict = model.get_losses(dummy_data, loss_config)
        
        print(f"  ✓ DHN forward pass: loss={loss.item():.6f}")
        
        # Test backward
        loss.backward()
        print(f"  ✓ DHN backward pass successful")
        
    except Exception as e:
        print(f"  ✗ DHN forward/backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test CNN baseline forward
    try:
        config = ml_collections.ConfigDict()
        config.hamiltonian = 'baseline_cnn_superres'
        config.num_embeddings = 100
        config.q_dim = q_dim
        config.t_span = (0, 10)
        config.stage_step_size = (32, 8)
        config.step_size = 8
        config.train_step_span = (0, 513)
        
        config.network = ml_collections.ConfigDict()
        config.network.name = 'baseline_cnn'
        config.network.q_dim = config.q_dim
        config.network.hidden_dim = 64
        config.network.num_stages = 2
        
        loss_config = ml_collections.ConfigDict()
        loss_config.weight_eom = 1.0
        loss_config.crop_interval = (0, -1)
        
        model = get_model_hamiltoinian(config, dtype=torch.float32).to(device)
        loss, loss_dict = model.get_losses(dummy_data, loss_config)
        
        print(f"  ✓ CNN forward pass: loss={loss.item():.6f}")
        
        # Test backward
        loss.backward()
        print(f"  ✓ CNN backward pass successful")
        
    except Exception as e:
        print(f"  ✗ CNN forward/backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  All forward pass tests passed!\n")
    return True


def test_training_step():
    """Test a single training step end-to-end."""
    print("=" * 60)
    print("Test 5: Testing single training step (2 epochs)...")
    print("=" * 60)
    
    import torch
    import tempfile
    import shutil
    import pickle
    import numpy as np
    
    # Create temporary data directory
    tmp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(tmp_dir, 'data', 'two_body')
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)
    
    try:
        # Generate minimal fake data
        from data_gen.simulators import get_simulator
        import ml_collections
        
        config = ml_collections.ConfigDict()
        config.simulator = 'two_body'
        config.save_dir = os.path.join(data_dir, 'train')
        config.num_data = 10
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
        for i in range(10):
            np.random.seed(i)
            simulator = get_simulator(config)
            traj = simulator.sample_trajectory()
            traj['idx'] = i
            trajectories.append(traj)
        
        with open(os.path.join(data_dir, 'train', 'data.pkl'), 'wb') as f:
            pickle.dump(trajectories, f)
        with open(os.path.join(data_dir, 'test', 'data.pkl'), 'wb') as f:
            pickle.dump(trajectories[:5], f)
        
        print(f"  ✓ Generated test data: {len(trajectories)} trajectories")
        
        # Run a quick training test
        from trainers.trainer import Trainer
        
        # Create training config
        train_config = ml_collections.ConfigDict()
        train_config.workdir = os.path.join(tmp_dir, 'results')
        
        train_config.data = ml_collections.ConfigDict()
        train_config.data.path = data_dir
        train_config.data.batch_size = 4
        train_config.data.num_workers = 0  # Single threaded for test
        train_config.data.prefetch_factor = 2
        train_config.data.pin_memory = False
        train_config.data.cache = False
        
        train_config.model = ml_collections.ConfigDict()
        train_config.model.hamiltonian = 'baseline_cnn_superres'  # Faster for testing
        train_config.model.num_embeddings = 10
        train_config.model.q_dim = 2
        train_config.model.t_span = (0, 10)
        train_config.model.stage_step_size = (32, 8)
        train_config.model.step_size = 8
        train_config.model.train_step_span = (0, 513)
        
        train_config.model.network = ml_collections.ConfigDict()
        train_config.model.network.name = 'baseline_cnn'
        train_config.model.network.q_dim = train_config.model.q_dim
        train_config.model.network.hidden_dim = 32
        train_config.model.network.num_stages = 2
        
        train_config.loss = ml_collections.ConfigDict()
        train_config.loss.weight_eom = 1.0
        train_config.loss.crop_interval = (0, -1)
        
        train_config.optim = ml_collections.ConfigDict()
        train_config.optim.num_epochs = 2
        train_config.optim.lr = 1e-3
        
        train_config.logging = ml_collections.ConfigDict()
        train_config.logging.per_save_epochs = 100
        train_config.logging.per_save_tmp_epochs = 100
        train_config.logging.per_eval_epochs = 1
        train_config.logging.num_eval_batches = 1
        train_config.logging.num_vis = 2
        
        os.makedirs(train_config.workdir, exist_ok=True)
        
        # Disable wandb for test
        import wandb
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        
        print("  Running 2 training epochs...")
        trainer = Trainer(train_config)
        trainer.train_and_eval()
        
        print(f"  ✓ Training step completed successfully!")
        
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    print("  Training test passed!\n")
    return True


def test_plotting():
    """Test that plotting script can be imported and runs."""
    print("=" * 60)
    print("Test 6: Testing plotting script...")
    print("=" * 60)
    
    try:
        # Just test that the module imports correctly
        import visualization_scripts.plot_superres_results as plot_script
        print("  ✓ Plotting module imports successfully")
        
        # Check that key functions exist
        assert hasattr(plot_script, 'collect_results'), "Missing collect_results function"
        assert hasattr(plot_script, 'plot_figure14_style'), "Missing plot_figure14_style function"
        assert hasattr(plot_script, 'print_results_table'), "Missing print_results_table function"
        print("  ✓ All plotting functions present")
        
    except Exception as e:
        print(f"  ✗ Plotting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  Plotting test passed!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("SUPER-RESOLUTION SANITY TEST")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_data_generation()
    all_passed &= test_model_initialization()
    all_passed &= test_forward_pass()
    all_passed &= test_training_step()
    all_passed &= test_plotting()
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou're ready to train! Run the following commands:\n")
        print("  1. Generate data (if not downloaded):")
        print("     bash scripts/generate_superres_data.sh")
        print("\n  2. Train all models:")
        print("     bash scripts/superres_train_all.sh")
        print("\n  3. Extract results:")
        print("     bash scripts/superres_extract_all.sh")
        print("\n  4. Generate plots:")
        print("     python visualization_scripts/plot_superres_results.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("=" * 60)
        print("\nPlease fix the failing tests before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())



