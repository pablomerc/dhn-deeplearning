"""Debug script to inspect checkpoint contents."""
import os
import torch

def inspect_checkpoint(path, name):
    print(f"\n{'='*60}")
    print(f"Inspecting: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")
    
    if not os.path.exists(path):
        print("  FILE NOT FOUND!")
        return
    
    ckpt = torch.load(path, map_location='cpu')
    print(f"Top-level keys: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print(f"\nModel state dict has {len(state_dict)} keys")
        
        # Check for codebook
        codebook_keys = [k for k in state_dict.keys() if 'codebook' in k]
        print(f"Codebook keys: {codebook_keys}")
        
        if codebook_keys:
            for k in codebook_keys:
                print(f"  {k}: shape={state_dict[k].shape}, dtype={state_dict[k].dtype}")
        
        # Show first 10 keys
        print(f"\nFirst 10 model keys:")
        for i, k in enumerate(list(state_dict.keys())[:10]):
            print(f"  {k}: {state_dict[k].shape}")

def main():
    base = 'results/superres'
    
    # Single Pendulum DHN
    inspect_checkpoint(
        os.path.join(base, 'sinpend_4x', 'checkpoint.pth'),
        'Single Pendulum DHN - Training'
    )
    inspect_checkpoint(
        os.path.join(base, 'sinpend_4x', 'extract', 'checkpoint.pth'),
        'Single Pendulum DHN - Extract'
    )
    
    # Single Pendulum CNN
    inspect_checkpoint(
        os.path.join(base, 'sinpend_baseline_cnn_4x', 'checkpoint.pth'),
        'Single Pendulum CNN - Training'
    )
    
    # Two-Body DHN
    inspect_checkpoint(
        os.path.join(base, 'two_body_4x', 'checkpoint.pth'),
        'Two-Body DHN - Training'
    )
    inspect_checkpoint(
        os.path.join(base, 'two_body_4x', 'extract', 'checkpoint.pth'),
        'Two-Body DHN - Extract'
    )

if __name__ == '__main__':
    main()
