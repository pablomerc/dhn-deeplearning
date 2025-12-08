"""Generate remaining data for superres experiments."""
import os
import pickle
import numpy as np
from data_gen.simulators import get_simulator

# Remaining systems to generate
systems = [
    ('double_pendulum', 'data_gen.configs.double_pendulum'),
    ('two_body', 'data_gen.configs.two_body'),
]

for system_name, config_module in systems:
    module = __import__(config_module, fromlist=['get_config'])
    
    # Train data
    config = module.get_config()
    config.save_dir = f'data/{system_name}/train'
    config.num_vis = 0
    np.random.seed(0)
    trajectories = []
    for i in range(config.num_data):
        simulator = get_simulator(config)
        traj = simulator.sample_trajectory()
        traj['idx'] = i
        trajectories.append(traj)
        if (i+1) % 100 == 0:
            print(f'{system_name} train: {i+1}/{config.num_data}')
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'data.pkl'), 'wb') as fp:
        pickle.dump(trajectories, fp)
    print(f'{system_name} train: {len(trajectories)} trajectories - DONE')
    
    # Test data
    config.num_data = 200
    config.save_dir = f'data/{system_name}/test'
    np.random.seed(1)
    trajectories = []
    for i in range(config.num_data):
        simulator = get_simulator(config)
        traj = simulator.sample_trajectory()
        traj['idx'] = i
        trajectories.append(traj)
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'data.pkl'), 'wb') as fp:
        pickle.dump(trajectories, fp)
    print(f'{system_name} test: {len(trajectories)} trajectories - DONE')

print('All data generation complete!')





