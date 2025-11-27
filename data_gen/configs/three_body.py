import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.simulator = 'three_body'

  config.save_dir = 'tmp'
  config.num_data = 1000
  config.num_vis = 5

  config.t_span = (0, 10)
  config.num_timesteps = 1025

  config.constants = constants = ml_collections.ConfigDict()
  constants.G = 1.0  # Gravitational constant (can be normalized to 1.0)
  constants.m1 = (0.5, 2.0)  # Mass of body 1 [kg]
  constants.m2 = (0.5, 2.0)  # Mass of body 2 [kg]
  constants.m3 = (0.5, 2.0)
  constants.r1 = (0.5,1.0) # Orbital radius of body 1 [km]
  constants.r2 = (1.1,1.5) # Orbital radius of body 2 [km]
  constants.r3 = (1.6, 2.0)
  return config
