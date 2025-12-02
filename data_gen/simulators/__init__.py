def get_simulator(config):

  if config.simulator == 'single_pendulum':
    from .single_pendulum import Simulator

  elif config.simulator == 'double_pendulum':
    from .double_pendulum import Simulator

  elif config.simulator == 'two_body':
    from .two_body import Simulator

  elif config.simulator == 'three_body':
    from .three_body import Simulator

  else:
    raise NotImplementedError('Simulator not implemented.')

  return Simulator(config)
