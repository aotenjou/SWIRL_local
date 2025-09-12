# Gym-DB

Gym environments for database index selection using reinforcement learning.

## Installation

### From source
```bash
cd gym_db
pip install -e .
```

### With development dependencies
```bash
cd gym_db
pip install -e ".[dev]"
```

## Environments

This package provides several gym environments for database index selection:

- **DB-v1**: Basic database index selection environment
- **DB-v2**: Environment with action masking (currently commented out)
- **DB-v3**: Environment for reproducing DRLindex paper implementation (currently commented out)

## Usage

```python
import gym
import gym_db

# Create environment
env = gym.make('DB-v1', config=your_config)

# Use the environment
observation = env.reset()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
```

## Dependencies

- gym>=0.17.0
- numpy
- pandas

## Development

This package is part of the SWIRL project for automated database index selection using reinforcement learning.

