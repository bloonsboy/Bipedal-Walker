#2 Reinforcement Learning Project: Bipedal Walker with PPO

This repository contains an implementation of the PPO (Proximal Policy Optimization) algorithm to solve the "BipedalWalker-v3" environment from Gymnasium.

#3 Project Structure

config.py: Contains all hyperparameters for training.

train.py: Main script to launch the training.

evaluate.py: Script to evaluate a saved model and see it in action.

ppo_agent/: Module containing the PPO logic.

model.py: Definition of the Actor-Critic networks.

storage.py: RolloutBuffer class for storing experiences.

models/: Directory where trained models (.pth) are saved.

logs/: Directory where TensorBoard logs are saved.

videos/: Directory where evaluation videos are saved.

requirements.txt: Python dependencies.

#3 Installation

Clone this repository and navigate into the directory.

Create a virtual environment and activate it:

```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the dependencies:

```python
pip install -r requirements.txt
```

#3 Usage

**1. Training**

To start a new training run, execute train.py:
```python
python train.py
```

Training logs will be saved in the logs/ directory. You can visualize them with TensorBoard:
```python
tensorboard --logdir=logs
```

The best model will be saved to models/ppo_bipedal_walker.pth.

**2. Evaluation**

To watch your trained agent in action:
```python
python evaluate.py
```

To record a video of your agent:
```python
python evaluate.py --record
```