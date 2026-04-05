# Lab 5 — Robot Manipulation with RL and Imitation Learning

## Overview

This lab teaches a robot arm (Hello Robot Stretch3) to touch a mustard bottle on a table. It is structured as a three-part pipeline:

```
Part 1: rl.py              — Define the environment, train a policy using RL (PPO)
Part 2: data_collection.py — Run the trained policy and record demonstrations
Part 3: imitation.py       — Train a neural network to copy the expert policy
```

This second approach (learn by copying a trained expert) is called **Imitation Learning** or **Behavior Cloning**.

---

## The Pipeline at a Glance

```
rl.py
  └── Defines TouchEnv (simulation world, observation, reward)
  └── Trains PPO policy
  └── Saves: log/PPOExperiment_.../exp_seed=.../policy.pt

data_collection.py
  └── Loads policy.pt
  └── Runs policy in environment for 1000 episodes
  └── Records every (observation, action) pair
  └── Saves: demos.pkl

imitation.py
  └── Loads demos.pkl
  └── Trains MLP: observation → action
  └── Saves: imitation_policy.pt

eval_il.py  (provided, not modified)
  └── Loads imitation_policy.pt
  └── Runs it visually in the environment
```

---

## Vocabulary Reference

| Term | Plain English |
|---|---|
| **Environment** | The simulated world the robot lives in |
| **Observation** | What the robot "sees" — a list of numbers describing current state |
| **Action** | What the robot does — numbers telling joints how to move |
| **Reward** | A score given after each action |
| **Policy** | The "brain" — takes an observation, outputs an action |
| **Episode** | One attempt from start (reset) to finish |
| **Step** | One tick of the simulation clock |
| **Epoch** | One complete pass over training data |
| **Neural network** | A mathematical function with many adjustable numbers (weights) |
| **Tensor** | A multi-dimensional array used by PyTorch |
| **Behavior cloning** | Supervised learning where labels are expert actions |
| **Local frame** | Coordinates relative to the robot's own body |
| **World frame** | Absolute coordinates in the simulation world |
| **End effector** | The tip of the robot arm — the "hand" |
| **Joint angle** | How far a joint is rotated, in radians |
| **URDF** | Unified Robot Description Format — an XML file describing a robot's shape and joints |

---

## The Simulation Engine: mengine

### What it is

**mengine** ("Manipulation Engine") is a custom Python library written by Professor Zackory Erickson at CMU (`zackory@cmu.edu`). It is open source (MIT license) and available at `github.com/Zackory/mengine`.

It is not an industry-standard framework — it is a research and teaching tool built specifically for CMU robotics courses.

### What it does

mengine is a **wrapper around PyBullet** (a C++ physics engine). PyBullet is powerful but verbose to use directly. mengine provides clean, readable Python classes on top of it.

**Raw PyBullet** (without mengine):
```python
body = p.loadURDF('stretch.urdf', basePosition=[0,0,0], physicsClientId=self.id)
pos, orient = p.getLinkState(body, 33, computeForwardKinematics=True, physicsClientId=self.id)[4:6]
```

**With mengine:**
```python
robot = m.Robot.Stretch3(position=[0,0,0])
pos, orient = robot.get_link_pos_orient(robot.end_effector)
```

### Layered architecture

```
rl.py (your code)
    ↓ uses
mengine  (m.Robot.Stretch3, m.Shape, m.step_simulation, ...)
    ↓ wraps
PyBullet  (p.loadURDF, p.stepSimulation, p.getLinkState, ...)
    ↓ runs
Bullet Physics Engine  (C++ physics solver)
```

### Key mengine components

| File | Provides |
|---|---|
| `env.py` | `Env`, `Robot`, `Shape`, `Ground`, `URDF`, `Camera` — world-building API |
| `bodies/body.py` | `Body` — base class with position/joint/contact methods |
| `bodies/robot.py` | `Robot` — adds `end_effector` and gripper concepts |
| `bodies/stretch3.py` | `Stretch3` — the specific robot; loads URDF, sets joint limits, fixes masses |
| `assets/` | 3D model files (URDF, `.obj` meshes) for the robot, table, mustard bottle, etc. |

### The Stretch3 robot

Stretch3 is a real-world mobile arm made by Hello Robot. Its URDF is loaded from:
```
mengine/assets/stretch3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf
```

Key properties set in `stretch3.py`:
- `end_effector = 33` — link index for the arm tip
- `controllable_joints = [0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 26, 29]` — 12 joints the policy can control
- Wheel masses set to 200 kg for stability; link masses reduced to 0.01 kg

---

## Observations and Actions

### Actions — "What the robot does"

An action is **7 numbers**, each between -1 and 1:

| Index | Controls | Scale in step() |
|---|---|---|
| 0 | Right wheel speed | × 0.5 |
| 1 | Left wheel speed | × 0.5 |
| 2 | Lift joint (arm up/down) | × 0.025 |
| 3 | Arm extension (reach forward) | ÷ 4 × 0.025, applied to 4 joints |
| 4 | Wrist yaw (rotate left/right) | × 0.025 |
| 5 | Wrist pitch (tilt up/down) | × 0.025 |
| 6 | Wrist roll (spin) | × 0.025 |

Each action is a **delta** (change) applied to the current joint angles, not an absolute target. The gripper is always fixed at 0 — never opened or closed in this lab.

### Observations — "What the robot sees"

An observation is **14 numbers** describing the world at one moment:

```
obs = [ee_x, ee_y, ee_z,  obj_x, obj_y, obj_z,  dx, dy, dz,  lift, arm, yaw, pitch, roll]
```

| Slice | Content | Dim |
|---|---|---|
| `obs[0:3]` | End effector position in robot's local frame | 3 |
| `obs[3:6]` | Mustard bottle position in robot's local frame | 3 |
| `obs[6:9]` | Difference: `ee_pos − obj_pos` | 3 |
| `obs[9]` | Lift joint angle (radians) | 1 |
| `obs[10]` | Sum of 4 arm extension joints | 1 |
| `obs[11]` | Wrist yaw angle | 1 |
| `obs[12]` | Wrist pitch angle | 1 |
| `obs[13]` | Wrist roll angle | 1 |

### Why local frame for positions?

All positions are expressed **relative to the robot's base**, not the world. The robot spawns at a random position each episode. If world coordinates were used, the same reaching geometry would look like different numbers episode to episode. In robot-local frame, "the bottle is 0.6m in front of me" always looks the same regardless of where in the room the robot started.

### One step, end to end

```
Observation (14 numbers)
  → Policy (neural network)
    → Action (7 numbers)
      → step() scales and adds to joint angles
        → physics simulation advances 10 steps
          → new Observation (14 numbers)
            → repeat
```

---

## File 1: `rl.py`

### Purpose

1. Defines `TouchEnv` — the Gymnasium-compatible simulation environment
2. Runs PPO training via Tianshou

### `TouchEnv.__init__`

```python
self.env = m.Env(gravity=[0, 0, -1], render=render_mode=='human')
self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(14,))
self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
```

- `gravity=[0, 0, -1]` — weaker than real gravity (-9.81) for simulation stability
- `observation_space` — tells RL libraries the shape and range of observations
- `action_space` — tells RL libraries the shape and range of actions

### `_get_obs()` — **[Lab TODO]**

```python
ee_pos, _ = self.robot.get_link_pos_orient(self.robot.end_effector, local_coordinate_frame=True)
obj_pos, _ = self.robot.global_to_local_coordinate_frame(*self.object.get_base_pos_orient())
diff = ee_pos - obj_pos
angles = self.robot.get_joint_angles(self.robot.controllable_joints)
joint_features = np.array([angles[2], np.sum(angles[3:7]), angles[7], angles[8], angles[9]])
return np.concatenate([ee_pos, obj_pos, diff, joint_features]).astype(np.float32)
```

- Gets end effector and object positions in the robot's local coordinate frame
- Computes their difference explicitly (gives the network a direct "direction to target" signal)
- Extracts 5 joint features: lift, summed arm extension, wrist yaw/pitch/roll
- Returns a flat float32 array of shape (14,) — required by PyTorch
- **Added because:** the prompt required an observation vector with those four components

### `reset()`

Rebuilds the world fresh each episode:
- Clears the simulation
- Spawns ground, table (randomized height), mustard bottle (randomized x/y), robot (randomized x/y/rotation)
- Runs 10 physics steps so the bottle settles on the table
- Returns the initial observation

Randomization is intentional — forces the policy to generalize instead of memorizing one fixed configuration.

### `step()` — **[Lab TODO for reward]**

Applies one action and advances the simulation:

```python
# Scale action from [-1,1] to actual joint deltas
scaled_action = np.concatenate([action[:2]*0.5, [action[2]*scale], [action[3]/4.0*scale]*4, action[4:]*scale, [0, 0]])
current_angles = self.robot.get_joint_angles(self.robot.controllable_joints)
self.robot.control(current_angles + scaled_action)
m.step_simulation(steps=10, realtime=self.env.render)

# Reward function (added for lab)
ee_pos, _ = self.robot.get_link_pos_orient(self.robot.end_effector)
obj_pos, _ = self.object.get_base_pos_orient()
reward = -float(np.linalg.norm(ee_pos - obj_pos))
```

- `scaled_action` maps policy output [-1,1] to small joint angle deltas
- `self.robot.control(current_angles + scaled_action)` — position control: "move to this angle"
- Reward = negative Euclidean distance between end effector and bottle in world frame
  - 0 when touching, increasingly negative when farther away
  - PPO maximizes reward, so it learns to minimize distance
- `terminated = False` always — episodes only end via `max_episode_steps=75` (set in `gym.register`)
- **Added because:** the prompt required a dense reward based on negative distance

### PPO Training Configuration

```python
gym.register(id='TouchEnv', entry_point=TouchEnv, max_episode_steps=75)
```

Registers the environment globally so `gym.make('TouchEnv')` works anywhere.

Key PPO settings:
| Parameter | Value | Meaning |
|---|---|---|
| `max_epochs` | 200 | Train for 200 epochs |
| `epoch_num_steps` | 2048 | Collect 2048 env steps per epoch |
| `num_training_envs` | 8 | 8 parallel environments |
| `batch_size` | 64 | Update network on 64 samples at a time |
| `lr` | 1e-3 | Learning rate |
| `gamma` | 0.99 | Discount factor (future rewards worth 99% of immediate) |
| `hidden_sizes` | (64, 64) | Both actor and critic have 2 layers of 64 neurons |
| `persistence_base_dir` | 'log' | Saves checkpoints to `log/` folder |

The `if __name__ == '__main__':` guard ensures `run_experiment()` only fires when you run `python rl.py` directly, not when `data_collection.py` imports `TouchEnv`.

---

## File 2: `data_collection.py`

### Purpose

Loads the trained RL policy, runs it in the environment for 1000 episodes, and records every (observation, action) pair into `demos.pkl`.

### Finding the checkpoint

```python
all_folders = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith('PPOExperiment')]
latest_folder = max(all_folders, key=os.path.getmtime)
checkpoint_path = os.path.join(latest_folder, seed_folder, 'policy.pt')
```

Finds the most recently modified PPOExperiment folder in `log/` and loads its `policy.pt`.

### Environment setup

```python
env = gym.make('TouchEnv')
venv = DummyVectorEnv([lambda: env])
```

`DummyVectorEnv` wraps the environment so all outputs are **batched** with a leading dimension of 1:
- `obs` shape: `(1, 14)` instead of `(14,)`
- `action` shape: `(1, 7)` instead of `(7,)`
- `terminated` shape: `(1,)` instead of a plain boolean

### Loading the policy

```python
data = torch.load(checkpoint_path, weights_only=False)
policy = data.policy if hasattr(data, 'policy') else data
policy.eval()
```

- Handles two possible checkpoint formats (wrapped in experiment object, or raw policy)
- `policy.eval()` — switches to inference mode; some layers behave differently during training vs. evaluation

### Collection loop — **[Lab TODO]**

```python
X = []
y = []
for i in range(n_demos):
    obs, info = venv.reset(seed=np.random.randint(1000000))
    terminated = False
    truncated = False
    while not terminated and not truncated:
        with torch.no_grad():
            result = policy(Batch(obs=obs, info=info))
            action = result.act

        X.append(obs[0])          # strip batch dim: (1,14) → (14,)
        y.append(action[0])       # strip batch dim: (1,7)  → (7,)
        obs, _, terminated, truncated, info = venv.step(action)
        terminated = terminated[0]
        truncated = truncated[0]
```

- `torch.no_grad()` — skip gradient tracking (not training, just inferring)
- `obs[0]` and `action[0]` — strip the batch dimension added by DummyVectorEnv
- `terminated[0]` and `truncated[0]` — unwrap from array to boolean for the while condition
- **Added because:** the prompt required appending obs/action to X/y, then stepping the environment

### Saving — **[Lab TODO]**

```python
with open('demos.pkl', 'wb') as f:
    pickle.dump({'X': np.array(X), 'y': np.array(y)}, f)
```

- `np.array(X)` converts the list of 14D arrays into one array of shape `(N, 14)`
- `np.array(y)` similarly gives shape `(N, 7)`
- Saved as a Python dictionary with keys `'X'` and `'y'`
- With 1000 episodes × 75 steps = ~75,000 transitions
- **Added because:** the prompt required saving to `demos.pkl` using Pickle

### Saved file structure

```
demos.pkl
  └── dict
        ├── 'X'  →  np.array, shape (75000, 14), dtype float32  — observations
        └── 'y'  →  np.array, shape (75000, 7),  dtype float32  — actions
```

Each row is one time step. The dataset is **per-step**, not per-trajectory — every individual moment is a separate training example.

---

## File 3: `imitation.py`

### Purpose

Loads `demos.pkl` and trains a neural network to predict actions from observations. This is **behavior cloning** — pure supervised learning on expert demonstrations. No reward signal, no trial and error.

### Loading the dataset

```python
with open('demos.pkl', 'rb') as f:
    data = pickle.load(f)

X = torch.tensor(data['X'], dtype=torch.float32)  # (75000, 14)
y = torch.tensor(data['y'], dtype=torch.float32)  # (75000, 7)
```

Converted to PyTorch tensors so they can flow through the network and participate in gradient computation.

### Network architecture

```python
class ImitationPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),   # 14 → 64
            nn.ReLU(),
            nn.Linear(64, 64),        # 64 → 64
            nn.ReLU(),
            nn.Linear(64, action_dim) # 64 → 7
        )
```

```
Input (14)  →  Linear  →  64  →  ReLU  →  Linear  →  64  →  ReLU  →  Linear  →  Output (7)
```

- **Linear layers** — matrix multiply + bias (the learnable parameters)
- **ReLU** — replaces negatives with 0; adds nonlinearity so the network can learn complex mappings
- **No final activation** — actions are unbounded real numbers, not probabilities
- Matches the PPO actor architecture (`hidden_sizes=(64, 64)`) used in training

### Training

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(1, 501):
    policy.train()
    pred = policy(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

| Step | What happens |
|---|---|
| `policy(X)` | Forward pass: run all 75k observations through the network |
| `loss_fn(pred, y)` | Compute MSE: mean of squared differences between predicted and expert actions |
| `optimizer.zero_grad()` | Clear old gradients (they accumulate by default in PyTorch) |
| `loss.backward()` | Backpropagation: compute how much each weight contributed to the loss |
| `optimizer.step()` | Update every weight slightly to reduce loss |

- **Loss function:** Mean Squared Error — `mean((predicted_action - expert_action)²)`
- **Optimizer:** Adam with learning rate 0.001
- **500 epochs**, full-batch (all 75k rows per update) — dataset is small enough for this
- Loss plateaus around 0.82, reflecting that the RL expert itself is imperfect

### Saving

```python
torch.jit.script(policy).save('imitation_policy.pt')
```

Saved as **TorchScript** — a format that embeds the full network graph and class definition into the file. This allows `eval_il.py` to load it with `torch.load(...)` without importing the `ImitationPolicy` class. Regular `torch.save(policy, ...)` would fail because Python's pickle format requires the class to be importable at load time.

---

## Setup and Installation

This project requires Python 3.12 (not 3.13 free-threaded, which lacks pre-built wheels for key dependencies).

```bash
# Create and activate virtual environment with Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install gymnasium tianshou rliable
pip install joblib
```

mengine is installed as an editable package from the local clone:
```bash
pip install -e C:\path\to\mengine
```

---

## Running the Lab

### Part 1 — Train the RL policy

```bash
cd "Labs/Lab 5"
python rl.py
```

Trains for 200 epochs. Saves checkpoints to `log/PPOExperiment_<timestamp>/exp_seed=42/`.

### Part 2 — Collect demonstrations

```bash
python data_collection.py
```

Runs the trained policy for 1000 episodes and saves `demos.pkl`.

### Part 3 — Train imitation policy

```bash
python imitation.py
```

Trains for 500 epochs and saves `imitation_policy.pt`.

### Evaluate imitation policy visually

```bash
python eval_il.py
```

Opens a PyBullet window and runs the imitation policy in real time.

---

## Common Pitfalls

**`numpy.object_` type error**
The observation returned `None` (the original TODO). PyTorch cannot convert an array of Python objects to a tensor. Fix: implement `_get_obs()` to return a proper float32 array.

**`cp313t` wheel mismatch**
Python 3.13 free-threaded (indicated by `t` suffix) has no pre-built wheels for grpcio, h5py, scipy, etc. Solution: use standard Python 3.12.

**`Can't get attribute 'ImitationPolicy'` on load**
`torch.save(policy, ...)` uses Python pickle, which requires the class definition to be importable at load time. `eval_il.py` never imports `ImitationPolicy`. Solution: use `torch.jit.script(policy).save(...)` which embeds the class definition in the file.

**High imitation loss (~0.82)**
The RL policy itself is not perfect. Behavior cloning inherits the noise from the expert. If the RL policy trained longer, demonstrations would be cleaner and imitation loss would be lower.

**`EpochStopCallbackRewardThreshold(195)` never triggers**
The reward is negative distance (around -0.1 to -2.0), so it will never reach 195. Training always runs the full 200 epochs. This is intentional behavior for this lab setup.

**`terminated[0]` vs `terminated`**
`DummyVectorEnv` returns `terminated` as a numpy array of shape `(1,)`. The `while` loop needs a plain boolean, so indexing `[0]` is required.

---

## Key Concepts Summary

### Reinforcement Learning (RL)
The robot learns by trial and error. It receives a reward after each action and gradually figures out which actions lead to higher rewards. PPO (Proximal Policy Optimization) is the specific algorithm used — it's popular for robotics because it's stable and handles continuous actions well.

### Behavior Cloning
Instead of learning from rewards, the network learns from recorded expert demonstrations. Given an observation, predict the action the expert took. This is standard supervised learning — the "labels" are just actions instead of categories.

### Why Behavior Cloning After RL?
- Behavior cloning is much simpler to train — no reward engineering needed
- A smaller, simpler network can often reproduce expert behavior adequately
- Useful when you want a lightweight deployable policy

### Dense vs. Sparse Reward
The reward here is **dense** — the robot gets a signal every single step (negative distance). A **sparse** reward would only give a signal on success (e.g., reward=1 only when touching). Dense rewards are much easier to learn from because there's always feedback.
