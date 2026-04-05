import pickle
import numpy as np
import torch
import torch.nn as nn

# Load dataset
with open('demos.pkl', 'rb') as f:
    data = pickle.load(f)

X = torch.tensor(data['X'], dtype=torch.float32)
y = torch.tensor(data['y'], dtype=torch.float32)

obs_dim = X.shape[1]
action_dim = y.shape[1]
print(f'Dataset: {X.shape[0]} transitions, obs_dim={obs_dim}, action_dim={action_dim}')

# Policy network
class ImitationPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

policy = ImitationPolicy(obs_dim, action_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training
epochs = 500
for epoch in range(1, epochs + 1):
    policy.train()
    pred = policy(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}/{epochs}  loss={loss.item():.6f}')

torch.jit.script(policy).save('imitation_policy.pt')
print('Saved imitation_policy.pt')
