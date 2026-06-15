import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Dummy dataset
X = torch.randn(10000, 100)
y = torch.randint(0, 10, (10000,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for inputs, targets in loader:
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: {loss.item():.4f}")
