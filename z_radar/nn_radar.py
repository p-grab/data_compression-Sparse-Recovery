import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time  


DATA_PATH = "./score_data_radar_0.05/cs_radar_train.npz"
BATCH_SIZE = 32
SIGMA = 0.1

# EPOCHS = 50
EPOCHS = 80

LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./score_model_radar.pt"


class RadarScoreDataset(Dataset):
    def __init__(self, path, sigma=0.1):
        data = np.load(path)
        self.h_clean = data["h"].astype(np.float32)
        self.sigma = sigma

    def __len__(self):
        return len(self.h_clean)

    def __getitem__(self, idx):
        h = self.h_clean[idx]
        noise = np.random.randn(*h.shape).astype(np.float32)
        h_noisy = h + self.sigma * noise
        
        target = -noise / (self.sigma ** 2)
        # target = -noise / self.sigma 

        
        return torch.tensor(h_noisy).unsqueeze(0), torch.tensor(target).unsqueeze(0)

class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

dataset = RadarScoreDataset(DATA_PATH, sigma=SIGMA)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ScoreNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"Training for {EPOCHS} epochs")

for epoch in range(EPOCHS):
    start_time = time.time() 
    model.train()
    total_loss = 0
    for batch_noisy, batch_target in loader:
        batch_noisy = batch_noisy.to(DEVICE)
        batch_target = batch_target.to(DEVICE)

        optimizer.zero_grad()
        output = model(batch_noisy)
        loss = loss_fn(output, batch_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    end_time = time.time()  
    epoch_time = end_time - start_time

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(loader):.6f} - Time: {epoch_time:.2f}s")

torch.save(model.state_dict(), SAVE_PATH)
print("saved to :", SAVE_PATH)
