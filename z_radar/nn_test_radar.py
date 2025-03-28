import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

MODEL_PATH = "./score_model_radar.pt"
DATA_PATH = "./score_data_radar_0.05/cs_radar_test.npz"
BATCH_SIZE = 8
SIGMA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return torch.tensor(h_noisy).unsqueeze(0), torch.tensor(target).unsqueeze(0)


print("Load model")
model = ScoreNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


print("Load testing")
dataset = RadarScoreDataset(DATA_PATH, sigma=SIGMA)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

loss_fn = nn.MSELoss()
total_loss = 0.0

with torch.no_grad():
    batch_noisy, batch_target = next(iter(loader))  
    batch_noisy = batch_noisy.to(DEVICE)
    batch_target = batch_target.to(DEVICE)
    preds = model(batch_noisy)

    loss = loss_fn(preds, batch_target)
    print(f"MSE Loss (1 batch): {loss.item():.4f}")
    total_loss += loss.item()

batch_noisy = batch_noisy.cpu()
batch_target = batch_target.cpu()
preds = preds.cpu()

fig, axs = plt.subplots(3, BATCH_SIZE, figsize=(4 * BATCH_SIZE, 8))
for i in range(BATCH_SIZE):
    axs[0, i].imshow(batch_noisy[i, 0], cmap='gray')
    axs[0, i].set_title("Noisy input")

    axs[1, i].imshow(batch_target[i, 0], cmap='gray')
    axs[1, i].set_title("True delta log p")

    axs[2, i].imshow(preds[i, 0], cmap='gray')
    axs[2, i].set_title("Predicted delta log p")

plt.tight_layout()
plt.show()
