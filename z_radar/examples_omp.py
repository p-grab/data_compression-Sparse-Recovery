import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error

TEST_DATA_PATH = "./score_data_radar_0.05/cs_radar_test.npz"
EXAMPLE_INDEX = 0
N_NONZERO_COEFFS = 500

data = np.load(TEST_DATA_PATH)
y_full = data["y"][EXAMPLE_INDEX].real
h_full = data["h"][EXAMPLE_INDEX].real
mask_full = data["mask"][EXAMPLE_INDEX]

mask_flat = mask_full.flatten()
h_flat = h_full.flatten()
y_flat = y_full.flatten()

M = np.count_nonzero(mask_flat)
N = h_flat.shape[0]

P = np.eye(N, dtype=np.float32)[mask_flat.astype(bool), :]
y_measured = y_flat[mask_flat.astype(bool)]

# OMP Reconstruction
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=N_NONZERO_COEFFS)
omp.fit(P, y_measured.astype(np.float32))
h_hat_flat = omp.coef_
h_hat = h_hat_flat.reshape(h_full.shape)

mse = mean_squared_error(h_full, h_hat)
nmse = mse / np.mean(h_full ** 2)
print(f"NMSE OMP: {nmse:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(h_full, cmap="gray")
plt.title("Original h")

plt.subplot(1, 3, 2)
plt.imshow(y_full, cmap="gray")
plt.title("measurements")

plt.subplot(1, 3, 3)
plt.imshow(h_hat, cmap="gray")
plt.title(" OMP")

plt.tight_layout()
plt.show()
