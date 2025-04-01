import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error

TEST_DATA_PATH = "./score_data_radar_0.05/cs_radar_test.npz"
EXAMPLE_INDEX = 0
N_NONZERO_COEFFS = 2000  # Adjust as needed

# Load dataset
data = np.load(TEST_DATA_PATH)
y_full = data["y"][EXAMPLE_INDEX].real
h_full = data["h"][EXAMPLE_INDEX].real
mask_full = data["mask"][EXAMPLE_INDEX]

# Step 1: Apply 2D Fourier Transform to the original image
h_fourier = np.fft.fft2(h_full)
h_fourier_flat = h_fourier.flatten()

# Step 2: Create undersampling mask in Fourier space
mask_flat = mask_full.flatten()
M = np.count_nonzero(mask_flat)  # Number of measured values

# Step 3: Create measurement matrix in Fourier domain
P = np.eye(len(h_fourier_flat), dtype=np.float32)[mask_flat.astype(bool), :]
y_measured = h_fourier_flat[mask_flat.astype(bool)]

# Step 4: Separate the real and imaginary parts of the measurements
y_real = np.real(y_measured)
y_imag = np.imag(y_measured)

# Step 5: Apply OMP to both the real and imaginary parts
omp_real = OrthogonalMatchingPursuit(n_nonzero_coefs=N_NONZERO_COEFFS)
omp_imag = OrthogonalMatchingPursuit(n_nonzero_coefs=N_NONZERO_COEFFS)

# Fit the model on the real and imaginary parts separately
omp_real.fit(P, y_real)
omp_imag.fit(P, y_imag)

# Reconstruct the real and imaginary parts
y_hat_real = omp_real.coef_
y_hat_imag = omp_imag.coef_

# Combine the real and imaginary parts to form the recovered complex Fourier coefficients
y_hat_flat = y_hat_real + 1j * y_hat_imag
y_hat_fourier = y_hat_flat.reshape(h_full.shape)

# Step 6: Apply inverse FFT to get the recovered image
h_hat = np.abs(np.fft.ifft2(y_hat_fourier))

# Compute error metrics
mse = mean_squared_error(h_full, h_hat)
nmse = mse / np.mean(h_full ** 2)
rmse = np.sqrt(mse)

print(f"NMSE OMP (Fourier): {nmse:.4f}")
print(f"RMSE OMP (Fourier): {rmse:.4f}")

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(h_full, cmap="gray")
plt.title("Original h")

plt.subplot(1, 3, 2)
plt.imshow(y_full, cmap="gray")
plt.title("Measurements (Masked)")

plt.subplot(1, 3, 3)
plt.imshow(h_hat, cmap="gray")
plt.title("OMP (Fourier Domain)")

plt.tight_layout()
plt.show()
