import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# DATA SOURCE:
# small data from here:
# https://oxford-robotics-institute.github.io/radar-robotcar-dataset/downloads

# resize_to = (448, 128)  
resize_to = (224, 64)
M = 5000
noise_std = 0.01
save_path = "./score_data_radar_0.05"
base_path = r"C:\TUDelft\Q3\data_entropy\radar_data\oxford_radar_robotcar_dataset_sample_small\2019-01-10-14-36-48-radar-oxford-10k-partial\radar"

os.makedirs(save_path, exist_ok=True)

def load_radar_images(base_path, resize_to=(448, 128)):
    file_list = []
    for ext in ["png", "PNG"]:
        file_list.extend(glob(os.path.join(base_path, f"*.{ext}")))
    file_list = sorted(file_list)
    
    images = []
    for file in file_list:
        print("Load:", file)
        img = Image.open(file).convert("L").resize(resize_to)
        img_np = np.array(img, dtype=np.float32) / 255.0 
        images.append(img_np)
    return np.stack(images)

def generate_mask(shape, M):
    N = shape[0] * shape[1]
    idx = np.random.choice(N, M, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return mask.reshape(shape)

def apply_measurement_noise(h_full, mask, noise_std=0.0):
    noise = noise_std * np.random.randn(*h_full.shape)
    return mask * (h_full + noise)

images = load_radar_images(base_path, resize_to)

X_all, Y_all, MASKS = [], [], []

for i, img in tqdm(enumerate(images), total=len(images)):
    h = img  
    mask = generate_mask(h.shape, M)
    y = apply_measurement_noise(h, mask, noise_std=noise_std)

    X_all.append(y)
    Y_all.append(h)
    MASKS.append(mask)

X_all = np.array(X_all)
Y_all = np.array(Y_all)
MASKS = np.array(MASKS)

np.savez_compressed(os.path.join(save_path, "cs_radar_data.npz"),
                    y=X_all, h=Y_all, mask=MASKS)

X_train, X_test, Y_train, Y_test, MASKS_train, MASKS_test = train_test_split(
    X_all, Y_all, MASKS, test_size=0.2, random_state=42
)

np.savez_compressed(os.path.join(save_path, "cs_radar_train.npz"),
                    y=X_train, h=Y_train, mask=MASKS_train)

np.savez_compressed(os.path.join(save_path, "cs_radar_test.npz"),
                    y=X_test, h=Y_test, mask=MASKS_test)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.abs(Y_train[0]), cmap='gray')
plt.title("original h")

plt.subplot(1, 3, 2)
plt.imshow(np.abs(X_train[0]), cmap='gray')
plt.title("measurements")

plt.subplot(1, 3, 3)
plt.imshow(MASKS_train[0], cmap='gray')
plt.title("Mask")
plt.tight_layout()
plt.show()
