import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm import tqdm


def load_mri_images(base_path, num_sessions=27, resize_to=(448, 448)):
    images = []
    for session in range(1, num_sessions + 1):
        path = os.path.join(base_path, f"ses-{session:02d}", "anat", "*.png")
        file_list = sorted(glob(path))
        for file in file_list:
            
            if "T2w" not in os.path.basename(file):
                continue
            
            img = Image.open(file).convert("L").resize(resize_to)
            # img_np = np.array(img) / 255.0  # Normalize to [0,1]
            img_np = np.array(img, dtype=np.float32) / 255.0

            images.append(img_np)
    return np.stack(images)


def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(freq):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(freq)))

def generate_mask(shape, M):
    # take random M entries
    N = shape[0] * shape[1]
    idx = np.random.choice(N, M, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return mask.reshape(shape)

def apply_measurement_operator(h_full, mask, noise_std=0.0):
    # add noise and undersampling
    noise = noise_std * (np.random.randn(*h_full.shape) + 1j * np.random.randn(*h_full.shape)) / np.sqrt(2)
    return mask * (h_full + noise)  # y = Ph + v

# measures (out of 448*448 = 200704)
# M = 5000 
M = 10000 
# noise cov
noise_std = 0.05
save_path = "./score_data_10k_0.05"
base_path = r"C:\TUDelft\Q3\data_entropy\output_png"

os.makedirs(save_path, exist_ok=True)
images = load_mri_images(base_path)
print("img shape:", images.shape)


# Subtrack mean?
# mean_img = np.mean(images, axis=0)
# images -= mean_img[None, :, :]  


X_train, Y_train, MASKS = [], [], []

for i, img in tqdm(enumerate(images), total=len(images)):
    
    h = fft2c(img)  
    mask = generate_mask(h.shape, M) 
    y = apply_measurement_operator(h, mask, noise_std=noise_std)

    X_train.append(y)
    Y_train.append(h)
    MASKS.append(mask)

np.savez_compressed(os.path.join(save_path, "cs_mri_data.npz"),
                    y=np.array(X_train),
                    h=np.array(Y_train),
                    mask=np.array(MASKS))

# WITH Subtracking
# np.savez_compressed(os.path.join(save_path, "cs_mri_data.npz"),
#                     y=np.array(X_train),
#                     h=np.array(Y_train),
#                     mask=np.array(MASKS),
#                     mean_img=mean_img)



print("Saved to ", os.path.join(save_path, "cs_mri_data.npz"))


# split data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test, MASKS_train, MASKS_test = train_test_split(
    X_train, Y_train, MASKS, test_size=0.2, random_state=42
)

np.savez_compressed(os.path.join(save_path, "cs_mri_train.npz"),
                    y=np.array(X_train),
                    h=np.array(Y_train),
                    mask=np.array(MASKS_train))

np.savez_compressed(os.path.join(save_path, "cs_mri_test.npz"),
                    y=np.array(X_test),
                    h=np.array(Y_test),
                    mask=np.array(MASKS_test))
 

plt.subplot(1, 3, 1)
plt.imshow(np.abs(img), cmap='gray')
plt.title("Original img")

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(np.abs(h)), cmap='gray')
plt.title("Fourier")

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.show()

