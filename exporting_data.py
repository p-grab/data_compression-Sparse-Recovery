import os
import gzip
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# directories
base_dir = r"C:\TUDelft\Q3\data_entropy\data_\sub-01"
output_png_dir = r"C:\TUDelft\Q3\data_entropy\output_png"
os.makedirs(output_png_dir, exist_ok=True)

def decompress_nii_gz():
    for i in range(1, 28):
        ses_id = f"ses-{i:02d}"
        anat_dir = os.path.join(base_dir, ses_id, "anat")

        for file in os.listdir(anat_dir):
            if file.endswith(".nii.gz"):
                gz_path = os.path.join(anat_dir, file)
                nii_path = os.path.join(anat_dir, file[:-3])

                # not create folders
                if not os.path.exists(nii_path):
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(nii_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Decompressed: {gz_path} to {nii_path}")
                else:
                    print(f"Already exists: {nii_path}")


def nii_to_png():
    for i in range(1, 28):
        ses_id = f"ses-{i:02d}"
        anat_dir = os.path.join(base_dir, ses_id, "anat")
        png_out_dir = os.path.join(output_png_dir, ses_id, "anat")
        os.makedirs(png_out_dir, exist_ok=True)

        for file in os.listdir(anat_dir):
            if file.endswith(".nii"):
                file_path = os.path.join(anat_dir, file)
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()

                num_slices = data.shape[2]
                base_filename = file.replace(".nii", "")

                for idx in range(num_slices):
                    img_slice = data[:, :, idx]

                    # Save as png
                    png_filename = f"{base_filename}_slice_{idx:03d}.png"
                    png_path = os.path.join(png_out_dir, png_filename)

                    plt.imsave(png_path, img_slice, cmap='gray')
                print(f"Saved {num_slices} slices from {file_path}")


if __name__ == "__main__":
    decompress_nii_gz()
    nii_to_png()
