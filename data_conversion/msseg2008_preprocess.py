import os
import subprocess
import nibabel as nib
import SimpleITK as sitk
import ants
import ants.utils.bias_correction as ants_bias
from tqdm import tqdm
import numpy as np

# this file is used to convert the msseg2008 dataset to nifti format
# then remove brain skull based on FLAIR then filter to T2 and T1
# correct bias field
# sub_data = "training" or "testing"
# the final dataset is stored in data/msseg-2008-{sub_data}-nii
# the original dataset is stored in data/msseg-2008-{sub_data}
# it requires fsl and antspyx to be installed

FSLDIR = os.environ.get("FSLDIR")

datasets = ["training", "testing"]

for sub_data in datasets:
    PATH = f"data/msseg-2008-{sub_data}/"
    dir_list = []
    file_list = []
    for folder in os.listdir(PATH):
        dir_list.append(folder)
    for folder in dir_list:
        folder_path = os.path.join(PATH, folder)
        for image in os.listdir(folder_path):
            if image.endswith(".nhdr"):
                file_list.append((folder, image[:-5]))

    print(f"remove brain skull and save to folder {sub_data} with _brain.nii.gz ...")

    for image in tqdm(file_list):
        (folder, file) = image
        img = sitk.ReadImage(f"{PATH}/{folder}/{file}.nhdr")
        newfile = file + ".nii.gz"
        NEW_PATH = f"data/msseg-2008-{sub_data}-nii/{folder}"
        if not os.path.exists(NEW_PATH):
            os.makedirs(NEW_PATH)
        sitk.WriteImage(img, f"data/msseg-2008-{sub_data}-nii/{folder}/{newfile}")
        if "T1" in file:
            command = [
                f"{FSLDIR}/bin/bet",
                f"data/msseg-2008-{sub_data}-nii/{folder}/{file}",
                f"data/msseg-2008-{sub_data}-nii/{folder}/{file}_brain",
                "-R",
                "-f",
                "0.6",
                "-g",
                "0",
            ]
            subprocess.run(command, check=True)

    for image in tqdm(file_list):
        (folder, file) = image
        if "T2" in file or "FLAIR" in file:
            t1 = nib.load(
                f"data/msseg-2008-{sub_data}-nii/{folder}/{folder}_T1_brain.nii.gz"
            ).get_fdata()
            t2 = nib.load(f"data/msseg-2008-{sub_data}-nii/{folder}/{file}.nii.gz")
            t2_numpy = t2.get_fdata()
            t2_numpy = np.where(t1 > 0, t2_numpy, 0)
            final_img = nib.Nifti1Image(t2_numpy, t2.affine, t2.header)
            nib.save(
                final_img,
                f"data/msseg-2008-{sub_data}-nii/{folder}/{file}_brain.nii.gz",
            )

    print(
        f"correct bias field and save to folder {sub_data} with _brain_bias_correction.nii.gz ..."
    )
    for image in tqdm(file_list):
        (folder, file) = image
        if "lesion" not in file:
            img = ants.image_read(
                f"data/msseg-2008-{sub_data}-nii/{folder}/{file}_brain.nii.gz"
            )
            image_n4 = ants.n4_bias_field_correction(img)
            ants.image_write(
                image_n4,
                f"data/msseg-2008-{sub_data}-nii/{folder}/{file}_brain_bias_correction.nii.gz",
            )
            os.remove(f"data/msseg-2008-{sub_data}-nii/{folder}/{file}.nii.gz")
            os.remove(f"data/msseg-2008-{sub_data}-nii/{folder}/{file}_brain.nii.gz")
