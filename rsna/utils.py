import glob
import re

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

__all__ = ["load_dicom_images_3d"]

data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"


def load_dicom_image(data_directory, img_size: int = 256, voi_lut: bool = True):
    dicom = pydicom.read_file(data_directory)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(
    scan_id,
    num_imgs: int = 64,
    img_size: int = 256,
    mri_type: str = "FLAIR",
    split: str = "train",
):

    files = sorted(
        glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"),
        key=lambda var: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
        ],
    )

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    img3d = np.stack([load_dicom_image(f) for f in files[p1:p2]]).T
    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d, n_zero), axis=-1)

    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)

    return np.expand_dims(img3d, 0)
