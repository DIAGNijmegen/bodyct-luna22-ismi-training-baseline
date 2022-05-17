import numpy as np
import SimpleITK as sitk
from pathlib import Path
from data import (
    center_crop_volume,
    get_cross_slices_from_cube,
)

import tensorflow.keras
from tensorflow.keras.applications import VGG16

# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")

model_malignancy = VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)
model_malignancy.load_weights(
    "vgg16_malignancy_best_val_accuracy.h5",
    by_name=True,
    skip_mismatch=True,
)

# load nodule_type model
model_nodule_type = VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=3,
    classifier_activation="softmax",
)
model_nodule_type.load_weights(
    "vgg16_noduletype_best_val_accuracy.h5",
    by_name=True,
    skip_mismatch=True,
)


def clip_and_scale(
    data: np.ndarray,
    min_value: float = -1000.0,
    max_value: float = 400.0,
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data


def preprocess(img: sitk.Image, input_spacing: float) -> sitk.Image:

    # Resample image
    original_spacing_mm = img.GetSpacing()
    original_size = img.GetSize()
    new_spacing = (
        input_spacing,
        input_spacing,
        input_spacing,
    )
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(
            original_size,
            original_spacing_mm,
            new_spacing,
        )
    ]
    resampled_img = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0,
        img.GetPixelID(),
    )

    # Return image data as a numpy array
    return sitk.GetArrayFromImage(resampled_img)


image = sitk.ReadImage("LUNA22 prequel/LIDC-IDRI/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_45_211_77_0000.nii.gz")

input_size = 224

nodule_data = preprocess(image, input_spacing=0.2)
nodule_data = center_crop_volume(
    volume=nodule_data,
    crop_size=np.array(
        (
            input_size,
            input_size,
            input_size,
        )
    ),
    pad_if_too_small=True,
    pad_value=-1024,
)
nodule_data = get_cross_slices_from_cube(nodule_data)
nodule_data = clip_and_scale(nodule_data)

malignancy = model_malignancy(nodule_data[None]).numpy()[0, 1]
nodule_type = np.argmax(model_nodule_type(nodule_data[None]).numpy())

result = dict(
    malignancy_risk=round(float(malignancy), 3),
    nodule_type=int(nodule_type),
)

print(result)
